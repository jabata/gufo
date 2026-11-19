#include "src/cli/serve/tp_peer_store.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "src/cache/disk.hpp"
#include "src/cli/serve/logging.hpp"

namespace gufo::server {
namespace {

constexpr std::uint32_t kFileMagic = 0x31535054U;  // "TPS1"
constexpr std::uint32_t kFileVersion = 1;
constexpr std::string_view kFileSuffix = ".tps";
// Recover reads the file headers on up to this many threads, each taking at
// least this many files.
constexpr std::size_t kRecoverThreads = 16;
constexpr std::size_t kRecoverPerThread = 256;
constexpr std::string_view kIndexName = "files.idx";
/// Index records are op, private flag, file id, key count, keys, checksum.
constexpr std::size_t kMaxIndexKeys = std::size_t{1} << 24;

/// The fixed header in front of every disk entry's bytes.
struct FileHeader {
  std::uint32_t magic{kFileMagic};
  std::uint32_t version{kFileVersion};
  std::uint64_t key{0};
  std::uint32_t component{0};
  std::uint32_t is_private{0};
  std::uint64_t rows{0};
  std::uint64_t bytes{0};
  std::uint64_t checksum{0};
  std::array<std::uint8_t, 16> reserved{};
};
static_assert(sizeof(FileHeader) == 64);

[[nodiscard]] std::string Hex(std::uint64_t value) {
  std::array<char, 17> text{};
  std::snprintf(text.data(), text.size(), "%016llx",
                static_cast<unsigned long long>(value));
  return text.data();
}

[[nodiscard]] std::uint64_t Checksum(std::span<const std::byte> bytes) {
  return cache::DiskChecksum(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

[[nodiscard]] std::vector<std::byte> IndexRecord(
    char op, const TpDiskFile& file, std::span<const std::uint64_t> keys) {
  std::vector<std::byte> record;
  const auto append = [&](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::byte*>(data);
    record.insert(record.end(), bytes, bytes + size);
  };
  const auto flag = static_cast<std::uint8_t>(file.first ? 1 : 0);
  const auto count = static_cast<std::uint32_t>(keys.size());
  append(&op, 1);
  append(&flag, 1);
  append(file.second.data(), file.second.size());
  append(&count, sizeof(count));
  append(keys.data(), keys.size_bytes());
  const auto checksum = Checksum(record);
  append(&checksum, sizeof(checksum));
  return record;
}

}  // namespace

TpPeerStore::TpPeerStore(Allocate allocate, TpPeerStoreOptions options)
    : allocate_(std::move(allocate)), options_(std::move(options)) {
  if (!allocate_) {
    throw std::invalid_argument("TP peer store needs an allocator");
  }
  if (!options_.disk_directory.empty()) {
    options_.disk_directory /= "tp2-peer";
    Recover();
    writer_ = std::thread([this] { Writer(); });
  }
}

TpPeerStore::~TpPeerStore() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  written_.notify_all();
  if (writer_.joinable()) {
    writer_.join();
  }
  if (index_ != nullptr) {
    std::fclose(index_);
  }
}

std::filesystem::path TpPeerStore::PathOf(std::uint64_t key) const {
  return options_.disk_directory / (Hex(key) + std::string(kFileSuffix));
}

void TpPeerStore::Stash(std::uint64_t key, const cache::PayloadRequest& request,
                        std::size_t rows, const Fill& fill) {
  if (key == 0 || rows == 0 || request.bytes == 0 ||
      request.bytes % rows != 0) {
    throw std::invalid_argument("TP stash request is invalid");
  }
  for (;;) {
    cache::ComponentCaptureBuffer buffer;
    try {
      buffer = allocate_(request);
    } catch (const cache::ResourceExhausted&) {
      if (!MakeRoom()) {
        throw;
      }
      continue;
    } catch (const std::bad_alloc&) {
      if (!MakeRoom()) {
        throw;
      }
      continue;
    }
    if (buffer.bytes.size() < request.bytes || !buffer.finish) {
      throw std::logic_error("TP stash backing is smaller than its request");
    }
    const auto bytes = buffer.bytes.first(request.bytes);
    fill(bytes);
    auto ram = std::make_shared<Ram>(Ram{buffer.finish(), bytes});
    const std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.contains(key)) {
      throw std::logic_error("TP stash key " + Hex(key) + " is already held");
    }
    Entry entry{.component = request.component.value,
                .is_private =
                    request.category == cache::ResourceCategory::kPrivateState,
                .rows = rows,
                .bytes = request.bytes,
                .ram = std::move(ram),
                .used = ++tick_};
    // Notices travel on another connection and may have come first.
    if (const auto early = early_.find(key); early != early_.end()) {
      entry.released = early->second.released;
      entry.files = early->second.files;
      early_.erase(early);
    }
    ram_bytes_ += request.bytes;
    entries_.emplace(key, std::move(entry));
    SettleLocked(key);
    return;
  }
}

void TpPeerStore::SettleLocked(std::uint64_t key) noexcept {
  const auto found = entries_.find(key);
  if (found == entries_.end()) {
    return;
  }
  auto& entry = found->second;
  if (entry.writing) {
    return;  // the write settles it when done
  }
  const bool disk = !options_.disk_directory.empty();
  // Rank 0 refers to the key from RAM, or (with a disk tier) from a file.
  const bool referred = !entry.released || (disk && entry.files != 0);
  if (!referred) {
    if (entry.on_disk) {
      UnlinkLocked(key, entry);
    }
    if (entry.ram != nullptr) {
      ram_bytes_ -= entry.bytes;
    }
    entries_.erase(found);
    return;
  }
  // A file refers to it, or its RAM copy spilled under pool pressure.
  const bool wants_disk = disk && (entry.files != 0 || entry.ram == nullptr);
  if (wants_disk && !entry.on_disk) {
    if (entry.ram == nullptr) {
      entries_.erase(found);  // lost: its loads will miss
      ++counters_.dropped;
      return;
    }
    entry.writing = true;
    writes_.push_back(key);
    wake_.notify_one();
    return;
  }
  if (!wants_disk && entry.on_disk) {
    UnlinkLocked(key, entry);
  }
  if (entry.released && entry.ram != nullptr && entry.on_disk) {
    ram_bytes_ -= entry.bytes;
    entry.ram.reset();
  }
}

void TpPeerStore::UnlinkLocked(std::uint64_t key, Entry& entry) noexcept {
  std::error_code ignored;
  std::filesystem::remove(PathOf(key), ignored);
  disk_bytes_ -= entry.bytes;
  entry.on_disk = false;
}

bool TpPeerStore::MakeRoom() {
  std::unique_lock<std::mutex> lock(mutex_);
  std::uint64_t victim = 0;
  std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
  bool written = false;
  bool writing = false;
  // Written entries' RAM copies first, then the least recently used.
  for (const auto& [key, entry] : entries_) {
    if (entry.ram == nullptr) {
      continue;
    }
    if (entry.writing) {
      writing = true;
      continue;
    }
    if (std::pair{!entry.on_disk, entry.used} < std::pair{!written, oldest}) {
      written = entry.on_disk;
      oldest = entry.used;
      victim = key;
    }
  }
  if (victim == 0) {
    if (!writing || stopping_) {
      return false;
    }
    written_.wait(lock);
    return true;
  }
  auto& entry = entries_.at(victim);
  if (entry.on_disk) {
    ram_bytes_ -= entry.bytes;
    entry.ram.reset();
    ++counters_.spilled;
    return true;
  }
  if (options_.disk_directory.empty()) {
    ram_bytes_ -= entry.bytes;
    entries_.erase(victim);
    ++counters_.dropped;
    return true;
  }
  entry.writing = true;
  ++active_;
  ++counters_.spilled;
  auto ram = entry.ram;
  auto header = entry;
  lock.unlock();
  Write(victim, std::move(ram), std::move(header), true);
  return true;
}

void TpPeerStore::Write(std::uint64_t key, std::shared_ptr<Ram> ram,
                        Entry header, bool drop_ram) {
  // Only `ram` holds the bytes while they are written: a copy in the header
  // (moved in by both callers) would keep the block allocated after Flush
  // reports the write done.
  header.ram.reset();
  const bool written = WriteFile(key, header, ram->bytes);
  ram.reset();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    --active_;
    const auto found = entries_.find(key);
    if (found == entries_.end()) {
      if (written) {
        std::error_code ignored;
        std::filesystem::remove(PathOf(key), ignored);
      }
    } else {
      auto& entry = found->second;
      entry.writing = false;
      if (written) {
        entry.on_disk = true;
        disk_bytes_ += entry.bytes;
        if (drop_ram && entry.ram != nullptr) {
          ram_bytes_ -= entry.bytes;
          entry.ram.reset();
        }
        SettleLocked(key);
      } else if (entry.released || drop_ram) {
        // Nothing more to try: a later load of it misses.
        if (entry.ram != nullptr) {
          ram_bytes_ -= entry.bytes;
        }
        entries_.erase(found);
        ++counters_.dropped;
      }
    }
  }
  written_.notify_all();
}

bool TpPeerStore::WriteFile(std::uint64_t key, const Entry& entry,
                            std::span<const std::byte> bytes) const {
  const auto path = PathOf(key);
  auto temporary = path;
  temporary += ".tmp";
  FileHeader header{.key = key,
                    .component = entry.component,
                    .is_private = entry.is_private ? 1U : 0U,
                    .rows = entry.rows,
                    .bytes = entry.bytes,
                    .checksum = Checksum(bytes)};
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      Logger::Warn("tp", "event=tp_peer_store_write_failed key=" + Hex(key));
      return false;
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    Logger::Warn("tp", "event=tp_peer_store_write_failed key=" + Hex(key));
    return false;
  }
  return true;
}

void TpPeerStore::Load(std::uint64_t key, cache::ComponentId component,
                       bool is_private, std::size_t first, std::size_t end,
                       const Scratch& scratch, const Consume& consume) {
  std::shared_ptr<Ram> ram;
  Entry header;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found == entries_.end()) {
      ++counters_.missed;
      throw std::runtime_error("TP stash key " + Hex(key) +
                               " is no longer stored");
    }
    auto& entry = found->second;
    if (entry.component != component.value || entry.is_private != is_private ||
        (!is_private && (first >= end || end > entry.rows))) {
      throw std::runtime_error("TP stash key " + Hex(key) +
                               " does not match the load");
    }
    entry.used = ++tick_;
    ram = entry.ram;
    header = entry;
  }
  const std::size_t row_bytes = header.bytes / header.rows;
  const std::size_t offset = is_private ? 0 : first * row_bytes;
  const std::size_t length =
      is_private ? header.bytes : (end - first) * row_bytes;
  if (ram != nullptr) {
    consume(std::span<const std::byte>(ram->bytes).subspan(offset, length));
    return;
  }
  if (!header.on_disk) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      ++counters_.missed;
    }
    throw std::runtime_error("TP stash key " + Hex(key) + " has no copy");
  }
  // Read and verify the whole entry: a partial load still checks every byte
  // its checksum covers.
  const auto bytes = scratch(header.bytes);
  FileHeader file{};
  std::ifstream in(PathOf(key), std::ios::binary);
  in.read(reinterpret_cast<char*>(&file), sizeof(file));
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(header.bytes));
  if (!in || file.magic != kFileMagic || file.version != kFileVersion ||
      file.key != key || file.bytes != header.bytes ||
      file.checksum != Checksum(bytes.first(header.bytes))) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      const auto found = entries_.find(key);
      if (found != entries_.end() && found->second.on_disk &&
          found->second.ram == nullptr && !found->second.writing) {
        UnlinkLocked(key, found->second);
        entries_.erase(found);
      }
      ++counters_.missed;
    }
    throw std::runtime_error("TP stash key " + Hex(key) +
                             " failed its disk check");
  }
  consume(std::span<const std::byte>(bytes).subspan(offset, length));
}

TpPeerStore::Early& TpPeerStore::EarlyLocked(std::uint64_t key) {
  auto [found, inserted] = early_.try_emplace(key);
  if (inserted) {
    // A key whose stash never comes (it failed) ages out.
    early_order_.push_back(key);
    if (early_order_.size() > kEarly) {
      early_.erase(early_order_.front());
      early_order_.pop_front();
    }
  }
  return early_.at(key);
}

void TpPeerStore::Release(std::uint64_t key) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = entries_.find(key);
  if (found == entries_.end()) {
    try {
      EarlyLocked(key).released = true;
    } catch (...) {
    }
    return;
  }
  found->second.released = true;
  SettleLocked(key);
}

void TpPeerStore::Published(const TpDiskFile& file,
                            std::span<const std::uint64_t> keys) {
  std::vector<std::uint64_t> distinct(keys.begin(), keys.end());
  std::ranges::sort(distinct);
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  const std::lock_guard<std::mutex> lock(mutex_);
  if (options_.disk_directory.empty() || files_.contains(file)) {
    return;
  }
  AppendIndexLocked('P', file, distinct);
  AddFileLocked(file, std::move(distinct));
}

void TpPeerStore::AddFileLocked(const TpDiskFile& file,
                                std::vector<std::uint64_t> keys) {
  for (const auto key : keys) {
    if (const auto found = entries_.find(key); found != entries_.end()) {
      ++found->second.files;
      SettleLocked(key);
    } else {
      ++EarlyLocked(key).files;
    }
  }
  files_.emplace(file, std::move(keys));
}

void TpPeerStore::Removed(const TpDiskFile& file) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!files_.contains(file)) {
    return;
  }
  AppendIndexLocked('R', file, {});
  RemoveFileLocked(file);
}

void TpPeerStore::RemoveFileLocked(const TpDiskFile& file) {
  const auto found = files_.find(file);
  if (found == files_.end()) {
    return;
  }
  const auto keys = std::move(found->second);
  files_.erase(found);
  for (const auto key : keys) {
    if (const auto entry = entries_.find(key); entry != entries_.end()) {
      if (entry->second.files != 0) {
        --entry->second.files;
      }
      SettleLocked(key);
    } else if (const auto early = early_.find(key); early != early_.end()) {
      if (early->second.files != 0) {
        --early->second.files;
      }
    }
  }
}

void TpPeerStore::Live(std::span<const TpDiskFile> files) {
  const std::set<TpDiskFile> live(files.begin(), files.end());
  const std::lock_guard<std::mutex> lock(mutex_);
  std::vector<TpDiskFile> gone;
  for (const auto& [file, keys] : files_) {
    if (!live.contains(file)) {
      gone.push_back(file);
    }
  }
  for (const auto& file : gone) {
    AppendIndexLocked('R', file, {});
    RemoveFileLocked(file);
  }
  Logger::Info("tp", "event=tp_peer_store_reconciled live_files=" +
                         std::to_string(live.size()) +
                         " dropped_files=" + std::to_string(gone.size()) +
                         " entries=" + std::to_string(entries_.size()));
}

void TpPeerStore::AppendIndexLocked(char op, const TpDiskFile& file,
                                    std::span<const std::uint64_t> keys) {
  if (index_ == nullptr) {
    return;
  }
  const auto record = IndexRecord(op, file, keys);
  // A lost record costs a miss after a crash, never a wrong restore: the
  // halves carry their own checksums and rank 0 validates what it restores.
  if (std::fwrite(record.data(), 1, record.size(), index_) != record.size() ||
      std::fflush(index_) != 0) {
    Logger::Warn("tp", "event=tp_peer_store_index_write_failed");
  }
}

bool TpPeerStore::Flush(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  return written_.wait_for(lock, timeout,
                           [&] { return writes_.empty() && active_ == 0; });
}

void TpPeerStore::Writer() {
  for (;;) {
    std::uint64_t key = 0;
    std::shared_ptr<Ram> ram;
    Entry header;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || !writes_.empty(); });
      // Queued writes finish before the store goes: rank 0's disk entries
      // may refer to them after a restart.
      if (writes_.empty()) {
        return;
      }
      key = writes_.front();
      writes_.pop_front();
      const auto found = entries_.find(key);
      if (found == entries_.end()) {
        continue;
      }
      if (found->second.ram == nullptr) {
        found->second.writing = false;
        SettleLocked(key);
        written_.notify_all();
        continue;
      }
      ram = found->second.ram;
      header = found->second;
      ++active_;
    }
    Write(key, std::move(ram), std::move(header), false);
  }
}

void TpPeerStore::Recover() {
  std::error_code error;
  std::filesystem::create_directories(options_.disk_directory, error);
  if (error) {
    throw std::runtime_error("cannot create TP peer store directory " +
                             options_.disk_directory.string() + ": " +
                             error.message());
  }
  // The file index first: which keys rank 0's files still refer to.
  const auto index_path = options_.disk_directory / kIndexName;
  std::size_t records = 0;
  {
    std::ifstream in(index_path, std::ios::binary);
    for (;;) {
      std::array<char, 2> head{};
      TpDiskFile file;
      std::uint32_t count = 0;
      if (!in.read(head.data(), 2) ||
          !in.read(reinterpret_cast<char*>(file.second.data()),
                   file.second.size()) ||
          !in.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
          count > kMaxIndexKeys) {
        break;
      }
      std::vector<std::uint64_t> keys(count);
      std::uint64_t checksum = 0;
      if (!in.read(
              reinterpret_cast<char*>(keys.data()),
              static_cast<std::streamsize>(count * sizeof(std::uint64_t))) ||
          !in.read(reinterpret_cast<char*>(&checksum), sizeof(checksum))) {
        break;
      }
      file.first = head[1] != 0;
      const auto record = IndexRecord(head[0], file, keys);
      std::uint64_t expected = 0;
      std::memcpy(&expected, record.data() + record.size() - sizeof(expected),
                  sizeof(expected));
      // A torn tail ends the index.
      if (expected != checksum || (head[0] != 'P' && head[0] != 'R')) {
        break;
      }
      ++records;
      if (head[0] == 'P') {
        files_[file] = std::move(keys);
      } else {
        files_.erase(file);
      }
    }
  }
  std::unordered_map<std::uint64_t, std::uint32_t> referred;
  for (const auto& [file, keys] : files_) {
    for (const auto key : keys) {
      ++referred[key];
    }
  }
  // Then the halves those keys name; any other file is unreferenced. A full
  // disk tier holds one file per half (a hundred thousand or more), and
  // rank 0's first request waits for this scan, so the header reads run on
  // several threads: on a cold page cache each one is a separate device read.
  struct Found {
    std::uint64_t key;
    Entry entry;
    std::filesystem::file_time_type time;
  };
  std::vector<std::filesystem::path> paths;
  std::size_t discarded = 0;
  for (const auto& item :
       std::filesystem::directory_iterator(options_.disk_directory, error)) {
    const auto& path = item.path();
    if (path.filename() == kIndexName) {
      continue;
    }
    paths.push_back(path);
  }
  std::vector<std::optional<Found>> scanned(paths.size());
  const auto scan = [&](std::size_t i) {
    const auto& path = paths[i];
    std::error_code file_error;
    FileHeader header{};
    bool valid = false;
    if (path.extension() == kFileSuffix &&
        std::filesystem::is_regular_file(path, file_error)) {
      std::ifstream in(path, std::ios::binary);
      in.read(reinterpret_cast<char*>(&header), sizeof(header));
      valid = static_cast<bool>(in) && header.magic == kFileMagic &&
              header.version == kFileVersion && header.key != 0 &&
              header.rows != 0 && header.bytes % header.rows == 0 &&
              std::filesystem::file_size(path, file_error) ==
                  sizeof(header) + header.bytes &&
              !file_error && path.stem() == Hex(header.key) &&
              referred.contains(header.key);
    }
    if (!valid) {
      std::filesystem::remove(path, file_error);
      return;
    }
    const auto time = std::filesystem::last_write_time(path, file_error);
    scanned[i] = Found{header.key,
                       Entry{.component = header.component,
                             .is_private = header.is_private != 0,
                             .rows = header.rows,
                             .bytes = header.bytes,
                             .on_disk = true,
                             .released = true,
                             .files = referred.at(header.key)},
                       time};
  };
  {
    const std::size_t workers = std::min<std::size_t>(
        kRecoverThreads,
        (paths.size() + kRecoverPerThread - 1) / kRecoverPerThread);
    std::atomic<std::size_t> next{0};
    std::mutex failure_mutex;
    std::exception_ptr failure;
    const auto work = [&] {
      try {
        for (std::size_t i = next++; i < paths.size(); i = next++) {
          scan(i);
        }
      } catch (...) {
        next = paths.size();
        const std::lock_guard<std::mutex> lock(failure_mutex);
        if (!failure) {
          failure = std::current_exception();
        }
      }
    };
    std::vector<std::thread> threads;
    for (std::size_t t = 1; t < workers; ++t) {
      threads.emplace_back(work);
    }
    work();
    for (auto& thread : threads) {
      thread.join();
    }
    if (failure) {
      std::rethrow_exception(failure);
    }
  }
  std::vector<Found> found;
  found.reserve(paths.size());
  for (auto& item : scanned) {
    if (item) {
      found.push_back(std::move(*item));
    } else {
      ++discarded;
    }
  }
  std::ranges::sort(found, {}, &Found::time);
  for (auto& item : found) {
    item.entry.used = ++tick_;
    disk_bytes_ += item.entry.bytes;
    entries_.emplace(item.key, std::move(item.entry));
  }
  // Rewrite the index whole, then append to it.
  auto temporary = index_path;
  temporary += ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    for (const auto& [file, keys] : files_) {
      const auto record = IndexRecord('P', file, keys);
      out.write(reinterpret_cast<const char*>(record.data()),
                static_cast<std::streamsize>(record.size()));
    }
    out.close();
    if (!out) {
      throw std::runtime_error("cannot write TP peer store index " +
                               temporary.string());
    }
  }
  std::filesystem::rename(temporary, index_path, error);
  if (error) {
    throw std::runtime_error("cannot write TP peer store index " +
                             index_path.string() + ": " + error.message());
  }
  index_ = std::fopen(index_path.c_str(), "ab");
  if (index_ == nullptr) {
    throw std::runtime_error("cannot open TP peer store index " +
                             index_path.string());
  }
  Logger::Info("tp", "event=tp_peer_store_recovered entries=" +
                         std::to_string(entries_.size()) +
                         " bytes=" + std::to_string(disk_bytes_) +
                         " files=" + std::to_string(files_.size()) +
                         " records=" + std::to_string(records) +
                         " discarded=" + std::to_string(discarded));
}

TpPeerStore::Stats TpPeerStore::Snapshot() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  Stats stats = counters_;
  for (const auto& [key, entry] : entries_) {
    if (entry.ram != nullptr) {
      ++stats.ram_entries;
    }
    if (entry.on_disk) {
      ++stats.disk_entries;
    }
  }
  stats.ram_bytes = ram_bytes_;
  stats.disk_bytes = disk_bytes_;
  stats.files = files_.size();
  return stats;
}

}  // namespace gufo::server
