// Rank 1's store of its half of rank 0's checkpoints (TpPeerStore): stash and
// load, the disk tier mirroring rank 0's published and removed files across a
// restart, notices before their stash, corrupt entries, and pool pressure.

#include "src/cli/serve/tp_peer_store.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/cache/ledger.hpp"

namespace {

using gufo::cache::ComponentId;
using gufo::cache::PayloadRequest;
using gufo::cache::ResourceCategory;
using gufo::server::TpPeerStore;
using gufo::server::TpPeerStoreOptions;

void Require(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    std::exit(1);
  }
}

/// A pool of `blocks` blocks of `block_bytes`, committed up front like the
/// model providers' pinned backing.
class Pool {
public:
  Pool(std::size_t blocks, std::size_t block_bytes)
      : ledger_({1U << 30, 1U << 30, 1U << 20, 1U << 30}),
        data_(std::make_shared<std::vector<std::byte>>(blocks * block_bytes)) {
    for (std::size_t i = 0; i < blocks; ++i) {
      auto reservation =
          ledger_.Reserve(ResourceCategory::kBackingFree, block_bytes);
      const auto bytes =
          std::span(*data_).subspan(i * block_bytes, block_bytes);
      blocks_.push_back({reservation.Convert(), bytes,
                         std::shared_ptr<const void>(data_, bytes.data())});
    }
  }
  [[nodiscard]] TpPeerStore::Allocate Allocator() {
    return [this](const PayloadRequest& request) {
      for (auto& block : blocks_) {
        if (request.bytes > block.bytes.size() ||
            block.pool.Info().category != ResourceCategory::kBackingFree) {
          continue;
        }
        try {
          auto reservation = std::make_shared<gufo::cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          return gufo::cache::ComponentCaptureBuffer{
              block.bytes,
              [reservation, owner = block.owner] {
                return gufo::cache::Payload::Committed(reservation->Convert(),
                                                       owner);
              },
              data_.get(),
              {}};
        } catch (const std::bad_alloc&) {
        }
      }
      throw gufo::cache::ResourceExhausted();
    };
  }
  [[nodiscard]] std::size_t Free() const {
    return static_cast<std::size_t>(
        std::ranges::count_if(blocks_, [](const auto& block) {
          return block.pool.Info().category == ResourceCategory::kBackingFree;
        }));
  }

private:
  struct Block {
    gufo::cache::ResourceCharge pool;
    std::span<std::byte> bytes;
    std::shared_ptr<const void> owner;
  };
  gufo::cache::ResourceLedger ledger_;
  std::shared_ptr<std::vector<std::byte>> data_;
  std::vector<Block> blocks_;
};

constexpr std::size_t kRowBytes = 8;

[[nodiscard]] PayloadRequest Rows(std::size_t rows) {
  return {ComponentId{3}, ResourceCategory::kBackingAssigned, 0, rows,
          rows * kRowBytes};
}

/// Rows whose bytes encode the key and the row, so a load proves its source.
void Stash(TpPeerStore& store, std::uint64_t key, std::size_t rows) {
  store.Stash(key, Rows(rows), rows, [&](std::span<std::byte> bytes) {
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<std::byte>((key * 31 + i) & 0xff);
    }
  });
}

/// Loads rows [first, end) of `key` and checks them; false when it fails.
[[nodiscard]] bool Loads(TpPeerStore& store, std::uint64_t key,
                         std::size_t first, std::size_t end) {
  std::vector<std::byte> scratch;
  bool exact = false;
  try {
    store.Load(
        key, ComponentId{3}, false, first, end,
        [&](std::size_t bytes) {
          scratch.resize(bytes);
          return std::span(scratch);
        },
        [&](std::span<const std::byte> bytes) {
          exact = bytes.size() == (end - first) * kRowBytes;
          for (std::size_t i = 0; exact && i < bytes.size(); ++i) {
            exact = bytes[i] == static_cast<std::byte>(
                                    (key * 31 + first * kRowBytes + i) & 0xff);
          }
        });
  } catch (const std::runtime_error&) {
    return false;
  }
  return exact;
}

[[nodiscard]] std::filesystem::path TemporaryRoot(const std::string& name) {
  auto root = std::filesystem::temp_directory_path() /
              ("tp-peer-store-" + name + "-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  return root;
}

/// One of rank 0's disk files.
[[nodiscard]] gufo::server::TpDiskFile File(std::uint8_t id,
                                            bool is_private = false) {
  gufo::server::TpDiskFile file{is_private, {}};
  file.second[0] = id;
  return file;
}

void Publish(TpPeerStore& store, std::uint8_t file,
             std::vector<std::uint64_t> keys) {
  store.Published(File(file), keys);
}

void Settle(TpPeerStore& store) {
  Require(store.Flush(std::chrono::seconds(10)), "disk writes finish");
}

[[nodiscard]] std::size_t Halves(const std::filesystem::path& root) {
  std::size_t count = 0;
  for (const auto& item :
       std::filesystem::directory_iterator(root / "tp2-peer")) {
    count += item.path().extension() == ".tps";
  }
  return count;
}

void TestRamOnly() {
  Pool pool(4, 256);
  TpPeerStore store(pool.Allocator(), {});
  Stash(store, 11, 8);
  Require(Loads(store, 11, 0, 8) && Loads(store, 11, 2, 5),
          "a stash loads whole and in part");
  Require(pool.Free() == 3, "a stash holds one block");
  Publish(store, 1, {11});
  store.Release(11);
  Require(pool.Free() == 4 && !Loads(store, 11, 0, 8),
          "without a disk tier a release frees the entry");
  bool rejected = false;
  try {
    Stash(store, 12, 4);
    Stash(store, 12, 4);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Require(rejected, "a key is stored once");
}

void TestMirror() {
  const auto root = TemporaryRoot("mirror");
  const TpPeerStoreOptions options{.disk_directory = root};
  {
    Pool pool(4, 256);
    TpPeerStore store(pool.Allocator(), options);
    Stash(store, 21, 8);
    Stash(store, 22, 4);
    Settle(store);
    Require(store.Snapshot().disk_entries == 0 && Halves(root) == 0,
            "a half no published file names stays in RAM only");
    Publish(store, 1, {21});
    Settle(store);
    Require(store.Snapshot().disk_entries == 1 && pool.Free() == 2,
            "a published half is written and keeps its RAM copy while held");
    Require(store.InRam(21), "a held half has a RAM copy");
    store.Release(21);
    Require(pool.Free() == 3 && !store.InRam(21) && Loads(store, 21, 1, 7),
            "a release frees the RAM copy; the half loads from disk");
    store.Release(22);
    Require(pool.Free() == 4 && !Loads(store, 22, 0, 4),
            "a released half no file names is gone");
  }
  {
    Pool pool(4, 256);
    TpPeerStore store(pool.Allocator(), options);
    Require(Loads(store, 21, 0, 8) && store.Snapshot().files == 1,
            "a restart recovers the index and its halves");
    store.Removed(File(1));
    Require(!Loads(store, 21, 0, 8) && Halves(root) == 0,
            "a half goes with the last file that names it");
  }
  std::filesystem::remove_all(root);
}

void TestSharedAndLive() {
  const auto root = TemporaryRoot("live");
  const TpPeerStoreOptions options{.disk_directory = root};
  {
    Pool pool(4, 256);
    TpPeerStore store(pool.Allocator(), options);
    Stash(store, 31, 4);
    Stash(store, 32, 4);
    Stash(store, 33, 4);
    Publish(store, 3, {31, 33});
    Publish(store, 4, {32, 33});
    for (const auto key : {31, 32, 33}) {
      store.Release(key);
    }
    Settle(store);
    store.Removed(File(3));
    Require(!Loads(store, 31, 0, 4) && Loads(store, 33, 0, 4),
            "a half two files name outlives one of them");
  }
  // The index tolerates a torn tail.
  {
    std::ofstream index(root / "tp2-peer" / "files.idx",
                        std::ios::binary | std::ios::app);
    index << "P\x01torn";
  }
  {
    Pool pool(4, 256);
    TpPeerStore store(pool.Allocator(), options);
    Require(Loads(store, 32, 0, 4) && Loads(store, 33, 0, 4),
            "a torn index tail keeps the records before it");
    const std::array live{File(9)};
    store.Live(live);
    Require(!Loads(store, 32, 0, 4) && !Loads(store, 33, 0, 4) &&
                Halves(root) == 0 && store.Snapshot().files == 0,
            "rank 0's startup list drops every file it no longer has");
  }
  std::filesystem::remove_all(root);
}

void TestEarlyNotices() {
  {
    // Notices travel apart from stashes and may arrive first.
    Pool pool(4, 256);
    TpPeerStore store(pool.Allocator(), {});
    store.Release(71);
    Stash(store, 71, 4);
    Require(pool.Free() == 4 && !Loads(store, 71, 0, 4),
            "a release that came first frees its stash");
  }
  {
    const auto root = TemporaryRoot("early");
    Pool pool(4, 256);
    TpPeerStore store(pool.Allocator(), {.disk_directory = root});
    Publish(store, 7, {72});
    store.Release(72);
    Stash(store, 72, 4);
    Settle(store);
    Require(pool.Free() == 4 && Loads(store, 72, 0, 4),
            "a publication and a release that came first keep it on disk");
    std::filesystem::remove_all(root);
  }
}

void TestCorruptEntry() {
  const auto root = TemporaryRoot("corrupt");
  Pool pool(4, 256);
  TpPeerStore store(pool.Allocator(), {.disk_directory = root});
  Stash(store, 61, 8);
  Publish(store, 6, {61});
  store.Release(61);
  Settle(store);
  for (const auto& item :
       std::filesystem::directory_iterator(root / "tp2-peer")) {
    if (item.path().extension() != ".tps") {
      continue;
    }
    std::fstream file(item.path(),
                      std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(64 + 3);
    file.put('\x5a');
  }
  Require(!Loads(store, 61, 0, 8), "a corrupt disk entry is refused");
  Require(store.Snapshot().disk_entries == 0 && Halves(root) == 0,
          "a corrupt disk entry is removed");
  std::filesystem::remove_all(root);
}

void TestPressure() {
  {
    // Without a disk tier the least recently used entry is dropped.
    Pool pool(2, 256);
    TpPeerStore store(pool.Allocator(), {});
    Stash(store, 41, 4);
    Stash(store, 42, 4);
    Require(Loads(store, 41, 0, 4), "touch the older entry");
    Stash(store, 43, 4);
    Require(Loads(store, 41, 0, 4) && !Loads(store, 42, 0, 4) &&
                Loads(store, 43, 0, 4),
            "pressure drops the least recently used entry");
    Require(store.Snapshot().dropped == 1, "the drop is counted");
  }
  {
    // With one, a held half spills to disk, loads, and goes on release.
    const auto root = TemporaryRoot("pressure");
    Pool pool(2, 256);
    TpPeerStore store(pool.Allocator(), {.disk_directory = root});
    Stash(store, 51, 4);
    Stash(store, 52, 4);
    Stash(store, 53, 4);
    Require(Loads(store, 51, 0, 4) && Loads(store, 52, 0, 4) &&
                Loads(store, 53, 0, 4) && store.Snapshot().spilled == 1,
            "pressure spills a held half to disk, where it still loads");
    store.Release(51);
    Require(Halves(root) == 0 && !Loads(store, 51, 0, 4),
            "a spilled half no file names goes on release");
    std::filesystem::remove_all(root);
  }
}

void TestRecoverMany() {
  // Enough halves that the restart scans them on several threads, plus a
  // corrupt half and a stray file, which the scan removes.
  constexpr std::uint64_t kHalves = 1200;
  const auto root = TemporaryRoot("many");
  const TpPeerStoreOptions options{.disk_directory = root};
  {
    Pool pool(kHalves, 64);
    TpPeerStore store(pool.Allocator(), options);
    std::vector<std::uint64_t> keys;
    for (std::uint64_t key = 1; key <= kHalves; ++key) {
      Stash(store, key, 4);
      keys.push_back(key);
    }
    Publish(store, 7, keys);
    Settle(store);
    for (const auto key : keys) {
      store.Release(key);
    }
    Require(store.Snapshot().disk_entries == kHalves && Halves(root) == kHalves,
            "every published half is on disk");
  }
  std::ofstream(root / "tp2-peer" / "ffffffffffff0001.tps") << "torn";
  std::ofstream(root / "tp2-peer" / "stray.tmp") << "stray";
  {
    Pool pool(4, 64);
    TpPeerStore store(pool.Allocator(), options);
    Require(store.Snapshot().disk_entries == kHalves &&
                Halves(root) == kHalves &&
                !std::filesystem::exists(root / "tp2-peer" / "stray.tmp"),
            "a restart recovers every half and removes the others");
    bool all = true;
    for (std::uint64_t key = 1; key <= kHalves && all; ++key) {
      all = Loads(store, key, 1, 4);
    }
    Require(all, "every recovered half loads");
  }
  std::filesystem::remove_all(root);
}

}  // namespace

int main() {
  TestRamOnly();
  TestMirror();
  TestSharedAndLive();
  TestEarlyNotices();
  TestCorruptEntry();
  TestPressure();
  TestRecoverMany();
  std::cout << "PASS: TP peer store keeps rank 1's checkpoint halves\n";
  return 0;
}
