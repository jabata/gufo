#ifndef GUFO_SERVER_TP_PEER_STORE_HPP_
#define GUFO_SERVER_TP_PEER_STORE_HPP_

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/cache/component_resources.hpp"

namespace gufo::server {

struct TpPeerStoreOptions {
  /// Rank 1's pinned backing for checkpoint halves rank 0's cache still holds:
  /// at most this much, and with `automatic_ram` no more than the runner's
  /// measured snapshot capacity, as rank 0's cache sizes its own RAM.
  std::size_t ram_bytes{0};
  bool automatic_ram{false};
  /// Empty keeps no disk tier: halves live only while rank 0 holds them in
  /// RAM, and rank 0's disk entries cannot be restored on rank 1.
  std::filesystem::path disk_directory;
};

/// One of rank 0's disk-tier files: whether it is a private file, and its id.
using TpDiskFile = std::pair<bool, std::array<std::uint8_t, 16>>;

/// Rank 1's half of every checkpoint in a TP2 pair, under the keys rank 0's
/// cache retains instead of the bytes.
///
/// A stash copies the slot's rows or private state into pinned backing from
/// the model's own pool. Its RAM copy lives while rank 0's cache holds the key
/// in RAM: a release drops it.
///
/// The disk tier mirrors rank 0's. Rank 0 announces each of its disk files
/// that holds references (`Published`), with their keys, and each one it
/// unlinks (`Removed`). A key is written to disk once a published file refers
/// to it and deleted once none does; the file-to-key index persists, so the
/// tier survives a restart, and rank 0's startup list (`Live`) drops whatever
/// it no longer has. The tier therefore needs no eviction policy of its own.
///
/// Notices travel apart from stashes, so one may arrive before its stash; it
/// applies once the stash does. When the pool is exhausted, the least recently
/// used RAM copies of written entries go first, then other entries spill to
/// disk (or, without a disk tier, are dropped). A load of a key the store no
/// longer has fails, and rank 0's cache then drops its own copy and computes
/// the state again, so a lost half costs time, never correctness.
///
/// Thread safe. Disk writes run on one background thread; destruction
/// finishes the writes already queued.
class TpPeerStore final {
public:
  using Allocate = std::function<cache::ComponentCaptureBuffer(
      const cache::PayloadRequest&)>;
  /// Fills a stash's bytes and waits until they are complete.
  using Fill = std::function<void(std::span<std::byte>)>;
  /// Hands a load its source bytes and waits until they are consumed.
  using Consume = std::function<void(std::span<const std::byte>)>;
  /// Host memory for a load read back from disk; valid until the next call.
  using Scratch = std::function<std::span<std::byte>(std::size_t)>;

  TpPeerStore(Allocate allocate, TpPeerStoreOptions options);
  ~TpPeerStore();
  TpPeerStore(const TpPeerStore&) = delete;
  TpPeerStore& operator=(const TpPeerStore&) = delete;

  /// Stores `bytes` for `key`: rows [0, rows) of `component`, or its private
  /// state. Throws when no backing can be found or `fill` throws.
  void Stash(std::uint64_t key, const cache::PayloadRequest& request,
             std::size_t rows, const Fill& fill);
  /// Loads `key`'s rows [first, end) (whole for private state) through
  /// `consume`. Throws when the store no longer has the key, or it does not
  /// match the request.
  void Load(std::uint64_t key, cache::ComponentId component, bool is_private,
            std::size_t first, std::size_t end, const Scratch& scratch,
            const Consume& consume);
  /// Whether `key` has a RAM copy now (a load of it needs no disk read).
  [[nodiscard]] bool InRam(std::uint64_t key) const;
  /// Rank 0's cache dropped its RAM copy of `key`.
  void Release(std::uint64_t key) noexcept;
  /// Rank 0 published `file`, which refers to `keys`.
  void Published(const TpDiskFile& file, std::span<const std::uint64_t> keys);
  /// Rank 0 unlinked `file`.
  void Removed(const TpDiskFile& file);
  /// Rank 0's disk tier at its startup: every other file is gone.
  void Live(std::span<const TpDiskFile> files);
  /// Waits up to `timeout` for every queued disk write; false if some are
  /// still pending. For a pair about to exit.
  [[nodiscard]] bool Flush(std::chrono::milliseconds timeout);

  struct Stats {
    std::size_t ram_entries{0}, ram_bytes{0}, disk_entries{0}, disk_bytes{0};
    std::size_t files{0}, spilled{0}, dropped{0}, missed{0};
  };
  [[nodiscard]] Stats Snapshot() const;

private:
  struct Ram {
    cache::Payload payload;
    std::span<std::byte> bytes;
  };
  struct Entry {
    std::uint32_t component{0};
    bool is_private{false};
    std::size_t rows{0};
    std::size_t bytes{0};
    /// Present while rank 0 holds the key in RAM, until the pool needs room.
    std::shared_ptr<Ram> ram;
    bool on_disk{false};
    bool writing{false};
    /// Rank 0 dropped its RAM copy.
    bool released{false};
    /// Rank 0's published files referring to the key.
    std::uint32_t files{0};
    std::uint64_t used{0};
  };
  /// A notice that came before its stash.
  struct Early {
    bool released{false};
    std::uint32_t files{0};
  };

  [[nodiscard]] std::filesystem::path PathOf(std::uint64_t key) const;
  /// Frees the least recently used RAM copy; false when there is none.
  [[nodiscard]] bool MakeRoom();
  /// Writes an entry to disk on the caller's thread, which counted itself in
  /// `active_`, then settles what the entry still needs.
  void Write(std::uint64_t key, std::shared_ptr<Ram> ram, Entry header,
             bool drop_ram);
  [[nodiscard]] bool WriteFile(std::uint64_t key, const Entry& entry,
                               std::span<const std::byte> bytes) const;
  /// Brings an entry in line with its references and RAM copy: queues the
  /// write it needs, deletes a disk copy nothing refers to, or erases it.
  void SettleLocked(std::uint64_t key) noexcept;
  void UnlinkLocked(std::uint64_t key, Entry& entry) noexcept;
  void AddFileLocked(const TpDiskFile& file, std::vector<std::uint64_t> keys);
  void RemoveFileLocked(const TpDiskFile& file);
  void AppendIndexLocked(char op, const TpDiskFile& file,
                         std::span<const std::uint64_t> keys);
  [[nodiscard]] Early& EarlyLocked(std::uint64_t key);
  void Recover();
  void Writer();

  Allocate allocate_;
  TpPeerStoreOptions options_;
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, Entry> entries_;
  /// Rank 0's published files that refer to keys, and those keys.
  std::map<TpDiskFile, std::vector<std::uint64_t>> files_;
  /// Appended file records; rewritten whole at startup.
  std::FILE* index_{nullptr};
  std::unordered_map<std::uint64_t, Early> early_;
  std::deque<std::uint64_t> early_order_;
  static constexpr std::size_t kEarly = 1U << 16;
  std::uint64_t tick_{0};
  std::size_t ram_bytes_{0};
  std::size_t disk_bytes_{0};
  Stats counters_;
  std::condition_variable wake_;
  std::condition_variable written_;
  std::deque<std::uint64_t> writes_;
  /// Writes in progress, on the writer or a stash's thread.
  std::size_t active_{0};
  bool stopping_{false};
  std::thread writer_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TP_PEER_STORE_HPP_
