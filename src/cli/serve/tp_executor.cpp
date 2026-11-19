#include "src/cli/serve/tp_executor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include "src/cache/disk.hpp"
#include "src/cache/ledger.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/response_format.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {

constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
/// Recorded in place of a result when a call throws.
constexpr std::uint64_t kFailureMarker = 0xdead'c0de'0000'0001ULL;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

[[nodiscard]] std::string_view OpName(TpInstructionOp op) {
  switch (op) {
    case TpInstructionOp::kInvalidate:
      return "reset";
    case TpInstructionOp::kPrefill:
      return "prefill";
    case TpInstructionOp::kAdvance:
      return "advance";
    case TpInstructionOp::kDecode:
      return "decode";
    case TpInstructionOp::kReuse:
      return "reuse";
    case TpInstructionOp::kCancelPrepare:
      return "cancel-prepare";
    case TpInstructionOp::kAdvanceBatch:
      return "batched advance";
    case TpInstructionOp::kPromptContext:
      return "prompt context";
    case TpInstructionOp::kBeginRestore:
      return "begin-restore";
    case TpInstructionOp::kValidate:
      return "validate";
    case TpInstructionOp::kReconcile:
      return "reconcile";
    case TpInstructionOp::kStash:
      return "stash";
    case TpInstructionOp::kLoadStashed:
      return "load-stashed";
    case TpInstructionOp::kDecodeBatch:
      return "batched decode";
    case TpInstructionOp::kEnd:
      return "end";
    case TpInstructionOp::kNone:
      break;
  }
  return "unknown";
}

[[nodiscard]] std::string Hex(std::uint64_t value) {
  char text[19];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(value));
  return text;
}

[[nodiscard]] std::uint32_t ToWire(std::size_t value, const char* what) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string("TP instruction ") + what +
                              " exceeds the protocol range");
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::int32_t TokenToWire(TextRunnerToken token) {
  if (token >
      static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 token exceeds the instruction range");
  }
  return static_cast<std::int32_t>(token);
}

/// A batch member is recorded in its request exactly like the advance or
/// decode it replaces, so both ranks agree on a request's calls however they
/// are batched.
[[nodiscard]] TpInstruction MemberCall(TpInstructionOp op,
                                       const TpBatchMember& member) {
  if (op == TpInstructionOp::kDecodeBatch) {
    return {.op = TpInstructionOp::kDecode,
            .state = member.state,
            .count = member.count,
            .rng = member.rng,
            .pending = member.pending};
  }
  return {.op = TpInstructionOp::kAdvance,
          .state = member.state,
          .token = member.token};
}

/// The sampler's draw state as a decode instruction carries it.
[[nodiscard]] std::int32_t PendingToWire(
    const sampling::SamplerState::DrawState& draw) {
  if (!draw.pending.has_value()) {
    return -1;
  }
  if (*draw.pending > static_cast<sampling::TokenId>(
                          std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 pending draw exceeds the protocol range");
  }
  return static_cast<std::int32_t>(*draw.pending);
}

[[nodiscard]] sampling::SamplerState::DrawState DrawFromWire(
    std::uint64_t rng, std::int32_t pending) {
  return {.rng = rng,
          .pending = pending >= 0 ? std::optional<sampling::TokenId>(
                                        static_cast<sampling::TokenId>(pending))
                                  : std::nullopt};
}

/// Both ranks record which prompt context a state was bound to, so a request
/// whose images reached rank 1 differently fails the agreement check.
void DigestTpContext(TpExecutionDigest& digest,
                     const std::shared_ptr<const TextPromptContext>& context) {
  const std::span<const std::uint8_t> identity =
      context != nullptr
          ? std::span<const std::uint8_t>(context->cache_identity)
          : std::span<const std::uint8_t>{};
  digest.Add(identity.size());
  for (const auto byte : identity) {
    digest.Add(byte);
  }
}

/// A rank's component descriptors, as rank 1 reports them when the pair forms.
[[nodiscard]] std::vector<std::byte> EncodeDescriptors(
    std::span<const cache::ComponentDescriptor> components) {
  std::vector<std::byte> out;
  const auto put = [&](std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
      out.push_back(static_cast<std::byte>(value >> (8 * i)));
    }
  };
  put(components.size());
  for (const auto& d : components) {
    put(d.id.value);
    put(d.layout_version);
    put(static_cast<std::uint64_t>(d.kind));
    put(d.row_bytes);
    put(d.rows_per_chunk);
    put(d.state_bytes);
    put(d.window_rows);
  }
  return out;
}

[[nodiscard]] std::vector<cache::ComponentDescriptor> DecodeDescriptors(
    std::span<const std::byte> bytes) {
  std::size_t offset = 0;
  const auto get = [&] {
    if (bytes.size() - offset < 8) {
      throw std::runtime_error("TP rank 1 sent truncated cache components");
    }
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
      value |= static_cast<std::uint64_t>(bytes[offset++]) << (8 * i);
    }
    return value;
  };
  const auto count = get();
  if (count > bytes.size() / 56) {
    throw std::runtime_error("TP rank 1 sent invalid cache components");
  }
  std::vector<cache::ComponentDescriptor> out(count);
  for (auto& d : out) {
    d.id.value = static_cast<std::uint32_t>(get());
    d.layout_version = static_cast<std::uint32_t>(get());
    d.kind = static_cast<cache::ComponentKind>(get());
    d.row_bytes = get();
    d.rows_per_chunk = get();
    d.state_bytes = get();
    d.window_rows = get();
  }
  return out;
}

std::uint64_t RandomNonzero() {
  std::random_device device;
  std::uint64_t value = 0;
  while (value == 0)
    value = (std::uint64_t{device()} << 32) ^ device();
  return value;
}

/// Forward calls exchange partials, so they run inside a collective scope.
[[nodiscard]] bool RunsForward(TpInstructionOp op) {
  return op == TpInstructionOp::kPrefill || op == TpInstructionOp::kAdvance ||
         op == TpInstructionOp::kDecode ||
         op == TpInstructionOp::kAdvanceBatch ||
         op == TpInstructionOp::kDecodeBatch;
}

/// The constraint rank 0 bound for a request, built again from its source by
/// the same code (ConstrainChatRequest) on this rank's runner.
std::shared_ptr<const sampling::TokenConstraint> RebuildConstraint(
    const TpConstraintSource& source, const TextModelRunner& runner) {
  if (source.tool_choice >
      static_cast<std::uint8_t>(ChatRequest::ToolChoice::kRequired))
    throw std::invalid_argument("unknown tool choice");
  ChatRequest request;
  for (const auto& tool : source.tools)
    request.tools.push_back({.name = tool.name,
                             .description = tool.description,
                             .parameters_json = tool.parameters_json,
                             .definition_json = tool.definition_json});
  request.tool_choice =
      static_cast<ChatRequest::ToolChoice>(source.tool_choice);
  request.parallel_tool_calls = source.parallel_tool_calls;
  if (!source.response_format_json.empty()) {
    const auto format = json::parse(source.response_format_json);
    request.response_format =
        ParseResponseFormat(&format, source.response_format_responses);
  }
  sampling::SamplingConfig config;
  (void)ConstrainChatRequest(
      request, runner, &config, nullptr, nullptr,
      source.reasoning ? TextGenerationBackend::InitialOutputState::kReasoning
                       : TextGenerationBackend::InitialOutputState::kContent);
  if (config.constraint == nullptr)
    throw std::invalid_argument("the source describes no constraint");
  return config.constraint;
}

}  // namespace

void TpExecutionDigest::Add(std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value_ ^= (value >> shift) & 0xffU;
    value_ *= kFnvPrime;
  }
}

void TpExecutionDigest::Add(std::span<const TextRunnerToken> tokens) noexcept {
  Add(tokens.size());
  for (const auto token : tokens) {
    Add(token);
  }
}

void DigestTpCall(TpExecutionDigest& digest, const TpInstruction& instruction,
                  std::span<const TextRunnerToken> prefill_prompt) {
  digest.Add(static_cast<std::uint64_t>(instruction.op));
  digest.Add(instruction.state);
  digest.Add(static_cast<std::uint32_t>(instruction.token));
  digest.Add(instruction.offset);
  digest.Add(instruction.count);
  digest.Add(instruction.prompt_size);
  digest.Add(instruction.rng);
  digest.Add(static_cast<std::uint32_t>(instruction.pending));
  if (instruction.op == TpInstructionOp::kPrefill) {
    digest.Add(prefill_prompt);
  }
}

void DigestTpPrefill(TpExecutionDigest& digest, const TextPrefillStep& step,
                     std::size_t checkpoint) {
  digest.Add(step.consumed_tokens);
  digest.Add(step.decode_ready ? 1U : 0U);
  digest.Add(checkpoint);
}

void DigestTpAdvance(TpExecutionDigest& digest, std::size_t checkpoint) {
  digest.Add(checkpoint);
}

void DigestTpDecode(TpExecutionDigest& digest, const TextDecodeStep& step,
                    std::size_t checkpoint) {
  digest.Add(step.selections.size());
  for (const auto& selection : step.selections) {
    digest.Add(selection.token);
    digest.Add(selection.stop ? 1U : 0U);
  }
  digest.Add(step.stop ? 1U : 0U);
  digest.Add(step.draft_tokens);
  digest.Add(step.draft_accepted_tokens);
  digest.Add(checkpoint);
}

void DigestTpFailure(TpExecutionDigest& digest) {
  digest.Add(kFailureMarker);
}

bool TpCacheAcknowledged(TpInstructionOp op) noexcept {
  return op == TpInstructionOp::kBeginRestore ||
         op == TpInstructionOp::kValidate || op == TpInstructionOp::kReuse ||
         op == TpInstructionOp::kCancelPrepare;
}

void TpControlInstructionSink::Synchronize(std::uint64_t sequence,
                                           TpInstruction instruction,
                                           const std::function<void()>& local) {
  const std::lock_guard<std::mutex> lock(mutex_);
  instruction.index = next_index_;
  std::string error;
  if (!broker_->RegisterInstruction(sequence, instruction.index, &error)) {
    throw std::runtime_error(error);
  }
  if (!channel_->SendCommand({.sequence = sequence,
                              .kind = TpControlCommandKind::kInstruction,
                              .instruction = instruction},
                             &error)) {
    broker_->FailAll("TP cache instruction send failed: " + error);
    throw std::runtime_error(error);
  }
  ++next_index_;
  std::exception_ptr local_failure;
  try {
    local();
  } catch (...) {
    local_failure = std::current_exception();
  }
  // Drain even after a local failure: the next instruction must not overtake
  // a worker copy, and its acknowledgement must not become an orphan.
  const bool ok = broker_->WaitForInstruction(&error);
  if (local_failure)
    std::rethrow_exception(local_failure);
  if (!ok)
    throw std::runtime_error("TP cache worker failed: " + error);
}

bool TpControlInstructionSink::Send(std::uint64_t sequence,
                                    TpInstruction instruction,
                                    std::uint64_t* index, std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  instruction.index = next_index_;
  const TpControlCommand command{
      .sequence = sequence,
      .kind = TpControlCommandKind::kInstruction,
      .instruction = std::move(instruction),
  };
  if (!channel_->SendCommand(command, error)) {
    return false;
  }
  if (index != nullptr) {
    *index = next_index_;
  }
  ++next_index_;
  return true;
}

/// A rank-0 state paired with the id that names rank 1's corresponding state.
class TpMirroredRunner::State final : public TextRunnerState {
public:
  State(const TpMirroredRunner* owner, std::unique_ptr<TextRunnerState> inner,
        std::uint32_t id)
      : owner_(owner), inner_(std::move(inner)), id_(id) {}

  void Invalidate() noexcept override {
    owner_->SendInvalidate(id_);
    inner_->Invalidate();
  }
  /// Deliberately not forwarded; see `TpMirroredRunner`.
  void SetCancellationCheck(const CancellationCheck&) override {}
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return inner_->MeasuredResources();
  }

  [[nodiscard]] TextRunnerState& inner() const noexcept { return *inner_; }
  [[nodiscard]] std::uint32_t id() const noexcept { return id_; }

private:
  const TpMirroredRunner* owner_;
  std::unique_ptr<TextRunnerState> inner_;
  std::uint32_t id_;
};

/// Binds a forward's collective scope while the forward runs. A scope that
/// cannot be bound or released leaves the pair's collectives untrustworthy, so
/// it fails the channel as well as the call.
class TpMirroredRunner::CallScope {
public:
  CallScope(const TpMirroredRunner& owner, std::uint64_t index)
      : owner_(owner), index_(index) {
    if (owner_.scope_ == nullptr) {
      return;
    }
    std::string error;
    if (!owner_.scope_->Begin(index_, &error)) {
      Fail("TP collective scope bind failed: " + error);
    }
    bound_ = true;
  }
  CallScope(const CallScope&) = delete;
  CallScope& operator=(const CallScope&) = delete;

  /// Releases the scope once the forward returned.
  void End() {
    if (!bound_) {
      return;
    }
    bound_ = false;
    std::string error;
    if (!owner_.scope_->End(index_, &error)) {
      Fail("TP collective scope release failed: " + error);
    }
  }

  ~CallScope() {
    if (!bound_) {
      return;
    }
    try {
      std::string error;
      if (!owner_.scope_->End(index_, &error)) {
        const std::lock_guard<std::mutex> lock(owner_.mutex_);
        if (owner_.failure_.empty()) {
          owner_.failure_ = "TP collective scope release failed: " + error;
        }
      }
    } catch (...) {
    }
  }

private:
  [[noreturn]] void Fail(const std::string& message) const {
    {
      const std::lock_guard<std::mutex> lock(owner_.mutex_);
      if (owner_.failure_.empty()) {
        owner_.failure_ = message;
      }
    }
    throw std::runtime_error(message);
  }

  const TpMirroredRunner& owner_;
  std::uint64_t index_;
  bool bound_{false};
};

void TpMirroredRunner::SendCacheOp(const TpInstruction& instruction) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!failure_.empty()) {
    throw std::runtime_error("TP instruction channel failed: " + failure_);
  }
  std::string error;
  if (!sink_->Send(0, instruction, nullptr, &error)) {
    failure_ = error.empty() ? "send failed" : error;
    throw std::runtime_error("TP instruction send failed: " + failure_);
  }
}

void TpMirroredRunner::SyncCacheOp(const TpInstruction& instruction) const {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty()) {
      throw std::runtime_error("TP instruction channel failed: " + failure_);
    }
  }
  sink_->Synchronize(0, instruction, [] {});
}

void TpMirroredRunner::FailBulk(const std::string& reason) const noexcept {
  try {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (failure_.empty()) {
      failure_ = reason;
    }
  } catch (...) {
  }
}

std::uint64_t TpMirroredRunner::ProducerOf(std::uint32_t state) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = producers_.find(state);
  return found != producers_.end() ? found->second : 0;
}

bool TpMirroredRunner::AdoptRestored(std::uint32_t state, std::uint64_t nonce,
                                     std::uint64_t producer) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (nonce != nonce_) {
    // Written by an earlier process, whose verdicts are gone. It finished, and
    // its pair restarted cleanly, or it would never have been kept.
    producers_.erase(state);
    return true;
  }
  if (rejected_.contains(producer)) {
    return false;
  }
  producers_[state] = producer;
  return true;
}

void TpMirroredRunner::ForgetProducer(std::uint32_t state) const noexcept {
  try {
    const std::lock_guard<std::mutex> lock(mutex_);
    producers_.erase(state);
  } catch (...) {
  }
}

/// Rank 0's end of the bulk connection: rank 1's descriptors when it opens,
/// rank 0's releases after that. Rank 1 sends nothing more, so its end closing
/// means the peer is gone.
class TpMirroredRunner::Bulk {
public:
  Bulk(const TpMirroredRunner* owner, std::shared_ptr<TpBulkSocket> socket)
      : owner_(owner), socket_(std::move(socket)) {
    if (socket_ == nullptr) {
      throw std::invalid_argument("TP mirrored runner needs a bulk connection");
    }
  }
  ~Bulk() {
    socket_->Shutdown();
    if (reader_.joinable()) {
      reader_.join();
    }
  }

  [[nodiscard]] std::vector<cache::ComponentDescriptor> ReceiveHello() {
    TpBulkHeader header;
    std::string error;
    if (!socket_->ReceiveHeader(&header, &error) ||
        header.kind != TpBulkKind::kHello || header.bytes > (1U << 24)) {
      throw std::runtime_error("TP bulk hello failed: " + error);
    }
    std::vector<std::byte> bytes(header.bytes);
    if (!socket_->ReceivePayload(bytes, &error)) {
      throw std::runtime_error("TP bulk hello failed: " + error);
    }
    return DecodeDescriptors(bytes);
  }

  void Start() {
    reader_ = std::thread([this] { Read(); });
  }

  /// Has rank 1 copy its rows [first, end) of `component` (or its private
  /// state) into its own store under `key`, before its next call.
  void Stash(std::uint32_t state, std::uint32_t component, bool is_private,
             std::uint64_t first, std::uint64_t end, std::uint64_t key) const {
    owner_->SendCacheOp({.op = TpInstructionOp::kStash,
                         .state = state,
                         .component = component,
                         .is_private = is_private ? 1U : 0U,
                         .first = first,
                         .end = end,
                         .key = key});
  }
  /// Has rank 1 load its stored `key`, from entry row `key_first` on, into
  /// rows [first, end) of `component` (or its private state); the restore's
  /// validation reports whether it could.
  void Load(std::uint32_t state, std::uint32_t component, bool is_private,
            std::uint64_t first, std::uint64_t end, std::uint64_t key,
            std::uint64_t key_first) const {
    owner_->SendCacheOp({.op = TpInstructionOp::kLoadStashed,
                         .state = state,
                         .component = component,
                         .is_private = is_private ? 1U : 0U,
                         .first = first,
                         .end = end,
                         .key = key,
                         .key_first = key_first});
  }
  [[nodiscard]] const std::shared_ptr<TpBulkSocket>& Socket() const noexcept {
    return socket_;
  }

private:
  void Read() noexcept {
    TpBulkHeader header;
    std::string error;
    if (socket_->ReceiveHeader(&header, &error)) {
      error = "unexpected frame";
    }
    owner_->FailBulk("TP bulk connection lost: " + error);
  }

  const TpMirroredRunner* owner_;
  std::shared_ptr<TpBulkSocket> socket_;
  std::thread reader_;
};

/// One execution slot of the pair: rank 0's own, and the id of the slot at the
/// same index on rank 1.
class TpMirroredRunner::PairSlot final : public cache::Slot {
public:
  PairSlot(std::unique_ptr<cache::Slot> slot, std::uint32_t slot_id,
           PairAdapter* adapter)
      : id(slot_id), slot_(std::move(slot)), adapter_(adapter) {}
  ~PairSlot() override;
  [[nodiscard]] bool IsValid() const noexcept override {
    return slot_->IsValid();
  }
  [[nodiscard]] bool ExecutionComplete() const noexcept override {
    return slot_->ExecutionComplete();
  }
  [[nodiscard]] cache::Slot& inner() const noexcept { return *slot_; }

  const std::uint32_t id;
  /// Who produced the rows being restored: the nonce of the writing process
  /// and its request.
  std::array<std::uint64_t, 2> restored{};

private:
  std::unique_ptr<cache::Slot> slot_;
  PairAdapter* adapter_;
};

/// Presents both ranks' execution state to rank 0's cache as one model.
///
/// The components are rank 0's, then rank 1's under ids from `kPeerBase`, then
/// `kProducer`: the request that left the rows, so a checkpoint carries its
/// producer through RAM and disk. Rank 1's components hold references, not
/// bytes: each row (and a private state) is a `PeerRef` naming an entry of
/// rank 1's `TpPeerStore`, which a capture fills on rank 1 (a stash) and a
/// restore reads back there (a load), so rank 1's half never crosses the link.
/// Retention, sharing, eviction and persistence stay upstream's, unchanged,
/// and a checkpoint holds both halves or neither; its references' payloads
/// tell rank 1 when the cache drops them (`CreateComponentCacheResources`).
/// Rank 1's rows are never borrowed.
class TpMirroredRunner::PairAdapter final : public cache::Adapter {
public:
  static constexpr std::uint32_t kPeerBase = 1U << 20;
  static constexpr std::uint32_t kProducer = 1U << 21;
  [[nodiscard]] static bool IsPair(cache::ComponentId id) noexcept {
    return id.value >= kPeerBase;
  }
  /// One row of a rank-1 component, or its private state, in rank 0's cache:
  /// the stored entry and the entry row, checked so that bytes that are not a
  /// reference never reach rank 1 as one.
  struct PeerRef {
    std::uint64_t key{0};
    std::uint32_t row{0};
    std::uint32_t check{0};
  };
  static_assert(sizeof(PeerRef) == 16);
  [[nodiscard]] static std::uint32_t RefCheck(std::uint64_t key,
                                              std::uint32_t row) noexcept {
    return 0x54505246U ^ static_cast<std::uint32_t>(key) ^
           static_cast<std::uint32_t>(key >> 32) ^ (row * 0x9E3779B1U);
  }
  /// Writes references to rows [0, count) of entry `key`.
  static void WriteRefs(std::span<std::byte> bytes, std::uint64_t key) {
    if (bytes.size() % sizeof(PeerRef) != 0) {
      throw std::invalid_argument("TP peer reference buffer is misaligned");
    }
    const auto count = bytes.size() / sizeof(PeerRef);
    for (std::size_t row = 0; row < count; ++row) {
      const PeerRef ref{key, static_cast<std::uint32_t>(row),
                        RefCheck(key, static_cast<std::uint32_t>(row))};
      std::memcpy(bytes.data() + row * sizeof(PeerRef), &ref, sizeof(ref));
    }
  }
  [[nodiscard]] static PeerRef ReadRef(std::span<const std::byte> bytes,
                                       std::size_t row) {
    PeerRef ref;
    std::memcpy(&ref, bytes.data() + row * sizeof(PeerRef), sizeof(ref));
    if (ref.key == 0 || ref.check != RefCheck(ref.key, ref.row)) {
      throw std::runtime_error("TP peer reference is corrupt");
    }
    return ref;
  }
  /// The distinct entries `bytes` refers to, for releasing them.
  [[nodiscard]] static std::vector<std::uint64_t> RefKeys(
      std::span<const std::byte> bytes) {
    std::vector<std::uint64_t> keys;
    for (std::size_t row = 0; row < bytes.size() / sizeof(PeerRef); ++row) {
      PeerRef ref;
      std::memcpy(&ref, bytes.data() + row * sizeof(PeerRef), sizeof(ref));
      if (ref.key != 0 && ref.check == RefCheck(ref.key, ref.row) &&
          std::ranges::find(keys, ref.key) == keys.end()) {
        keys.push_back(ref.key);
      }
    }
    return keys;
  }

  PairAdapter(const TpMirroredRunner* owner,
              std::unique_ptr<cache::Adapter> inner)
      : owner_(owner), inner_(std::move(inner)) {
    const auto own = inner_->Components();
    const auto& peer = owner_->peer_components_;
    if (peer.size() != own.size()) {
      throw std::runtime_error("TP ranks hold different cache components");
    }
    descriptors_.assign(own.begin(), own.end());
    identity_ = inner_->CompatibilityIdentity();
    constexpr std::string_view kMarker = "tp2=pair-v2-peer-store\n";
    identity_.insert(identity_.end(), kMarker.begin(), kMarker.end());
    const auto add = [&](std::uint64_t value) {
      for (unsigned i = 0; i < 8; ++i) {
        identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
      }
    };
    for (std::size_t i = 0; i < own.size(); ++i) {
      if (peer[i].id != own[i].id || peer[i].kind != own[i].kind ||
          peer[i].rows_per_chunk != own[i].rows_per_chunk ||
          own[i].id.value == 0 || own[i].id.value >= kPeerBase) {
        throw std::runtime_error("TP ranks hold different cache components");
      }
      auto descriptor = peer[i];
      descriptor.id.value += kPeerBase;
      // Rank 1's layout keys the cache, though only references are kept.
      add(descriptor.row_bytes);
      add(descriptor.state_bytes);
      if (cache::HasRows(descriptor.kind)) {
        descriptor.row_bytes = sizeof(PeerRef);
      } else {
        descriptor.state_bytes = sizeof(PeerRef);
      }
      descriptors_.push_back(descriptor);
      const auto bytes = [](const cache::ComponentDescriptor& d) {
        return cache::HasRows(d.kind) ? d.row_bytes * d.rows_per_chunk
                                      : d.state_bytes;
      };
      if (largest_.value == 0 || bytes(own[i]) >= largest_bytes_) {
        largest_ = own[i].id;
        largest_bytes_ = bytes(own[i]);
      }
    }
    descriptors_.push_back({.id = {kProducer},
                            .layout_version = 1,
                            .kind = cache::ComponentKind::kPrivateState,
                            .state_bytes = sizeof(PairSlot::restored)});
  }

  [[nodiscard]] cache::Adapter& inner() const noexcept { return *inner_; }
  [[nodiscard]] cache::ComponentId LargestComponent() const noexcept {
    return largest_;
  }
  void Release(std::uint32_t id) noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.push_back(id);
  }

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t bytes) const override {
    return inner_->BackingPlan(bytes);
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override {
    return inner_->GetCapabilities();
  }
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override {
    return descriptors_;
  }
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override {
    return identity_;
  }
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard& guard) override {
    auto slot = inner_->CreateSlot(guard);
    std::uint32_t id = 0;
    bool reused = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!free_slots_.empty()) {
        id = free_slots_.back();
        free_slots_.pop_back();
        reused = true;
      } else {
        id = next_slot_++;
      }
    }
    // The cache replaced a slot it could not reset; rank 1 keeps its own, and
    // empties it.
    if (reused) {
      owner_->SendInvalidate(id);
      owner_->ForgetProducer(id);
    }
    return std::make_unique<PairSlot>(std::move(slot), id, this);
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot& slot) const override {
    auto positions = inner_->Positions(As(slot).inner());
    return WithPeer(std::move(positions));
  }
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot& slot, cache::Rows first,
      cache::Rows end) const override {
    return inner_->PlanPrefill(As(slot).inner(), first, end);
  }
  /// A prediction that sizes a capture; the pool asks for one at an aligned
  /// prefill boundary even without in-pass checkpoints.
  [[nodiscard]] std::vector<cache::ComponentPosition>
  PrefillCheckpointPositions(const cache::Slot& slot,
                             cache::Rows boundary) const override {
    auto positions =
        inner_->PrefillCheckpointPositions(As(slot).inner(), boundary);
    return positions.empty() ? positions : WithPeer(std::move(positions));
  }
  void BeginRestore(
      cache::Slot& slot,
      std::span<const cache::ComponentPosition> positions) override {
    auto& pair = As(slot);
    const auto split = Split(positions);
    inner_->BeginRestore(pair.inner(), split.own);
    pair.restored = {};
    owner_->SyncCacheOp({.op = TpInstructionOp::kBeginRestore,
                         .state = pair.id,
                         .positions = split.peer});
  }
  [[nodiscard]] bool Validate(
      cache::Slot& slot,
      std::span<const cache::ComponentPosition> positions) override {
    auto& pair = As(slot);
    const auto split = Split(positions);
    if (!inner_->Validate(pair.inner(), split.own)) {
      return false;
    }
    try {
      owner_->SyncCacheOp({.op = TpInstructionOp::kValidate,
                           .state = pair.id,
                           .positions = split.peer});
    } catch (const std::exception& exception) {
      Logger::Warn("tp", std::string("event=tp_restore_refused reason=") +
                             exception.what());
      return false;
    }
    return owner_->AdoptRestored(pair.id, pair.restored[0], pair.restored[1]);
  }
  [[nodiscard]] bool Invalidate(cache::Slot& slot) noexcept override {
    auto& pair = As(slot);
    if (!inner_->Invalidate(pair.inner())) {
      return false;
    }
    owner_->SendInvalidate(pair.id);
    owner_->ForgetProducer(pair.id);
    return true;
  }
  [[nodiscard]] cache::Completion CapturePrivate(
      const cache::Slot& slot, cache::ComponentId id,
      std::span<std::byte> bytes, cache::Stream& stream) override {
    return CapturePrivatePiece(slot, id, 0, bytes, stream);
  }
  [[nodiscard]] cache::Completion CapturePrivatePiece(
      const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
      std::span<std::byte> bytes, cache::Stream& stream) override {
    const auto& pair = As(slot);
    if (id.value == kProducer) {
      const std::array<std::uint64_t, 2> producer{owner_->nonce_,
                                                  owner_->ProducerOf(pair.id)};
      if (offset + bytes.size() > sizeof(producer)) {
        throw std::invalid_argument("TP producer capture is out of range");
      }
      std::memcpy(bytes.data(),
                  reinterpret_cast<const std::byte*>(producer.data()) + offset,
                  bytes.size());
      return Ready();
    }
    if (id.value >= kPeerBase) {
      if (offset != 0 || bytes.size() != sizeof(PeerRef)) {
        throw std::invalid_argument("TP peer private capture is not whole");
      }
      const auto key = RandomNonzero();
      WriteRefs(bytes, key);
      owner_->bulk_->Stash(pair.id, id.value - kPeerBase, true, 0, 1, key);
      return Ready();
    }
    return offset == 0 && bytes.size() == Private(id)
               ? inner_->CapturePrivate(pair.inner(), id, bytes, stream)
               : inner_->CapturePrivatePiece(pair.inner(), id, offset, bytes,
                                             stream);
  }
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot& slot,
                                              cache::ComponentId id,
                                              std::span<const std::byte> bytes,
                                              cache::Stream& stream) override {
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  }
  [[nodiscard]] cache::Completion LoadPrivatePiece(
      cache::Slot& slot, cache::ComponentId id, std::size_t offset,
      std::span<const std::byte> bytes, cache::Stream& stream) override {
    auto& pair = As(slot);
    if (id.value == kProducer) {
      if (offset + bytes.size() > sizeof(pair.restored)) {
        throw std::invalid_argument("TP producer restore is out of range");
      }
      std::memcpy(reinterpret_cast<std::byte*>(pair.restored.data()) + offset,
                  bytes.data(), bytes.size());
      return Ready();
    }
    if (id.value >= kPeerBase) {
      if (offset != 0 || bytes.size() != sizeof(PeerRef)) {
        throw std::invalid_argument("TP peer private restore is not whole");
      }
      const auto ref = ReadRef(bytes, 0);
      owner_->bulk_->Load(pair.id, id.value - kPeerBase, true, 0, 1, ref.key,
                          0);
      return Ready();
    }
    return offset == 0 && bytes.size() == Private(id)
               ? inner_->LoadPrivate(pair.inner(), id, bytes, stream)
               : inner_->LoadPrivatePiece(pair.inner(), id, offset, bytes,
                                          stream);
  }
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot& slot,
                                              cache::ComponentId id,
                                              cache::Rows first,
                                              cache::Rows end,
                                              std::span<std::byte> bytes,
                                              cache::Stream& stream) override {
    const auto& pair = As(slot);
    if (id.value >= kPeerBase) {
      if (end <= first || bytes.size() != (end - first) * sizeof(PeerRef)) {
        throw std::invalid_argument("TP peer row capture size is invalid");
      }
      const auto key = RandomNonzero();
      WriteRefs(bytes, key);
      owner_->bulk_->Stash(pair.id, id.value - kPeerBase, false, first, end,
                           key);
      return Ready();
    }
    return inner_->CopyRowsOut(pair.inner(), id, first, end, bytes, stream);
  }
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot& slot, cache::ComponentId id, cache::Rows first,
      cache::Rows end) const override {
    return id.value >= kPeerBase
               ? std::span<const std::byte>{}
               : inner_->BorrowRowsView(As(slot).inner(), id, first, end);
  }
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot& slot,
                                             cache::ComponentId id,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte> bytes,
                                             cache::Stream& stream) override {
    auto& pair = As(slot);
    if (id.value >= kPeerBase) {
      if (end <= first || bytes.size() != (end - first) * sizeof(PeerRef)) {
        throw std::invalid_argument("TP peer row restore size is invalid");
      }
      // Rows may come from several entries (merged adjacent chunks): one load
      // per run of consecutive rows of one entry.
      std::size_t row = 0;
      const auto rows = static_cast<std::size_t>(end - first);
      while (row < rows) {
        const auto ref = ReadRef(bytes, row);
        auto next = row + 1;
        while (next < rows) {
          const auto following = ReadRef(bytes, next);
          if (following.key != ref.key ||
              following.row != ref.row + (next - row)) {
            break;
          }
          ++next;
        }
        owner_->bulk_->Load(pair.id, id.value - kPeerBase, false, first + row,
                            first + next, ref.key, ref.row);
        row = next;
      }
      return Ready();
    }
    return inner_->CopyRowsIn(pair.inner(), id, first, end, bytes, stream);
  }

private:
  struct Positions_ {
    std::vector<cache::ComponentPosition> own;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> peer;
  };
  class ReadySignal final : public cache::CompletionSignal {
  public:
    [[nodiscard]] bool Ready() const noexcept override { return true; }
    [[nodiscard]] cache::TransferResult Wait() noexcept override {
      return cache::TransferResult::kSucceeded;
    }
  };
  [[nodiscard]] static cache::Completion Ready() {
    return cache::Completion(std::make_unique<ReadySignal>());
  }
  [[nodiscard]] static PairSlot& As(cache::Slot& slot) {
    return dynamic_cast<PairSlot&>(slot);
  }
  [[nodiscard]] static const PairSlot& As(const cache::Slot& slot) {
    return dynamic_cast<const PairSlot&>(slot);
  }
  [[nodiscard]] std::size_t Private(cache::ComponentId id) const {
    for (const auto& descriptor : descriptors_) {
      if (descriptor.id == id) {
        return descriptor.state_bytes;
      }
    }
    throw std::invalid_argument("unknown TP cache component");
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> WithPeer(
      std::vector<cache::ComponentPosition> own) const {
    const auto count = own.size();
    own.reserve(2 * count + 1);
    // Rows count themselves, and both ranks made the same calls, so a peer
    // component stands exactly where rank 0's does.
    cache::Rows boundary = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const auto position = own[i];
      own.push_back({{position.id.value + kPeerBase}, position.valid_rows});
      boundary = std::max(boundary, position.valid_rows);
    }
    own.push_back({{kProducer}, boundary});
    return own;
  }
  [[nodiscard]] static Positions_ Split(
      std::span<const cache::ComponentPosition> positions) {
    Positions_ split;
    for (const auto& position : positions) {
      if (position.id.value == kProducer) {
        continue;
      }
      if (position.id.value >= kPeerBase) {
        split.peer.emplace_back(position.id.value - kPeerBase,
                                position.valid_rows);
      } else {
        split.own.push_back(position);
      }
    }
    return split;
  }

  const TpMirroredRunner* owner_;
  std::unique_ptr<cache::Adapter> inner_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  cache::ComponentId largest_{};
  std::size_t largest_bytes_{0};
  std::mutex mutex_;
  std::uint32_t next_slot_{0};
  std::vector<std::uint32_t> free_slots_;
};

TpMirroredRunner::PairSlot::~PairSlot() {
  adapter_->Release(id);
}

TpMirroredRunner::TpMirroredRunner(std::shared_ptr<TextModelRunner> inner,
                                   std::shared_ptr<TpInstructionSink> sink,
                                   std::shared_ptr<TpBulkSocket> bulk,
                                   std::shared_ptr<TpCallScope> scope)
    : inner_(std::move(inner)),
      sink_(std::move(sink)),
      scope_(std::move(scope)),
      bulk_(std::make_unique<Bulk>(this, std::move(bulk))) {
  if (inner_ == nullptr || sink_ == nullptr) {
    throw std::invalid_argument("TP mirrored runner needs a runner and a sink");
  }
  const auto capabilities = inner_->Descriptor().capabilities;
  multi_token_decode_ = capabilities.multi_token_decode;
  nonce_ = RandomNonzero();
  peer_components_ = bulk_->ReceiveHello();
  bulk_->Start();
}

TpMirroredRunner::~TpMirroredRunner() = default;

void TpMirroredRunner::BeginRequest(std::uint64_t sequence,
                                    bool constraint_mirrored) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (sequence == 0 || requests_.contains(sequence) ||
      unsettled_.contains(sequence) || rejected_.contains(sequence)) {
    throw std::logic_error("TP request is unnamed or already open");
  }
  requests_.emplace(sequence,
                    Request{.constraint_mirrored = constraint_mirrored});
  unsettled_.insert(sequence);
}

bool TpMirroredRunner::CyclesMirrored(
    const TextRunnerState& state, const sampling::SamplerState& sampler) const {
  if (sampler.config().constraint == nullptr)
    return true;
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto lease = leases_.find(Mirrored(state).id());
  if (lease == leases_.end())
    return false;
  const auto request = requests_.find(lease->second);
  return request != requests_.end() && request->second.constraint_mirrored;
}

bool TpMirroredRunner::EndRequest(std::uint64_t sequence, std::string* error) {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = requests_.find(sequence);
  if (found == requests_.end()) {
    SetError(error, "TP request " + std::to_string(sequence) + " is not open");
    return false;
  }
  const Request request = found->second;
  requests_.erase(found);
  std::erase_if(leases_,
                [&](const auto& lease) { return lease.second == sequence; });
  if (!failure_.empty()) {
    SetError(error, "TP instruction channel failed: " + failure_);
    return false;
  }
  if (request.count > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "TP request made too many model calls to report");
    return false;
  }
  const TpInstruction end{.op = TpInstructionOp::kEnd,
                          .count = static_cast<std::uint32_t>(request.count),
                          .digest = request.digest.value()};
  std::string send_error;
  if (!sink_->Send(sequence, end, nullptr, &send_error)) {
    failure_ = send_error.empty() ? "send failed" : send_error;
    SetError(error, "TP request end send failed: " + failure_);
    return false;
  }
  return true;
}

void TpMirroredRunner::Settle(std::uint64_t sequence, bool agreed) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    unsettled_.erase(sequence);
    dependencies_.erase(sequence);
    if (!agreed) {
      rejected_.insert(sequence);
    }
  }
  settled_.notify_all();
}

bool TpMirroredRunner::AwaitDependencies(std::uint64_t sequence) const {
  // A verdict follows within one round trip once rank 1 reaches a producer's
  // end, so only a producer that has ended can be overdue. One still open may
  // legitimately run for much longer than this request.
  const auto grace = dependency_grace_;
  std::unique_lock<std::mutex> lock(mutex_);
  std::optional<std::chrono::steady_clock::time_point> deadline;
  for (;;) {
    const auto found = dependencies_.find(sequence);
    if (found == dependencies_.end()) {
      return true;
    }
    bool pending = false;
    bool open = false;
    for (const auto producer : found->second) {
      if (rejected_.contains(producer)) {
        return false;
      }
      pending = pending || unsettled_.contains(producer);
      open = open || requests_.contains(producer);
    }
    if (!pending) {
      return true;
    }
    // A failed channel settles nothing more.
    if (!failure_.empty()) {
      return false;
    }
    if (open) {
      deadline.reset();
      settled_.wait_for(lock, grace);
      continue;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!deadline.has_value()) {
      deadline = now + grace;
    } else if (now >= *deadline) {
      return false;
    }
    settled_.wait_until(lock, *deadline);
  }
}

std::string TpMirroredRunner::Failure() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return failure_;
}

bool TpMirroredRunner::HasDecodeFrontier(const TextRunnerState& state) const {
  return inner_->HasDecodeFrontier(Mirrored(state).inner());
}

bool TpMirroredRunner::CanContinue(const TextRunnerState& state,
                                   std::size_t prompt_tokens) const {
  if (!inner_->CanContinue(Mirrored(state).inner(), prompt_tokens)) {
    return false;
  }
  // A released state is reused before its producer's verdict arrives, as on
  // one host; the reusing request then depends on that verdict.
  const auto id = Mirrored(state).id();
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = producers_.find(id);
  return found == producers_.end() || !rejected_.contains(found->second);
}

TpMirroredRunner::State& TpMirroredRunner::Mirrored(TextRunnerState& state) {
  auto* mirrored = dynamic_cast<State*>(&state);
  if (mirrored == nullptr) {
    throw std::logic_error("text runner state is not a TP2 mirrored state");
  }
  return *mirrored;
}

const TpMirroredRunner::State& TpMirroredRunner::Mirrored(
    const TextRunnerState& state) {
  const auto* mirrored = dynamic_cast<const State*>(&state);
  if (mirrored == nullptr) {
    throw std::logic_error("text runner state is not a TP2 mirrored state");
  }
  return *mirrored;
}

TpMirroredRunner::Sent TpMirroredRunner::Send(
    std::uint32_t state, const TpInstruction& instruction,
    std::span<const TextRunnerToken> prefill_prompt) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!failure_.empty()) {
    throw std::runtime_error("TP instruction channel failed: " + failure_);
  }
  const auto lease = leases_.find(state);
  if (lease == leases_.end()) {
    throw std::logic_error("TP2 model call outside a request");
  }
  Sent sent{.sequence = lease->second};
  auto& request = requests_.at(sent.sequence);
  DigestTpCall(request.digest, instruction, prefill_prompt);
  ++request.count;
  producers_[state] = sent.sequence;
  std::string error;
  if (!sink_->Send(sent.sequence, instruction, &sent.index, &error)) {
    failure_ = error.empty() ? "send failed" : error;
    throw std::runtime_error("TP instruction send failed: " + failure_);
  }
  return sent;
}

TpMirroredRunner::Sent TpMirroredRunner::SendBatch(
    TpInstruction& instruction) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!failure_.empty()) {
    throw std::runtime_error("TP instruction channel failed: " + failure_);
  }
  for (auto& member : instruction.batch) {
    const auto lease = leases_.find(member.state);
    if (lease == leases_.end()) {
      throw std::logic_error("TP2 model call outside a request");
    }
    member.sequence = lease->second;
  }
  // Recorded only once every member has a request, so a refused batch leaves
  // no trace in any of them.
  for (const auto& member : instruction.batch) {
    auto& request = requests_.at(member.sequence);
    DigestTpCall(request.digest, MemberCall(instruction.op, member));
    ++request.count;
    producers_[member.state] = member.sequence;
  }
  Sent sent;
  std::string error;
  if (!sink_->Send(0, instruction, &sent.index, &error)) {
    failure_ = error.empty() ? "send failed" : error;
    throw std::runtime_error("TP instruction send failed: " + failure_);
  }
  return sent;
}

void TpMirroredRunner::SendInvalidate(std::uint32_t state) const noexcept {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
  try {
    lock.lock();
    if (!failure_.empty()) {
      return;
    }
    const TpInstruction instruction{.op = TpInstructionOp::kInvalidate,
                                    .state = state};
    // A reset of a leased state belongs to its request; the cache also resets
    // states between requests.
    std::uint64_t sequence = 0;
    if (const auto lease = leases_.find(state); lease != leases_.end()) {
      sequence = lease->second;
      auto& request = requests_.at(sequence);
      DigestTpCall(request.digest, instruction);
      ++request.count;
    }
    producers_.erase(state);
    std::string error;
    if (!sink_->Send(sequence, instruction, nullptr, &error)) {
      failure_ = error.empty() ? "reset send failed" : error;
    }
  } catch (...) {
    if (lock.owns_lock()) {
      try {
        failure_ = "reset send failed";
      } catch (...) {
      }
    }
  }
}

void TpMirroredRunner::Record(
    std::uint64_t sequence,
    const std::function<void(TpExecutionDigest&)>& add) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = requests_.find(sequence);
  if (found != requests_.end()) {
    add(found->second.digest);
  }
}

TextRunnerDescriptor TpMirroredRunner::Descriptor() const {
  auto descriptor = inner_->Descriptor();
  // An in-pass checkpoint would capture inside a call rank 1 cannot mirror
  // yet; the pool then splits the pass at the boundary and captures there.
  descriptor.capabilities.in_pass_checkpoint = false;
  // Rolling back an interrupted prefill needs the call to stop mid-pass,
  // where rank 1 would be left inside an exchange; calls here always finish.
  descriptor.capabilities.prefill_rollback = false;
  return descriptor;
}

TextRunnerResourceClaim TpMirroredRunner::ResourceClaim() const {
  return inner_->ResourceClaim();
}

std::vector<TextExecutionPlan> TpMirroredRunner::SupportedPlans() const {
  // Batched advances and decodes are each mirrored as one instruction.
  return inner_->SupportedPlans();
}

std::shared_ptr<const sampling::ConstraintVocabulary>
TpMirroredRunner::BuildConstraintVocabulary() const {
  return inner_->BuildConstraintVocabulary();
}

sampling::JsonConstraint::ToolFormat TpMirroredRunner::ToolFormat() const {
  return inner_->ToolFormat();
}

TextModelRunner::ReasoningTags TpMirroredRunner::ReasoningBudgetTags() const {
  return inner_->ReasoningBudgetTags();
}

std::uint32_t TpMirroredRunner::VocabularySize() const {
  return inner_->VocabularySize();
}

// Device probes are local to rank 0's GPU and exchange nothing with rank 1.
TextModelRunner::DeviceProbeStatus TpMirroredRunner::PollDevice() const {
  return inner_->PollDevice();
}

bool TpMirroredRunner::DeviceUsable() const {
  return inner_->DeviceUsable();
}

std::vector<TextRunnerToken> TpMirroredRunner::Tokenize(
    std::string_view text) const {
  return inner_->Tokenize(text);
}

std::vector<TextRunnerToken> TpMirroredRunner::TokenizeControl(
    std::string_view text) const {
  return inner_->TokenizeControl(text);
}

std::optional<std::vector<TextRunnerToken>> TpMirroredRunner::RenderAndTokenize(
    const ChatRequest& request) const {
  return inner_->RenderAndTokenize(request);
}

std::optional<TextPreparedPrompt> TpMirroredRunner::PreparePrompt(
    const ChatRequest& request) const {
  return inner_->PreparePrompt(request);
}

std::optional<std::size_t> TpMirroredRunner::CountPromptTokens(
    const ChatRequest& request) const {
  return inner_->CountPromptTokens(request);
}

std::optional<std::string> TpMirroredRunner::RenderPrompt(
    const ChatRequest& request) const {
  return inner_->RenderPrompt(request);
}

void TpMirroredRunner::SetPromptContext(
    TextRunnerState& state,
    std::shared_ptr<const TextPromptContext> context) const {
  auto& mirrored = Mirrored(state);
  const auto* request = dynamic_cast<const TpRequestContext*>(context.get());
  if (context != nullptr && request == nullptr) {
    throw std::invalid_argument("TP2 prompt context names no request");
  }
  const auto images = request != nullptr ? request->images : nullptr;
  // Binding a context can reset the model state, so rank 1 binds it too, at
  // the same point in the call order.
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty()) {
      throw std::runtime_error("TP instruction channel failed: " + failure_);
    }
    std::uint64_t sequence = 0;
    if (request == nullptr) {
      leases_.erase(mirrored.id());
    } else if (!requests_.contains(request->sequence)) {
      throw std::logic_error("TP request context names no open request");
    } else {
      sequence = request->sequence;
      leases_[mirrored.id()] = sequence;
      // The state may hold what a request rank 1 has not judged yet.
      const auto producer = producers_.find(mirrored.id());
      if (producer != producers_.end() && producer->second != sequence &&
          unsettled_.contains(producer->second)) {
        dependencies_[sequence].insert(producer->second);
      }
    }
    const TpInstruction instruction{.op = TpInstructionOp::kPromptContext,
                                    .state = mirrored.id()};
    if (sequence != 0) {
      auto& open = requests_.at(sequence);
      DigestTpCall(open.digest, instruction);
      DigestTpContext(open.digest, images);
      ++open.count;
    }
    std::string error;
    if (!sink_->Send(sequence, instruction, nullptr, &error)) {
      failure_ = error.empty() ? "send failed" : error;
      throw std::runtime_error("TP instruction send failed: " + failure_);
    }
  }
  // The model reads the request's EOS handling from the state it runs on.
  mirrored.inner().SetStopAtEos(state.stop_at_eos());
  inner_->SetPromptContext(mirrored.inner(), images);
}

std::vector<std::uint8_t> TpMirroredRunner::EncodePromptContext(
    const TextPromptContext& context) const {
  return inner_->EncodePromptContext(context);
}

std::shared_ptr<const TextPromptContext> TpMirroredRunner::DecodePromptContext(
    std::span<const std::uint8_t> bytes) const {
  return inner_->DecodePromptContext(bytes);
}

TextGenerationBackend::InitialOutputState TpMirroredRunner::InitialOutputState(
    const ChatRequest& request) const {
  return inner_->InitialOutputState(request);
}

std::string TpMirroredRunner::Decode(
    std::span<const TextRunnerToken> tokens) const {
  return inner_->Decode(tokens);
}

/// Tells rank 1 which of rank 0's disk files hold references to its stored
/// halves: once published, with the keys they name, and once unlinked.
class TpMirroredRunner::DiskMirror {
public:
  explicit DiskMirror(std::shared_ptr<TpBulkSocket> socket)
      : socket_(std::move(socket)) {}

  void Opened(cache::DiskStore& store) {
    std::vector<std::byte> list;
    const std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : store.Entries()) {
      ForEachPeerFile(entry.Manifest(), [&](bool private_file,
                                            const cache::DiskPayload& payload) {
        if (announced_.insert({private_file, payload.file}).second) {
          const auto* id =
              reinterpret_cast<const std::byte*>(payload.file.data());
          list.insert(list.end(), id, id + payload.file.size());
          list.push_back(
              std::byte{private_file ? std::uint8_t{1} : std::uint8_t{0}});
        }
      });
    }
    Send({.kind = TpBulkKind::kLive, .bytes = list.size()}, list);
  }

  void Published(cache::DiskStore& store, cache::CheckpointId id) {
    const auto snapshot = store.Open(id);
    if (!snapshot) {
      return;
    }
    std::vector<std::uint8_t> staging;
    ForEachPeerFile(
        snapshot->Manifest(),
        [&](bool private_file, const cache::DiskPayload& payload) {
          const TpDiskFile file{private_file, payload.file};
          {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (announced_.contains(file)) {
              return;  // an earlier checkpoint's chunk, already announced
            }
          }
          try {
            // The references just written, read back and checked.
            std::vector<std::byte> refs(payload.bytes);
            staging.resize(
                std::clamp<std::size_t>(payload.bytes, 4096, 1U << 20));
            (void)store.ReadPayload(
                *snapshot, private_file, payload, staging,
                [&](std::uint64_t offset, std::span<const std::uint8_t> piece) {
                  std::memcpy(refs.data() + offset, piece.data(), piece.size());
                });
            const auto keys = PairAdapter::RefKeys(refs);
            std::vector<std::byte> message(file.second.size() +
                                           keys.size() * sizeof(std::uint64_t));
            std::memcpy(message.data(), file.second.data(), file.second.size());
            std::memcpy(message.data() + file.second.size(), keys.data(),
                        keys.size() * sizeof(std::uint64_t));
            const std::lock_guard<std::mutex> lock(mutex_);
            if (announced_.insert(file).second) {
              Send({.kind = TpBulkKind::kPublished,
                    .is_private = private_file ? 1U : 0U,
                    .bytes = message.size()},
                   message);
            }
          } catch (const std::exception& exception) {
            // Rank 1 keeps no copy: restoring this checkpoint will miss.
            Logger::Warn("tp",
                         std::string("event=tp_disk_mirror_failed reason=") +
                             exception.what());
          }
        });
  }

  /// Runs under the disk store's locks: no store calls, one small send.
  void Removed(bool private_file, const cache::DiskFileId& id) noexcept {
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (announced_.erase({private_file, id}) != 0) {
        Send({.kind = TpBulkKind::kRemoved,
              .is_private = private_file ? 1U : 0U,
              .bytes = id.size()},
             std::as_bytes(std::span(id)));
      }
    } catch (...) {
    }
  }

private:
  template<class F>
  static void ForEachPeerFile(const cache::DiskManifest& manifest, F&& f) {
    for (const auto& component : manifest.components) {
      if (!PairAdapter::IsPair(component.descriptor.id) ||
          component.descriptor.id.value == PairAdapter::kProducer) {
        continue;
      }
      for (const auto& chunk : component.chunks) {
        f(false, chunk);
      }
      for (const auto& piece : {component.tail, component.private_state}) {
        if (piece) {
          f(true, *piece);
        }
      }
    }
  }
  void Send(const TpBulkHeader& header, std::span<const std::byte> payload) {
    std::string ignored;
    // A lost connection fails the pair anyway: nothing to mirror then.
    (void)socket_->Send(header, payload, &ignored);
  }

  std::shared_ptr<TpBulkSocket> socket_;
  std::mutex mutex_;
  std::set<TpDiskFile> announced_;
};

std::unique_ptr<ComponentCacheResources>
TpMirroredRunner::CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                                std::size_t ram) const {
  auto resources = inner_->CreateComponentCacheResources(ledger, ram);
  if (resources == nullptr || resources->adapter == nullptr) {
    throw std::runtime_error("TP2 needs a runner with a component cache");
  }
  auto pair =
      std::make_unique<PairAdapter>(this, std::move(resources->adapter));
  const auto largest_component = pair->LargestComponent();
  // The provider sizes its allocations by the components it was built with.
  // Rank 1's references take whole shared backing pages too, so they ask under
  // the provider's largest class; the request's bytes still bound them. Their
  // payload tells rank 1 when the cache drops it, so rank 1 frees its RAM copy
  // (`TpPeerStore::Release`).
  struct PeerRefs {
    PeerRefs(cache::Payload references, std::vector<std::uint64_t> stored,
             std::shared_ptr<TpBulkSocket> connection)
        : payload(std::move(references)),
          keys(std::move(stored)),
          socket(std::move(connection)) {}
    PeerRefs(const PeerRefs&) = delete;
    PeerRefs& operator=(const PeerRefs&) = delete;
    ~PeerRefs() {
      for (const auto key : keys) {
        try {
          std::string ignored;
          // A lost connection fails the pair anyway: nothing to release then.
          (void)socket->Send({.kind = TpBulkKind::kRelease, .transfer = key},
                             {}, &ignored);
        } catch (...) {
        }
      }
    }
    cache::Payload payload;
    std::vector<std::uint64_t> keys;
    std::shared_ptr<TpBulkSocket> socket;
  };
  auto allocate = std::move(resources->allocate);
  resources->allocate =
      [allocate = std::move(allocate), largest = largest_component,
       socket = bulk_->Socket()](cache::PayloadRequest request) {
        if (!PairAdapter::IsPair(request.component)) {
          return allocate(request);
        }
        request.component = largest;
        auto buffer = allocate(request);
        const auto refs = buffer.bytes.first(request.bytes);
        buffer.finish = [finish = std::move(buffer.finish), refs, socket] {
          auto holder = std::make_shared<PeerRefs>(
              finish(), PairAdapter::RefKeys(refs), socket);
          auto charge = holder->payload.CommittedCharge();
          // The owner's address is the payload's bytes to the cache: keep it,
          // under the holder's lifetime.
          const void* data = holder->payload.Owner().get();
          return cache::Payload::Committed(
              std::move(charge),
              std::shared_ptr<const void>(std::move(holder), data));
        };
        buffer.borrow = {};
        return buffer;
      };
  // Rank 1's disk tier mirrors this one (`TpPeerStore`).
  auto mirror = std::make_shared<DiskMirror>(bulk_->Socket());
  resources->disk_opened = [mirror](cache::DiskStore& store) {
    mirror->Opened(store);
  };
  resources->disk_published = [mirror](cache::DiskStore& store,
                                       cache::CheckpointId id) {
    mirror->Published(store, id);
  };
  resources->disk_removed = [mirror](bool private_file, cache::DiskFileId id) {
    mirror->Removed(private_file, id);
  };
  resources->adapter = std::move(pair);
  return resources;
}

std::unique_ptr<TextRunnerState> TpMirroredRunner::BindComponentState(
    cache::Adapter& adapter, cache::Slot& slot) const {
  auto& pair = dynamic_cast<PairAdapter&>(adapter);
  auto& bound = dynamic_cast<PairSlot&>(slot);
  return std::make_unique<State>(
      this, inner_->BindComponentState(pair.inner(), bound.inner()), bound.id);
}

void TpMirroredRunner::ReconcileComponentState(TextRunnerState& state,
                                               std::size_t position) const {
  auto& mirrored = Mirrored(state);
  SendCacheOp({.op = TpInstructionOp::kReconcile,
               .state = mirrored.id(),
               .first = position});
  inner_->ReconcileComponentState(mirrored.inner(), position);
}

bool TpMirroredRunner::BeginTpOperation(std::uint64_t index,
                                        std::string* error) const {
  return inner_->BeginTpOperation(index, error);
}

bool TpMirroredRunner::EndTpOperation(std::uint64_t index,
                                      std::string* error) const {
  return inner_->EndTpOperation(index, error);
}

std::optional<TextDecodeSelection> TpMirroredRunner::PreviewFirstToken(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  return inner_->PreviewFirstToken(Mirrored(state).inner(), sampler);
}

void TpMirroredRunner::PreparePrefixReuse(
    TextRunnerState& state, std::span<const TextRunnerToken> prefix) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  const auto sequence =
      CacheCall(mirrored.id(),
                {.op = TpInstructionOp::kReuse,
                 .state = mirrored.id(),
                 .prompt_size = ToWire(prefix.size(), "reuse prefix")},
                [&] { inner_->PreparePrefixReuse(mirrored.inner(), prefix); });
  Record(sequence, [&](TpExecutionDigest& digest) {
    digest.Add(inner_->CheckpointPosition(mirrored.inner()));
  });
}

// Rank 0's own gathers; rank 1 gathers the same rows for its state.
TextRunnerNgramRows TpMirroredRunner::NgramRows(
    const TextRunnerState& state) const {
  return inner_->NgramRows(Mirrored(state).inner());
}

void TpMirroredRunner::PrepareBatchExecution(TextRunnerState& state) const {
  inner_->PrepareBatchExecution(Mirrored(state).inner());
}

TextPrefillStep TpMirroredRunner::Prefill(
    TextRunnerState& state, std::span<const TextRunnerToken> prompt,
    std::size_t offset, std::size_t max_input_tokens) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  if (offset >= prompt.size() || max_input_tokens == 0) {
    // Nothing to prefill: runners reject the call before any model work, so
    // there is nothing for rank 1 to join.
    return inner_->Prefill(mirrored.inner(), prompt, offset, max_input_tokens);
  }
  // A budget beyond the remaining prompt means the same as the remaining
  // prompt, and fits the wire; both ranks prefill with the clamped value.
  const std::size_t count = std::min(max_input_tokens, prompt.size() - offset);
  // The pool's lookahead continues the request prompt past this frontier, as
  // rank 1's copy of it does, so its length is all rank 1 needs.
  const auto lookahead = state.prefill_lookahead();
  mirrored.inner().SetPrefillPromptEnd(state.prefill_prompt_end());
  const TpInstruction instruction{
      .op = TpInstructionOp::kPrefill,
      .state = mirrored.id(),
      .offset = ToWire(offset, "offset"),
      .count = ToWire(count, "budget"),
      .prompt_size = ToWire(prompt.size(), "prompt"),
      .lookahead = ToWire(lookahead.size(), "lookahead")};
  const auto sent = Send(mirrored.id(), instruction, prompt);
  TextPrefillStep step;
  struct ClearLookahead {
    TextRunnerState& state;
    ~ClearLookahead() { state.SetPrefillLookahead({}); }
  } clear_lookahead{mirrored.inner()};
  mirrored.inner().SetPrefillLookahead(lookahead);
  try {
    CallScope scope(*this, sent.index);
    step = inner_->Prefill(mirrored.inner(), prompt, offset, count);
    scope.End();
  } catch (...) {
    Record(sent.sequence, DigestTpFailure);
    throw;
  }
  Record(sent.sequence, [&](TpExecutionDigest& digest) {
    DigestTpPrefill(digest, step, inner_->CheckpointPosition(mirrored.inner()));
  });
  return step;
}

TextDecodeSelection TpMirroredRunner::SelectNext(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  return inner_->SelectNext(Mirrored(state).inner(), sampler);
}

void TpMirroredRunner::Advance(TextRunnerState& state,
                               TextRunnerToken token) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  const auto sent = Send(mirrored.id(), {.op = TpInstructionOp::kAdvance,
                                         .state = mirrored.id(),
                                         .token = TokenToWire(token)});
  try {
    CallScope scope(*this, sent.index);
    inner_->Advance(mirrored.inner(), token);
    scope.End();
  } catch (...) {
    Record(sent.sequence, DigestTpFailure);
    throw;
  }
  Record(sent.sequence, [&](TpExecutionDigest& digest) {
    DigestTpAdvance(digest, inner_->CheckpointPosition(mirrored.inner()));
  });
}

TextDecodeStep TpMirroredRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  // A multi-token cycle is one instruction. It carries the sampler's draw
  // state, so rank 1, whose sampler otherwise matches, draws what rank 0 draws
  // and makes the same draft and acceptance decisions. A one-token budget runs
  // through the base implementation instead, which selects here, on rank 0
  // only, and mirrors the `Advance`; forwarding it to the wrapped runner would
  // let that `Advance` bypass this wrapper.
  // A constrained request (structured output, tool calls) whose constraint
  // rank 1 could not rebuild also runs one token at a time: rank 1 could not
  // make a cycle's draft and acceptance decisions.
  if (max_tokens < 2 || !multi_token_decode_ ||
      !CyclesMirrored(state, sampler)) {
    return TextModelRunner::DecodeStep(state, max_tokens, sampler);
  }
  auto& mirrored = Mirrored(state);
  const auto count = ToWire(max_tokens, "decode budget");
  const auto draw = sampler.SaveDrawState();
  // Rank 0's draft count fixes the cycle width on both ranks: rank 1's own
  // acceptance history could choose another and meet the trunk forward with
  // a different all-reduce size.
  const auto plan = inner_->PlanDecodeStep(mirrored.inner(), count, sampler);
  const auto sent =
      Send(mirrored.id(),
           {.op = TpInstructionOp::kDecode,
            .state = mirrored.id(),
            .count = count,
            .rng = draw.rng,
            .pending = PendingToWire(draw),
            .drafts = plan.has_value()
                          ? ToWire(std::size_t{*plan} + 1, "draft count")
                          : 0});
  TextDecodeStep step;
  try {
    CallScope scope(*this, sent.index);
    step = inner_->DecodeStepPlanned(mirrored.inner(), count, sampler, plan);
    scope.End();
  } catch (...) {
    Record(sent.sequence, DigestTpFailure);
    throw;
  }
  Record(sent.sequence, [&](TpExecutionDigest& digest) {
    DigestTpDecode(digest, step, inner_->CheckpointPosition(mirrored.inner()));
  });
  return step;
}

std::vector<TextDecodeStep> TpMirroredRunner::DecodeBatch(
    std::span<const TextRunnerDecode> decodes) const {
  if (decodes.size() < 2 || !multi_token_decode_ ||
      std::ranges::any_of(decodes, [this](const TextRunnerDecode& decode) {
        return !CyclesMirrored(decode.state.get(), decode.sampler.get());
      })) {
    // The base batch runs each member's DecodeStep, which takes members with
    // an unmirrored constraint one token at a time.
    return TextModelRunner::DecodeBatch(decodes);
  }
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  // One instruction carries every member's budget and draw state, like the
  // decode each replaces, and the batch's draft count: rank 0 chooses it from
  // its own cycle timings, which rank 1 cannot reproduce.
  TpInstruction instruction{.op = TpInstructionOp::kDecodeBatch};
  std::vector<TextRunnerDecode> inner;
  inner.reserve(decodes.size());
  for (const auto& decode : decodes) {
    auto& mirrored = Mirrored(decode.state.get());
    const auto draw = decode.sampler.get().SaveDrawState();
    instruction.batch.push_back(
        {.state = mirrored.id(),
         .count = ToWire(decode.max_tokens, "decode budget"),
         .rng = draw.rng,
         .pending = PendingToWire(draw)});
    inner.push_back({.state = mirrored.inner(),
                     .max_tokens = decode.max_tokens,
                     .sampler = decode.sampler});
  }
  const auto plan = inner_->PlanDecodeBatch(inner);
  instruction.batch_drafts =
      plan.has_value() ? ToWire(std::size_t{*plan} + 1, "draft count") : 0;
  if (!plan.has_value()) {
    // A batch without a shared plan (a sampled one) takes each member's own
    // draft count; rank 0 chooses them and both ranks run the same widths.
    for (std::size_t index = 0; index < inner.size(); ++index) {
      const auto member_plan = inner_->PlanDecodeStep(
          inner[index].state.get(), inner[index].max_tokens,
          inner[index].sampler.get(), inner.size());
      instruction.batch[index].drafts =
          member_plan.has_value()
              ? ToWire(std::size_t{*member_plan} + 1, "draft count")
              : 0;
      inner[index].drafts = member_plan;
    }
  }
  const auto sent = SendBatch(instruction);
  std::vector<TextDecodeStep> steps;
  try {
    CallScope scope(*this, sent.index);
    steps = inner_->DecodeBatchPlanned(inner, plan);
    scope.End();
    if (steps.size() != decodes.size()) {
      throw std::runtime_error(
          "text runner returned an invalid decode batch size");
    }
  } catch (...) {
    // The batch failed as a whole, including members it had decoded.
    const auto failure = std::current_exception();
    steps.assign(decodes.size(), TextDecodeStep{});
    for (auto& step : steps) {
      step.failure = failure;
    }
  }
  for (std::size_t index = 0; index < decodes.size(); ++index) {
    const auto sequence = instruction.batch[index].sequence;
    if (steps[index].failure) {
      Record(sequence, DigestTpFailure);
      continue;
    }
    Record(sequence, [&](TpExecutionDigest& digest) {
      DigestTpDecode(digest, steps[index],
                     inner_->CheckpointPosition(inner[index].state.get()));
    });
  }
  return steps;
}

void TpMirroredRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  if (advances.size() < 2) {
    TextModelRunner::AdvanceBatch(advances);
    return;
  }
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  TpInstruction instruction{.op = TpInstructionOp::kAdvanceBatch};
  std::vector<std::exception_ptr> failures(advances.size());
  std::vector<TextRunnerAdvance> inner;
  inner.reserve(advances.size());
  for (std::size_t index = 0; index < advances.size(); ++index) {
    auto& mirrored = Mirrored(advances[index].state.get());
    instruction.batch.push_back(
        {.state = mirrored.id(), .token = TokenToWire(advances[index].token)});
    inner.push_back({.state = mirrored.inner(),
                     .token = advances[index].token,
                     .failure = &failures[index]});
  }
  const auto sent = SendBatch(instruction);
  try {
    CallScope scope(*this, sent.index);
    inner_->AdvanceBatch(inner);
    scope.End();
  } catch (...) {
    // The batch failed as a whole, including members it had advanced.
    const auto failure = std::current_exception();
    for (auto& member : failures) {
      if (!member) {
        member = failure;
      }
    }
  }
  std::exception_ptr unreported;
  for (std::size_t index = 0; index < advances.size(); ++index) {
    const auto sequence = instruction.batch[index].sequence;
    if (failures[index]) {
      Record(sequence, DigestTpFailure);
      if (advances[index].failure != nullptr) {
        *advances[index].failure = failures[index];
      } else if (!unreported) {
        unreported = failures[index];
      }
      continue;
    }
    Record(sequence, [&](TpExecutionDigest& digest) {
      DigestTpAdvance(digest,
                      inner_->CheckpointPosition(inner[index].state.get()));
    });
  }
  if (unreported) {
    std::rethrow_exception(unreported);
  }
}

std::size_t TpMirroredRunner::CheckpointPosition(
    const TextRunnerState& state) const {
  return inner_->CheckpointPosition(Mirrored(state).inner());
}

void TpMirroredRunner::PrepareCancellation(TextRunnerState& state) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  (void)CacheCall(
      mirrored.id(),
      {.op = TpInstructionOp::kCancelPrepare, .state = mirrored.id()},
      [&] { inner_->PrepareCancellation(mirrored.inner()); });
}

std::uint64_t TpMirroredRunner::CacheCall(
    std::uint32_t state, const TpInstruction& instruction,
    const std::function<void()>& local) const {
  std::uint64_t sequence = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto lease = leases_.find(state);
    if (lease == leases_.end() || !failure_.empty()) {
      throw std::logic_error("TP cache call outside a healthy request");
    }
    sequence = lease->second;
    auto& request = requests_.at(sequence);
    DigestTpCall(request.digest, instruction);
    ++request.count;
    producers_[state] = sequence;
  }
  try {
    sink_->Synchronize(sequence, instruction, local);
  } catch (...) {
    Record(sequence, DigestTpFailure);
    throw;
  }
  return sequence;
}

/// One open request on rank 1: what it needs to execute rank 0's calls for it
/// and to judge them.
struct TpExecutor::Request {
  /// `sampling` is the request's, with its constraint when rank 1 rebuilt it.
  Request(const TpControlCommand& begin,
          const sampling::SamplingConfig& sampling,
          std::vector<TextRunnerToken> tokens, std::size_t states)
      : prompt(std::move(tokens)),
        greedy(sampling.can_use_unmodified_argmax() && !begin.constrained),
        stop_at_eos(begin.stop_at_eos),
        sampler(sampling, prompt),
        own(states) {}

  void Note(std::string message) {
    if (failure.empty()) {
      failure = std::move(message);
    }
  }

  std::vector<TextRunnerToken> prompt;
  bool stop_at_eos{true};
  /// The request's prompt context (its images), rebuilt from `kSingle`.
  std::shared_ptr<const TextPromptContext> images;
  /// Under greedy decoding the logits are bit-identical on both ranks, so rank
  /// 1's own choice must equal every token rank 0 advances. Rank 0's token is
  /// what both ranks feed, so without this check a numerical divergence (the
  /// full-Q8 bug was one) would go unnoticed. The choice is made after each
  /// call while rank 0 is still selecting, so it costs rank 1 idle time only.
  bool greedy;
  /// Built as rank 0's pool builds the request's sampler, and fed the same
  /// accepted tokens, so a multi-token decode sees the same history; each
  /// `decode` instruction supplies rank 0's draw state.
  sampling::SamplerState sampler;
  std::vector<std::optional<OwnChoice>> own;
  TpExecutionDigest digest;
  std::uint64_t count{0};
  std::string failure;
};

namespace {
/// Rank 1 keeps no checkpoints: nothing borrows its slots' rows, so nothing
/// needs preserving before they change.
class NoBorrowers final : public cache::MutationGuard {
public:
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
};
}  // namespace

/// Rank 1's end of the bulk connection: its descriptors once, then rank 0's
/// releases, which a thread applies to the store.
class TpExecutor::Bulk {
public:
  Bulk(TpExecutor* owner, std::shared_ptr<TpBulkSocket> socket)
      : owner_(owner), socket_(std::move(socket)) {}
  ~Bulk() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // Releases rank 0 sent before it closed its end still count: they tell
      // the store which halves to keep on disk.
      if (in_.joinable()) {
        (void)received_.wait_for(lock, std::chrono::seconds(2),
                                 [&] { return received_all_; });
      }
    }
    socket_->Shutdown();
    if (in_.joinable()) {
      in_.join();
    }
  }
  void SendHello(std::span<const cache::ComponentDescriptor> components) {
    const auto bytes = EncodeDescriptors(components);
    std::string error;
    if (!socket_->Send({.kind = TpBulkKind::kHello, .bytes = bytes.size()},
                       bytes, &error)) {
      throw std::runtime_error("TP bulk hello failed: " + error);
    }
  }
  void Start() {
    in_ = std::thread([this] { ReceiveLoop(); });
  }

private:
  void ReceiveLoop() noexcept {
    constexpr std::size_t kFileBytes = 16;
    constexpr std::uint64_t kMaxPayload = std::uint64_t{1} << 28;
    std::vector<std::byte> payload;
    for (;;) {
      TpBulkHeader header;
      std::string error;
      if (!socket_->ReceiveHeader(&header, &error) ||
          header.bytes > kMaxPayload) {
        break;
      }
      const auto bytes = static_cast<std::size_t>(header.bytes);
      const bool valid =
          header.kind == TpBulkKind::kRelease ? bytes == 0
          : header.kind == TpBulkKind::kPublished
              ? bytes >= kFileBytes && (bytes - kFileBytes) % 8 == 0
          : header.kind == TpBulkKind::kRemoved ? bytes == kFileBytes
          : header.kind == TpBulkKind::kLive    ? bytes % (kFileBytes + 1) == 0
                                                : false;
      if (!valid) {
        break;
      }
      try {
        payload.resize(bytes);
      } catch (...) {
        break;
      }
      if (bytes != 0 && !socket_->ReceivePayload(payload, &error)) {
        break;
      }
      const auto file_at = [&](std::size_t offset, bool is_private) {
        TpDiskFile file{is_private, {}};
        std::memcpy(file.second.data(), payload.data() + offset, kFileBytes);
        return file;
      };
      try {
        switch (header.kind) {
          case TpBulkKind::kRelease:
            owner_->store_->Release(header.transfer);
            break;
          case TpBulkKind::kPublished: {
            std::vector<std::uint64_t> keys((bytes - kFileBytes) / 8);
            std::memcpy(keys.data(), payload.data() + kFileBytes,
                        keys.size() * 8);
            owner_->store_->Published(file_at(0, header.is_private != 0), keys);
            break;
          }
          case TpBulkKind::kRemoved:
            owner_->store_->Removed(file_at(0, header.is_private != 0));
            break;
          case TpBulkKind::kLive: {
            std::vector<TpDiskFile> files;
            for (std::size_t at = 0; at < bytes; at += kFileBytes + 1) {
              files.push_back(
                  file_at(at, payload[at + kFileBytes] != std::byte{0}));
            }
            owner_->store_->Live(files);
            break;
          }
          default:
            break;
        }
      } catch (const std::exception& exception) {
        Logger::Warn("tp", std::string("event=tp_peer_store_notice_failed "
                                       "reason=") +
                               exception.what());
      }
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      received_all_ = true;
    }
    received_.notify_all();
  }

  TpExecutor* owner_;
  std::shared_ptr<TpBulkSocket> socket_;
  std::thread in_;
  std::mutex mutex_;
  std::condition_variable received_;
  bool received_all_{false};
};

std::span<std::byte> TpExecutor::Staging(std::size_t bytes) {
  if (bytes > staging_.size()) {
    const auto size = std::max<std::size_t>(std::bit_ceil(bytes), 1U << 20);
    auto allocation = resources_->staging(size);
    staging_owner_ = std::move(allocation.owner);
    staging_ = {reinterpret_cast<std::byte*>(allocation.bytes.data()), size};
  }
  return staging_.first(bytes);
}

void TpExecutor::RunStored(const TpInstruction& job) {
  auto& slot = *slots_.at(job.state);
  const cache::ComponentId id{job.component};
  const auto& descriptor = Descriptor(id);
  const bool whole = job.is_private != 0;
  if (whole != (descriptor.kind == cache::ComponentKind::kPrivateState)) {
    throw std::invalid_argument("TP stored transfer names the wrong component");
  }
  std::unique_ptr<cache::Stream> stream;
  while ((stream = resources_->stream()) == nullptr) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  const auto settle = [](cache::Completion completion) {
    if (completion.Wait() != cache::TransferResult::kSucceeded) {
      throw std::runtime_error("device transfer failed");
    }
  };
  if (job.op == TpInstructionOp::kStash) {
    const std::size_t rows = whole ? 1 : job.end - job.first;
    const cache::PayloadRequest request{
        id,
        whole ? cache::ResourceCategory::kPrivateState
              : cache::ResourceCategory::kBackingAssigned,
        whole ? 0 : job.first, whole ? 0 : job.end,
        whole ? descriptor.state_bytes : rows * descriptor.row_bytes};
    store_->Stash(
        job.key, request, rows, [&](std::span<std::byte> destination) {
          settle(whole
                     ? adapter_->CapturePrivate(slot, id, destination, *stream)
                     : adapter_->CopyRowsOut(slot, id, job.first, job.end,
                                             destination, *stream));
        });
    return;
  }
  store_->Load(
      job.key, id, whole, whole ? 0 : job.key_first,
      whole ? 1 : job.key_first + (job.end - job.first),
      [&](std::size_t bytes) { return Staging(bytes); },
      [&](std::span<const std::byte> source) {
        settle(whole ? adapter_->LoadPrivate(slot, id, source, *stream)
                     : adapter_->CopyRowsIn(slot, id, job.first, job.end,
                                            source, *stream));
      });
}

void TpExecutor::CopyStored(const TpInstruction& job,
                            std::span<const std::byte> source) {
  auto& slot = *slots_.at(job.state);
  const cache::ComponentId id{job.component};
  const bool whole = job.is_private != 0;
  std::unique_ptr<cache::Stream> stream;
  while ((stream = resources_->stream()) == nullptr) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  auto completion = whole ? adapter_->LoadPrivate(slot, id, source, *stream)
                          : adapter_->CopyRowsIn(slot, id, job.first, job.end,
                                                 source, *stream);
  if (completion.Wait() != cache::TransferResult::kSucceeded) {
    throw std::runtime_error("device transfer failed");
  }
}

void TpExecutor::QueueLoad(const TpInstruction& job) {
  // Same checks as an inline load, before any read starts.
  if (job.state >= slots_.size()) {
    throw std::out_of_range("TP cache instruction names no slot");
  }
  const auto& descriptor = Descriptor(cache::ComponentId{job.component});
  if ((job.is_private != 0) !=
      (descriptor.kind == cache::ComponentKind::kPrivateState)) {
    throw std::invalid_argument("TP stored transfer names the wrong component");
  }
  if (pending_loads_.size() >= read_ahead_) {
    DrainOneLoad();
  }
  const auto index = free_load_buffers_.back();
  free_load_buffers_.pop_back();
  auto read = [this, job, index]() -> std::span<const std::byte> {
    auto& buffer = load_buffers_[index];
    const auto room = [&](std::size_t bytes) {
      if (bytes > buffer.bytes.size()) {
        const auto size = std::max<std::size_t>(std::bit_ceil(bytes), 1U << 20);
        const std::lock_guard<std::mutex> lock(load_buffer_mutex_);
        auto allocation = resources_->staging(size);
        buffer.owner = std::move(allocation.owner);
        buffer.bytes = {reinterpret_cast<std::byte*>(allocation.bytes.data()),
                        size};
      }
      return buffer.bytes.first(bytes);
    };
    const bool whole = job.is_private != 0;
    std::span<const std::byte> out;
    store_->Load(
        job.key, cache::ComponentId{job.component}, whole,
        whole ? 0 : job.key_first,
        whole ? 1 : job.key_first + (job.end - job.first), room,
        [&](std::span<const std::byte> source) {
          const auto* begin = buffer.bytes.data();
          if (begin != nullptr && source.data() >= begin &&
              source.data() + source.size() <= begin + buffer.bytes.size()) {
            out = source;  // read from disk into this buffer
            return;
          }
          // A RAM copy may go once Load returns: keep the bytes.
          auto destination = room(source.size());
          std::memcpy(destination.data(), source.data(), source.size());
          out = destination;
        });
    return out;
  };
  try {
    pending_loads_.push_back(
        {job, index, std::async(std::launch::async, std::move(read))});
  } catch (...) {
    free_load_buffers_.push_back(index);
    throw;
  }
}

void TpExecutor::DrainOneLoad() noexcept {
  auto pending = std::move(pending_loads_.front());
  pending_loads_.pop_front();
  try {
    try {
      CopyStored(pending.instruction, pending.read.get());
    } catch (const std::exception& exception) {
      Logger::Warn("tp", std::string("event=tp_peer_store_failed op=") +
                             std::string(OpName(pending.instruction.op)) +
                             " reason=" + exception.what());
      failed_restores_.insert(pending.instruction.state);
    }
  } catch (...) {
    failed_restores_.insert(pending.instruction.state);
  }
  free_load_buffers_.push_back(pending.buffer);
}

void TpExecutor::DrainLoads() noexcept {
  while (!pending_loads_.empty()) {
    DrainOneLoad();
  }
}

const cache::ComponentDescriptor& TpExecutor::Descriptor(
    cache::ComponentId id) const {
  for (const auto& descriptor : adapter_->Components()) {
    if (descriptor.id == id) {
      return descriptor;
    }
  }
  throw std::invalid_argument("TP transfer names no component");
}

TpExecutor::TpExecutor(std::shared_ptr<TextModelRunner> runner,
                       std::size_t state_count,
                       std::shared_ptr<TpBulkSocket> bulk,
                       std::shared_ptr<TpCallScope> scope,
                       TpPeerStoreOptions store)
    : runner_(std::move(runner)), scope_(std::move(scope)) {
  if (runner_ == nullptr || state_count == 0 || bulk == nullptr) {
    throw std::invalid_argument(
        "TP executor needs a runner, a state and a bulk connection");
  }
  // The store's pool is bounded by its committed backing, not by the ledger.
  constexpr std::size_t kUnlimited = std::size_t{1} << 62;
  ledger_ = std::make_unique<cache::ResourceLedger>(
      cache::ResourceLimits{kUnlimited, kUnlimited, kUnlimited, kUnlimited});
  resources_ = runner_->CreateComponentCacheResources(*ledger_, 0);
  if (resources_ == nullptr || resources_->adapter == nullptr) {
    throw std::runtime_error("TP2 needs a runner with a component cache");
  }
  adapter_ = resources_->adapter.get();
  guard_ = std::make_unique<NoBorrowers>();
  bulk_ = std::make_unique<Bulk>(this, std::move(bulk));
  // Rank 0 sizes its cache from these before either rank allocates a session.
  bulk_->SendHello(adapter_->Components());
  slots_.reserve(state_count);
  states_.reserve(state_count);
  for (std::size_t index = 0; index < state_count; ++index) {
    slots_.push_back(adapter_->CreateSlot(*guard_));
    auto state = runner_->BindComponentState(*adapter_, *slots_.back());
    if (state == nullptr) {
      throw std::runtime_error("TP executor state creation failed");
    }
    states_.push_back(std::move(state));
  }
  // As rank 0's cache does: backing is sized and committed once the slots
  // exist, from what the host still holds then.
  {
    const auto claim = runner_->ResourceClaim();
    const auto measured = claim.retained_snapshot_capacity_bytes.value_or(0);
    const auto ceiling =
        claim.retained_snapshot_ceiling_bytes.value_or(measured);
    store.ram_bytes =
        std::min(store.ram_bytes, store.automatic_ram ? measured : ceiling);
  }
  if (resources_->commit_backing && store.ram_bytes != 0) {
    resources_->commit_backing(store.ram_bytes);
  }
  Logger::Info("tp", "event=tp_peer_store_configured ram_bytes=" +
                         std::to_string(store.ram_bytes) + " disk_mirror=" +
                         (store.disk_directory.empty() ? "off" : "on"));
  const bool disk_tier = !store.disk_directory.empty();
  store_ =
      std::make_unique<TpPeerStore>(resources_->allocate, std::move(store));
  // Disk loads are separate small reads; reading several ahead overlaps them
  // (GUFO_TP_LOAD_READAHEAD=0 loads each inline, as before).
  read_ahead_ = disk_tier ? 8 : 0;
  if (const char* value = std::getenv("GUFO_TP_LOAD_READAHEAD")) {
    read_ahead_ = std::min<std::size_t>(std::strtoul(value, nullptr, 10), 32);
  }
  load_buffers_.resize(read_ahead_);
  for (std::size_t i = read_ahead_; i > 0; --i) {
    free_load_buffers_.push_back(i - 1);
  }
  Logger::Info("tp",
               "event=tp_load_read_ahead depth=" + std::to_string(read_ahead_));
  bulk_->Start();
  // Rank 0's scheduler prepares its vocabulary before serving; rank 1 would
  // otherwise build its own inside the first constrained request, while rank 0
  // waits for it in the request's first exchange.
  (void)runner_->PrepareConstraintVocabulary();
}

TpExecutor::~TpExecutor() = default;

bool TpExecutor::TakeIndex(const TpControlCommand& command,
                           std::string* error) {
  if (command.instruction.index != next_index_) {
    SetError(error, "TP instruction " +
                        std::to_string(command.instruction.index) +
                        " arrived, expected " + std::to_string(next_index_));
    return false;
  }
  ++next_index_;
  return true;
}

TextRunnerState& TpExecutor::StateFor(std::uint32_t id) {
  if (id >= states_.size()) {
    throw std::out_of_range("TP instruction names state " + std::to_string(id) +
                            " of " + std::to_string(states_.size()));
  }
  return *states_[id];
}

TpExecutor::OwnChoice TpExecutor::ChooseGreedy(TextRunnerState& state) const {
  sampling::SamplerState greedy{sampling::SamplingConfig{}};
  const auto selection = runner_->SelectNext(state, greedy);
  return {.stop = selection.stop, .token = selection.token};
}

bool TpExecutor::BeginScope(std::uint64_t index, std::string* error) {
  std::string scope_error;
  if (scope_ != nullptr && !scope_->Begin(index, &scope_error)) {
    SetError(error, "TP collective scope bind failed: " + scope_error);
    return false;
  }
  return true;
}

bool TpExecutor::EndScope(std::uint64_t index, std::string* error) {
  std::string scope_error;
  if (scope_ != nullptr && !scope_->End(index, &scope_error)) {
    SetError(error, "TP collective scope release failed: " + scope_error);
    return false;
  }
  return true;
}

bool TpExecutor::Execute(const TpControlCommand& command,
                         const Respond& respond, std::string* error) {
  // Queued loads land before anything else runs, in their channel order.
  if (command.kind == TpControlCommandKind::kSingle ||
      command.instruction.op != TpInstructionOp::kLoadStashed) {
    DrainLoads();
  }
  if (command.kind == TpControlCommandKind::kSingle) {
    return Open(command, error);
  }
  if (!TakeIndex(command, error)) {
    return false;
  }
  const auto& instruction = command.instruction;
  if (instruction.op == TpInstructionOp::kBeginRestore ||
      instruction.op == TpInstructionOp::kValidate ||
      instruction.op == TpInstructionOp::kReconcile ||
      instruction.op == TpInstructionOp::kStash ||
      instruction.op == TpInstructionOp::kLoadStashed) {
    return ExecuteCacheOp(command.sequence, instruction, respond, error);
  }
  if (instruction.op == TpInstructionOp::kAdvanceBatch ||
      instruction.op == TpInstructionOp::kDecodeBatch) {
    return ExecuteBatch(instruction, error);
  }
  if (command.sequence == 0) {
    return ExecuteIdle(instruction, error);
  }
  const auto found = requests_.find(command.sequence);
  if (found == requests_.end()) {
    SetError(error, "TP instruction names request " +
                        std::to_string(command.sequence) +
                        ", which is not open");
    return false;
  }
  auto& request = *found->second;
  if (instruction.op != TpInstructionOp::kEnd) {
    return ExecuteCall(command.sequence, request, instruction, respond, error);
  }
  if (request.failure.empty() &&
      (instruction.count != request.count ||
       instruction.digest != request.digest.value())) {
    request.failure = "TP ranks ran different model calls: rank 0 made " +
                      std::to_string(instruction.count) + " with digest " +
                      Hex(instruction.digest) + ", rank 1 made " +
                      std::to_string(request.count) + " with digest " +
                      Hex(request.digest.value());
  }
  TpControlResponse response{.sequence = command.sequence,
                             .error = std::move(request.failure)};
  if (response.error.size() > (1U << 20)) {
    response.error.resize(1U << 20);
  }
  requests_.erase(found);
  return respond(response, error);
}

bool TpExecutor::Open(const TpControlCommand& begin, std::string* error) {
  if (begin.sequence == 0 || requests_.contains(begin.sequence)) {
    SetError(error, "TP request " + std::to_string(begin.sequence) +
                        " is unnamed or already open");
    return false;
  }
  std::vector<TextRunnerToken> prompt;
  prompt.reserve(begin.prompt_tokens.size());
  for (const auto token : begin.prompt_tokens) {
    if (token < 0) {
      SetError(error, "TP request prompt holds a negative token");
      return false;
    }
    prompt.push_back(static_cast<TextRunnerToken>(token));
  }
  // A constraint or reasoning budget rank 1 cannot rebuild fails the request,
  // not the pair.
  auto sampling = begin.sampling;
  std::string constraint_error;
  if (begin.constraint_source) {
    try {
      sampling.constraint =
          RebuildConstraint(*begin.constraint_source, *runner_);
    } catch (const std::exception& exception) {
      constraint_error = exception.what();
    }
  }
  if (begin.reasoning_budget) {
    sampling.reasoning_budget =
        runner_->BuildReasoningBudget(*begin.reasoning_budget, prompt);
  }
  auto request = std::make_unique<Request>(begin, sampling, std::move(prompt),
                                           states_.size());
  if (!constraint_error.empty()) {
    request->Note("rank 1 could not rebuild the request's constraint: " +
                  constraint_error);
  }
  if (begin.reasoning_budget && sampling.reasoning_budget == nullptr) {
    request->Note("rank 1 could not rebuild the request's reasoning budget");
  }
  if (!begin.prompt_context.empty()) {
    // A context rank 1 cannot rebuild fails the request, not the pair: its
    // calls still run, and the digest reports the difference.
    try {
      request->images = runner_->DecodePromptContext(begin.prompt_context);
    } catch (const std::exception& exception) {
      request->Note(std::string("rank 1 prompt context is invalid: ") +
                    exception.what());
    }
  }
  requests_.emplace(begin.sequence, std::move(request));
  return true;
}

bool TpExecutor::ExecuteIdle(const TpInstruction& instruction,
                             std::string* error) {
  if (instruction.op != TpInstructionOp::kInvalidate &&
      instruction.op != TpInstructionOp::kPromptContext) {
    SetError(error, "TP worker received a model call outside a request");
    return false;
  }
  try {
    if (instruction.op == TpInstructionOp::kPromptContext) {
      runner_->SetPromptContext(StateFor(instruction.state), nullptr);
    } else {
      // A restore rank 0 abandoned leaves the slot refusing execution; this
      // rank's adapter makes it an empty slot again.
      StateFor(instruction.state).Invalidate();
    }
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
  return true;
}

bool TpExecutor::ExecuteCacheOp(std::uint64_t sequence,
                                const TpInstruction& instruction,
                                const Respond& respond, std::string* error) {
  std::string call_error;
  try {
    if (instruction.state >= slots_.size()) {
      throw std::out_of_range("TP cache instruction names no slot");
    }
    auto& slot = *slots_[instruction.state];
    std::vector<cache::ComponentPosition> positions;
    positions.reserve(instruction.positions.size());
    for (const auto& [component, rows] : instruction.positions) {
      positions.push_back({{component}, rows});
    }
    switch (instruction.op) {
      case TpInstructionOp::kBeginRestore:
        failed_restores_.erase(instruction.state);
        adapter_->BeginRestore(slot, positions);
        break;
      case TpInstructionOp::kValidate:
        if (failed_restores_.erase(instruction.state) != 0) {
          throw std::runtime_error("rank 1 could not load its stored state");
        }
        if (!adapter_->Validate(slot, positions)) {
          throw std::runtime_error("rank 1 refused the restored state");
        }
        break;
      case TpInstructionOp::kReconcile:
        runner_->ReconcileComponentState(StateFor(instruction.state),
                                         instruction.first);
        break;
      case TpInstructionOp::kStash:
      case TpInstructionOp::kLoadStashed:
        // Unanswered: a failed stash leaves its key missing, and a failed
        // load fails the restore's validation.
        try {
          // Only disk reads go ahead; a RAM copy loads inline as before,
          // after the queued loads, so the copies keep the channel order.
          if (instruction.op == TpInstructionOp::kLoadStashed &&
              read_ahead_ != 0 && !store_->InRam(instruction.key)) {
            QueueLoad(instruction);
          } else {
            if (instruction.op == TpInstructionOp::kLoadStashed) {
              DrainLoads();
            }
            RunStored(instruction);
          }
        } catch (const std::exception& exception) {
          Logger::Warn("tp", std::string("event=tp_peer_store_failed op=") +
                                 std::string(OpName(instruction.op)) +
                                 " reason=" + exception.what());
          if (instruction.op == TpInstructionOp::kLoadStashed) {
            failed_restores_.insert(instruction.state);
          }
        }
        break;
      default:
        throw std::logic_error("not a TP cache instruction");
    }
  } catch (const std::exception& exception) {
    call_error = "rank 1 " + std::string(OpName(instruction.op)) +
                 " failed: " + exception.what();
  }
  if (TpCacheAcknowledged(instruction.op)) {
    return respond({.instruction_index = instruction.index,
                    .sequence = sequence,
                    .error = call_error,
                    .kind = TpControlResponseKind::kInstruction},
                   error);
  }
  if (!call_error.empty()) {
    SetError(error, call_error);
    return false;
  }
  return true;
}

bool TpExecutor::ExecuteCall(std::uint64_t sequence, Request& request,
                             const TpInstruction& instruction,
                             const Respond& respond, std::string* error) {
  ++request.count;
  const bool prompt_fits = instruction.prompt_size <= request.prompt.size();
  const auto prefill_prompt =
      instruction.op == TpInstructionOp::kPrefill && prompt_fits
          ? std::span<const TextRunnerToken>(request.prompt)
                .first(instruction.prompt_size)
          : std::span<const TextRunnerToken>{};
  DigestTpCall(request.digest, instruction, prefill_prompt);
  const bool forward = RunsForward(instruction.op);
  if (forward && !BeginScope(instruction.index, error)) {
    return false;
  }
  std::string call_error;
  try {
    auto& state = StateFor(instruction.state);
    auto& choice = request.own[instruction.state];
    // Multi-token cycles stop at EOS unless the request ignores it, on both
    // ranks alike.
    state.SetStopAtEos(request.stop_at_eos);
    switch (instruction.op) {
      case TpInstructionOp::kReuse:
        choice.reset();
        if (!prompt_fits)
          throw std::invalid_argument("reuse exceeds request prompt");
        runner_->PreparePrefixReuse(
            state, std::span<const TextRunnerToken>(request.prompt)
                       .first(instruction.prompt_size));
        request.digest.Add(runner_->CheckpointPosition(state));
        if (request.greedy)
          choice = ChooseGreedy(state);
        break;
      case TpInstructionOp::kCancelPrepare:
        runner_->PrepareCancellation(state);
        break;
      case TpInstructionOp::kPromptContext:
        DigestTpContext(request.digest, request.images);
        runner_->SetPromptContext(state, request.images);
        break;
      case TpInstructionOp::kInvalidate:
        choice.reset();
        state.Invalidate();
        break;
      case TpInstructionOp::kPrefill: {
        if (!prompt_fits) {
          throw std::invalid_argument("prefill exceeds the request prompt");
        }
        choice.reset();
        struct ClearLookahead {
          TextRunnerState& state;
          ~ClearLookahead() { state.SetPrefillLookahead({}); }
        } clear_lookahead{state};
        state.SetPrefillPromptEnd(instruction.prompt_size +
                                  instruction.lookahead);
        state.SetPrefillLookahead(
            std::span<const TextRunnerToken>(request.prompt)
                .subspan(instruction.prompt_size)
                .first(std::min<std::size_t>(
                    instruction.lookahead,
                    request.prompt.size() - instruction.prompt_size)));
        const auto step = runner_->Prefill(
            state, prefill_prompt, instruction.offset, instruction.count);
        DigestTpPrefill(request.digest, step,
                        runner_->CheckpointPosition(state));
        if (request.greedy && step.decode_ready) {
          choice = ChooseGreedy(state);
        }
        break;
      }
      case TpInstructionOp::kAdvance: {
        const auto token = static_cast<TextRunnerToken>(instruction.token);
        if (request.greedy &&
            (!choice.has_value() || choice->stop || choice->token != token)) {
          request.Note("rank 1 " +
                       (choice.has_value()
                            ? (choice->stop ? std::string("stopped")
                                            : "selected token " +
                                                  std::to_string(choice->token))
                            : std::string("had no token")) +
                       " where rank 0 advanced token " + std::to_string(token) +
                       " at instruction " + std::to_string(instruction.index));
        }
        choice.reset();
        runner_->Advance(state, token);
        request.sampler.Accept(token);
        DigestTpAdvance(request.digest, runner_->CheckpointPosition(state));
        if (request.greedy) {
          choice = ChooseGreedy(state);
        }
        break;
      }
      case TpInstructionOp::kDecode: {
        choice.reset();
        request.sampler.RestoreDrawState(
            DrawFromWire(instruction.rng, instruction.pending));
        // Rank 0's draft count, which it chose from its own history.
        const auto plan =
            instruction.drafts != 0
                ? std::optional<std::uint32_t>(instruction.drafts - 1)
                : std::nullopt;
        const auto step = runner_->DecodeStepPlanned(state, instruction.count,
                                                     request.sampler, plan);
        for (const auto& selection : step.selections) {
          request.sampler.Accept(selection.token);
        }
        DigestTpDecode(request.digest, step,
                       runner_->CheckpointPosition(state));
        if (request.greedy) {
          choice = ChooseGreedy(state);
        }
        break;
      }
      case TpInstructionOp::kAdvanceBatch:
      case TpInstructionOp::kDecodeBatch:
      case TpInstructionOp::kBeginRestore:
      case TpInstructionOp::kValidate:
      case TpInstructionOp::kReconcile:
      case TpInstructionOp::kStash:
      case TpInstructionOp::kLoadStashed:
      case TpInstructionOp::kEnd:
      case TpInstructionOp::kNone:
        throw std::logic_error("TP instruction has no model call");
    }
  } catch (const std::exception& exception) {
    call_error = "rank 1 " + std::string(OpName(instruction.op)) +
                 " failed: " + exception.what();
    DigestTpFailure(request.digest);
    request.Note(call_error);
  }
  if (forward && !EndScope(instruction.index, error)) {
    if (!call_error.empty() && error != nullptr) {
      *error = call_error + "; " + *error;
    }
    return false;
  }
  if (TpCacheAcknowledged(instruction.op)) {
    const TpControlResponse response{
        .instruction_index = instruction.index,
        .sequence = sequence,
        .error = call_error,
        .kind = TpControlResponseKind::kInstruction};
    return respond(response, error);
  }
  return true;
}

bool TpExecutor::ExecuteBatch(const TpInstruction& instruction,
                              std::string* error) {
  // Every member must name an open request and a state: a batch that does
  // not is a broken stream, not a failed call.
  std::vector<Request*> members;
  members.reserve(instruction.batch.size());
  for (const auto& member : instruction.batch) {
    const auto found = requests_.find(member.sequence);
    if (found == requests_.end() || member.state >= states_.size()) {
      SetError(error, "TP batch member names request " +
                          std::to_string(member.sequence) + " and state " +
                          std::to_string(member.state) +
                          ", which are not open");
      return false;
    }
    members.push_back(found->second.get());
  }
  const bool decode = instruction.op == TpInstructionOp::kDecodeBatch;
  std::vector<std::exception_ptr> failures(members.size());
  std::vector<TextRunnerAdvance> advances;
  std::vector<TextRunnerDecode> decodes;
  if (decode) {
    decodes.reserve(members.size());
  } else {
    advances.reserve(members.size());
  }
  for (std::size_t index = 0; index < members.size(); ++index) {
    const auto& member = instruction.batch[index];
    auto& request = *members[index];
    ++request.count;
    DigestTpCall(request.digest, MemberCall(instruction.op, member));
    auto& choice = request.own[member.state];
    if (decode) {
      choice.reset();
      request.sampler.RestoreDrawState(
          DrawFromWire(member.rng, member.pending));
      decodes.push_back(
          {.state = *states_[member.state],
           .max_tokens = member.count,
           .sampler = request.sampler,
           .drafts = member.drafts != 0
                         ? std::optional<std::uint32_t>(member.drafts - 1)
                         : std::nullopt});
      continue;
    }
    const auto token = static_cast<TextRunnerToken>(member.token);
    if (request.greedy &&
        (!choice.has_value() || choice->stop || choice->token != token)) {
      request.Note(
          "rank 1 " +
          (choice.has_value()
               ? (choice->stop
                      ? std::string("stopped")
                      : "selected token " + std::to_string(choice->token))
               : std::string("had no token")) +
          " where rank 0 advanced token " + std::to_string(token) +
          " in batched instruction " + std::to_string(instruction.index));
    }
    choice.reset();
    advances.push_back({.state = *states_[member.state],
                        .token = token,
                        .failure = &failures[index]});
  }
  if (!BeginScope(instruction.index, error)) {
    return false;
  }
  std::vector<TextDecodeStep> steps;
  try {
    if (decode) {
      // Rank 0's draft count, which it chose from its own timings.
      const auto plan =
          instruction.batch_drafts != 0
              ? std::optional<std::uint32_t>(instruction.batch_drafts - 1)
              : std::nullopt;
      steps = runner_->DecodeBatchPlanned(decodes, plan);
      if (steps.size() != members.size()) {
        throw std::runtime_error(
            "text runner returned an invalid decode batch size");
      }
      for (std::size_t index = 0; index < members.size(); ++index) {
        failures[index] = steps[index].failure;
      }
    } else {
      runner_->AdvanceBatch(advances);
    }
  } catch (...) {
    // The batch failed as a whole, including members it had run.
    const auto failure = std::current_exception();
    for (auto& member : failures) {
      if (!member) {
        member = failure;
      }
    }
  }
  if (!EndScope(instruction.index, error)) {
    return false;
  }
  for (std::size_t index = 0; index < members.size(); ++index) {
    const auto& member = instruction.batch[index];
    auto& request = *members[index];
    auto& state = *states_[member.state];
    if (failures[index]) {
      DigestTpFailure(request.digest);
      std::string what = "unknown error";
      try {
        std::rethrow_exception(failures[index]);
      } catch (const std::exception& exception) {
        what = exception.what();
      } catch (...) {
      }
      request.Note("rank 1 " + std::string(OpName(instruction.op)) +
                   " failed: " + what);
      continue;
    }
    // As in ExecuteCall: a member whose constraint refuses the token fails
    // its request, not the batch or the pair.
    try {
      if (decode) {
        for (const auto& selection : steps[index].selections) {
          request.sampler.Accept(selection.token);
        }
        DigestTpDecode(request.digest, steps[index],
                       runner_->CheckpointPosition(state));
      } else {
        const auto token = static_cast<TextRunnerToken>(member.token);
        request.sampler.Accept(token);
        DigestTpAdvance(request.digest, runner_->CheckpointPosition(state));
      }
    } catch (const std::exception& exception) {
      DigestTpFailure(request.digest);
      request.Note("rank 1 " + std::string(OpName(instruction.op)) +
                   " failed: " + exception.what());
      continue;
    }
    if (request.greedy) {
      try {
        request.own[member.state] = ChooseGreedy(state);
      } catch (const std::exception& exception) {
        request.Note(std::string("rank 1 greedy choice failed: ") +
                     exception.what());
      }
    }
  }
  return true;
}

}  // namespace gufo::server
