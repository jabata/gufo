// Times the routed-expert decode projections of one Flash-Next layer
// (diagnostic): the Q4_K gate/up vector product with SwiGLU, as the
// executor and the request batch call it for 1 to 64 verify rows, with the
// lane kernel (production) and the scalar kernel it replaced, then the Q5_1
// down projection over the rows' top-10 slots. Each row routes to ten
// distinct experts of 512, drawn uniformly (eight rows touch about 75
// experts); `overlap` is the chance that a row reuses the previous row's
// experts. Every repetition draws new routing over full-size expert
// tensors, so the active experts come from DRAM as in a decode step.
//
// expert_bench [reps] [max_rows] [overlap]
//
// Each line prints the projection, the kernel, the rows, the mean distinct
// experts, the median time, the distinct expert bytes per second (the
// bandwidth if every active expert is read once) and the time per row. The
// `dense` lines read the same number of gate rows with the dense K-quant
// vector kernel for one token: the streaming rate of contiguous rows. A
// summary per row count adds gate/up and down per layer and over 48 layers.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kHidden = 2560;
constexpr int kFf = 640;
constexpr int kLayers = 48;
constexpr int kQ4K = 12;
constexpr int kQ5_1 = 7;
// One expert's bytes: gate or up (640 x 2560 Q4_K), down (2560 x 640 Q5_1).
constexpr std::size_t kGateBytes = std::size_t{kFf} * (kHidden / 256) * 144;
constexpr std::size_t kDownBytes = std::size_t{kHidden} * (kFf / 32) * 24;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Random blocks with F16 scales of 1/512 and minimums of 1/1024. A 64 MiB
/// pattern fills the tensor; the bench reads each expert at most once per
/// repetition, so repeated bytes do not change the traffic.
void* Weights(std::size_t blocks, std::size_t block_bytes, std::uint32_t seed) {
  const std::size_t bytes = blocks * block_bytes;
  const std::size_t chunk_blocks =
      std::min(blocks, (std::size_t{64} << 20) / block_bytes);
  std::vector<std::uint8_t> w(chunk_blocks * block_bytes);
  for (auto& b : w)
    b = static_cast<std::uint8_t>(Next(&seed));
  const std::uint16_t d = 0x1800;
  const std::uint16_t mn = 0x1400;
  for (std::size_t i = 0; i < chunk_blocks; ++i) {
    std::memcpy(w.data() + i * block_bytes, &d, 2);
    std::memcpy(w.data() + i * block_bytes + 2, &mn, 2);
  }
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes + 4096), "hipMalloc");
  for (std::size_t off = 0; off < bytes; off += w.size())
    CheckHip(hipMemcpy(static_cast<std::uint8_t*>(p) + off, w.data(),
                       std::min(w.size(), bytes - off), hipMemcpyHostToDevice),
             "hipMemcpy");
  return p;
}

float* Floats(std::size_t count, std::uint32_t seed) {
  std::vector<float> x(count);
  for (auto& v : x)
    v = static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F;
  void* p = nullptr;
  CheckHip(hipMalloc(&p, count * sizeof(float)), "hipMalloc");
  CheckHip(hipMemcpy(p, x.data(), count * sizeof(float), hipMemcpyHostToDevice),
           "hipMemcpy");
  return static_cast<float*>(p);
}

/// Top-10 routing of `rows` rows; returns the distinct experts.
int Route(std::vector<std::int32_t>* ids, int rows, double overlap,
          std::uint32_t* state) {
  ids->assign(std::size_t(rows) * kUsed, 0);
  std::vector<bool> seen(kExperts, false);
  int distinct = 0;
  for (int t = 0; t < rows; ++t) {
    std::int32_t* row = ids->data() + std::size_t(t) * kUsed;
    if (t > 0 && (Next(state) % 1000) < overlap * 1000.0) {
      std::copy(row - kUsed, row, row);
      continue;
    }
    for (int j = 0; j < kUsed; ++j) {
      std::int32_t e;
      do {
        e = static_cast<std::int32_t>(Next(state) % kExperts);
      } while (std::find(row, row + j, e) != row + j);
      row[j] = e;
    }
  }
  for (const std::int32_t e : *ids)
    if (!seen[e]) {
      seen[e] = true;
      ++distinct;
    }
  return distinct;
}

struct Timing {
  double us;
  double distinct;
};

/// Median of `reps` launches; repetition i uploads routing i first (outside
/// the timed region).
Timing Median(const std::function<int(int)>& prepare,
              const std::function<void()>& launch, int reps) {
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (int i = 0; i < 3; ++i) {
    prepare(reps + i);
    launch();
  }
  std::vector<float> ms(reps);
  double distinct = 0.0;
  for (int i = 0; i < reps; ++i) {
    distinct += prepare(i);
    CheckHip(hipDeviceSynchronize(), "sync");
    CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
    launch();
    CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
    CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
    CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
  }
  CheckHip(hipEventDestroy(start), "hipEventDestroy");
  CheckHip(hipEventDestroy(stop), "hipEventDestroy");
  std::sort(ms.begin(), ms.end());
  return {1000.0 * ms[reps / 2], distinct / reps};
}

void Print(const char* projection, const char* kernel, int rows,
           const Timing& t, std::size_t expert_bytes) {
  const double bytes = t.distinct * static_cast<double>(expert_bytes);
  std::printf(
      "%-7s %-6s n=%-2d experts=%6.1f %9.1f us %7.1f GB/s %8.2f us/row\n",
      projection, kernel, rows, t.distinct, t.us, bytes / t.us / 1e3,
      t.us / rows);
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  const int max_rows = std::clamp(argc > 2 ? std::atoi(argv[2]) : 64, 1, 64);
  const double overlap = argc > 3 ? std::atof(argv[3]) : 0.0;
  if (reps < 1)
    throw std::runtime_error("reps must be positive");
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init");
  const std::size_t q4_blocks = std::size_t{kExperts} * kFf * (kHidden / 256);
  const std::size_t q5_blocks = std::size_t{kExperts} * kHidden * (kFf / 32);
  void* gate = Weights(q4_blocks, 144, 11U);
  void* up = Weights(q4_blocks, 144, 23U);
  void* down = Weights(q5_blocks, 24, 37U);
  float* x = Floats(std::size_t(max_rows) * kHidden, 7U);
  void* x_q8 = nullptr;
  CheckHip(hipMalloc(&x_q8, qfn_mmq_q8_1_bytes(max_rows, kHidden)),
           "hipMalloc");
  if (qfn_mmq_quantize_q8_1(x, x_q8, max_rows, kHidden, nullptr) != 0)
    throw std::runtime_error("quantize");
  const std::size_t slots = std::size_t(max_rows) * kUsed;
  float* act = Floats(slots * kFf, 5U);
  float* out = nullptr;
  CheckHip(hipMalloc(&out, slots * kHidden * sizeof(float)), "hipMalloc");
  std::int32_t* ids = nullptr;
  CheckHip(hipMalloc(&ids, slots * sizeof(std::int32_t)), "hipMalloc");
  std::vector<std::int32_t> host_ids;
  std::printf(
      "# reps=%d overlap=%.2f; distinct expert bytes: gate+up %zu, "
      "down %zu per expert\n",
      reps, overlap, 2 * kGateBytes, kDownBytes);
  for (const int n : {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64}) {
    if (n > max_rows)
      break;
    std::uint32_t state = 0x9E3779B9U ^ static_cast<std::uint32_t>(n);
    // Routing i is the same for every kernel at this width.
    std::vector<std::vector<std::int32_t>> routes(reps + 3);
    std::vector<int> distinct(reps + 3);
    for (int i = 0; i < reps + 3; ++i)
      distinct[i] = Route(&routes[i], n, overlap, &state);
    const auto prepare = [&](int i) {
      CheckHip(hipMemcpy(ids, routes[i].data(),
                         routes[i].size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice),
               "ids upload");
      return distinct[i];
    };
    Timing gateup[2];
    for (int route = 0; route < 2; ++route) {
      gateup[route] = Median(
          prepare,
          [&] {
            if (qfn_mmq_moe_gated_vec_route(kQ4K, gate, up, x_q8, ids, out, kFf,
                                            kHidden, n, kExperts, kUsed, route,
                                            nullptr) != 0)
              throw std::runtime_error("gated launch");
          },
          reps);
      Print("gate_up", route == 0 ? "lane" : "scalar", n, gateup[route],
            2 * kGateBytes);
    }
    // The executor's down projection: one slot row per (row, expert).
    const Timing down_t = Median(
        prepare,
        [&] {
          if (qfn_mmq_moe_vec(kQ5_1, down, act, ids, out, kHidden, kFf,
                              n * kUsed, kExperts, 1, nullptr) != 0)
            throw std::runtime_error("down launch");
        },
        reps);
    Print("down", "vec", n, down_t, kDownBytes);
    // The same number of gate rows, contiguous, through the dense vector
    // kernel for one token; the first expert rotates so they are not cached.
    const int d = static_cast<int>(gateup[0].distinct + 0.5);
    int first = 0;
    const Timing dense_t = Median(
        [&](int i) {
          std::uint32_t s = 0xD1B54A35U + static_cast<std::uint32_t>(i);
          first = static_cast<int>(Next(&s) % (kExperts - d + 1));
          return d;
        },
        [&] {
          if (qfn_mmq_kquant_dense_vec_route(
                  kQ4K, static_cast<std::uint8_t*>(gate) + first * kGateBytes,
                  x_q8, out, d * kFf, 1, kHidden, 1, nullptr) != 0)
            throw std::runtime_error("dense launch");
        },
        reps);
    Print("gate", "dense", n, dense_t, kGateBytes);
    std::printf(
        "layer   n=%-2d gate_up(lane)+down %8.1f us (scalar %8.1f us); "
        "x%d layers %6.2f ms (scalar %6.2f ms)\n",
        n, gateup[0].us + down_t.us, gateup[1].us + down_t.us, kLayers,
        kLayers * (gateup[0].us + down_t.us) / 1e3,
        kLayers * (gateup[1].us + down_t.us) / 1e3);
  }
  for (void* p :
       {gate, up, down, static_cast<void*>(x), x_q8, static_cast<void*>(act),
        static_cast<void*>(out), static_cast<void*>(ids)})
    (void)hipFree(p);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "expert_bench: %s\n", e.what());
  return 1;
}
