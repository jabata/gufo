// Times the dense Flash-Next projections per weight format (diagnostic):
// the Q8_0 decode GEMV against the Q4_K/Q5_K/Q6_K/Q5_1 vector kernel at 1, 4
// and 8 rows, and the Q8_0 W8A8 prefill GEMM against the K-quant tiled GEMM.
// Weights rotate through enough copies to exceed the last-level cache, as in
// a full decode step where every projection is read from DRAM.
//
// dense_kquant_bench [prefill_tokens] [reps]
//
// Each line prints the median time and the weight bytes per second.
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

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

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

struct Format {
  const char* name;
  int ggml;
  std::size_t block;
  std::size_t bytes;
};
constexpr Format kQ8_0{"Q8_0", 8, 32, 34};
constexpr Format kFormats[] = {
    {"Q4_K", 12, 256, 144},
    {"Q5_K", 13, 256, 176},
    {"Q6_K", 14, 256, 210},
    {"Q5_1", 7, 32, 24},
};

/// Random blocks with F16 scales of 1/512 (and minimums of 1/1024).
void* Weights(const Format& f, std::size_t m, std::size_t k,
              std::uint32_t seed) {
  const std::size_t blocks = m * (k / f.block);
  std::vector<std::uint8_t> w(blocks * f.bytes + 4096);
  for (auto& b : w)
    b = static_cast<std::uint8_t>(Next(&seed));
  const std::uint16_t d = 0x1800;
  const std::uint16_t mn = 0x1400;
  for (std::size_t i = 0; i < blocks; ++i) {
    std::uint8_t* block = w.data() + i * f.bytes;
    if (f.ggml == 14) {
      std::memcpy(block + 208, &d, 2);
    } else {
      std::memcpy(block, &d, 2);
      if (f.ggml != 8)
        std::memcpy(block + 2, &mn, 2);
    }
  }
  void* p = nullptr;
  CheckHip(hipMalloc(&p, w.size()), "hipMalloc");
  CheckHip(hipMemcpy(p, w.data(), w.size(), hipMemcpyHostToDevice),
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

/// Median of `reps` timed launches; `launch(i)` uses weight copy i.
double MedianUs(const std::function<void(int)>& launch, int reps) {
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (int i = 0; i < 3; ++i)
    launch(i);
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
    launch(i);
    CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
    CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
    CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
  }
  CheckHip(hipEventDestroy(start), "hipEventDestroy");
  CheckHip(hipEventDestroy(stop), "hipEventDestroy");
  std::sort(ms.begin(), ms.end());
  return 1000.0 * ms[reps / 2];
}

struct Shape {
  const char* name;
  std::size_t m;
  std::size_t k;
};

}  // namespace

int main(int argc, char** argv) try {
  const std::size_t prefill = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init");
  // Decode-time dense projections of one token (rows x reduction width).
  const Shape shapes[] = {
      {"ssm_in", 16384, 2560},   {"attn_qkv", 13312, 2560},
      {"ssm_out", 2560, 6144},   {"hc_down", 320, 10240},
      {"hc_up", 10240, 320},     {"shexp_gate", 640, 2560},
      {"shexp_down", 2560, 640}, {"output", 248320, 2560},
  };
  constexpr std::size_t kCacheBytes = std::size_t{512} << 20;
  for (const Shape& s : shapes) {
    std::vector<const Format*> formats{&kQ8_0};
    for (const Format& f : kFormats)
      if (s.k % f.block == 0)
        formats.push_back(&f);
    // The output head is timed for decode only.
    const bool tiled = prefill > 0 && s.m < 100000;
    const std::size_t rows = tiled ? std::max<std::size_t>(prefill, 8) : 8;
    float* x = Floats(rows * s.k, 7U);
    float* y = nullptr;
    CheckHip(hipMalloc(&y, rows * s.m * sizeof(float)), "hipMalloc");
    void* x_q8 = nullptr;
    CheckHip(hipMalloc(&x_q8, qfn_mmq_q8_1_bytes(8, static_cast<int>(s.k))),
             "hipMalloc");
    for (const Format* f : formats) {
      const std::size_t bytes = s.m * (s.k / f->block) * f->bytes;
      const int copies = static_cast<int>(
          std::clamp<std::size_t>(kCacheBytes / bytes + 1, 1, 64));
      std::vector<void*> w;
      for (int c = 0; c < copies; ++c)
        w.push_back(Weights(*f, s.m, s.k, 100U + c));
      for (const int n : {1, 4, 8}) {
        CheckHip(hipDeviceSynchronize(), "sync");
        if (qfn_mmq_quantize_q8_1(x, x_q8, n, static_cast<int>(s.k), nullptr) !=
            0)
          throw std::runtime_error("quantize");
        const double us = MedianUs(
            [&](int i) {
              const int rc = f->ggml == 8
                                 ? qfn_mmq_q8_0_dense_vec_preq(
                                       w[i % copies], nullptr, x_q8, y,
                                       static_cast<int>(s.m), n,
                                       static_cast<int>(s.k), nullptr)
                                 : qfn_mmq_kquant_dense_vec(
                                       f->ggml, w[i % copies], nullptr, x_q8, y,
                                       static_cast<int>(s.m), n,
                                       static_cast<int>(s.k), nullptr);
              if (rc != 0)
                throw std::runtime_error("vector launch");
            },
            reps);
        std::printf("decode  %-10s %6zux%-6zu %-4s n=%d %9.1f us %7.1f GB/s\n",
                    s.name, s.m, s.k, f->name, n, us, bytes / us / 1e3);
      }
      if (tiled) {
        void* xt = nullptr;
        if (f->ggml == 8) {
          CheckHip(hipMalloc(&xt, q::Q8TiledBytes(prefill, s.k)), "hipMalloc");
          q::QuantizeQ8Tiled(x, xt, prefill, s.k, nullptr);
        }
        const double us = MedianUs(
            [&](int i) {
              const bool ok =
                  f->ggml == 8
                      ? q::W8A8Gemm(w[i % copies], xt, y, prefill, s.m, s.k,
                                    nullptr)
                      : qfn_mmq_kquant_dense(
                            f->ggml, w[i % copies], x, y, static_cast<int>(s.m),
                            static_cast<int>(prefill), static_cast<int>(s.k),
                            nullptr) == 0;
              if (!ok)
                throw std::runtime_error("tiled launch");
            },
            reps);
        std::printf(
            "prefill %-10s %6zux%-6zu %-4s n=%zu %9.1f us %7.2f TFLOPS\n",
            s.name, s.m, s.k, f->name, prefill, us,
            2.0 * prefill * s.m * s.k / us / 1e6);
        if (xt != nullptr)
          (void)hipFree(xt);
      }
      for (void* p : w)
        (void)hipFree(p);
    }
    (void)hipFree(x_q8);
    (void)hipFree(y);
    (void)hipFree(x);
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "dense_kquant_bench: %s\n", e.what());
  return 1;
}
