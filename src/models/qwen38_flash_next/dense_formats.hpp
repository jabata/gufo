#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DENSE_FORMATS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DENSE_FORMATS_HPP_

#include <array>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

/// Storage formats of the dense (non-routed) projections: attention and
/// DeltaNet projections, shared experts, hyper-connection mixers, PLE and the
/// output head. Each tensor may use its own format.
namespace gufo::models::qwen38_flash_next {

/// The sub-8-bit block formats the dense projections accept besides Q8_0 and
/// the 16/32-bit formats: Q4_K, Q5_K and Q6_K (256-element superblocks), and
/// Q5_1 for reduction widths that are not a multiple of 256 (the 640-wide
/// shared-expert down and the 320-wide mixer up projections).
[[nodiscard]] constexpr bool IsDenseKQuant(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ5_1;
}

/// Every format a dense projection may be stored in.
inline constexpr std::array<core::GgmlType, 8> kDenseMatrixTypes = {
    core::GgmlType::kQ8_0, core::GgmlType::kQ4_K, core::GgmlType::kQ5_K,
    core::GgmlType::kQ6_K, core::GgmlType::kQ5_1, core::GgmlType::kBF16,
    core::GgmlType::kF16,  core::GgmlType::kF32};

/// Kernels for a dense IsDenseKQuant projection. Decode, verification and
/// batched decode take the vector kernel, whose per-row arithmetic is
/// independent of the batch width; prefill takes the tiled GEMM at every chunk
/// width, so prompt arithmetic does not depend on where the chunks fall.
enum class KQuantDenseKernel { kVector, kTiled };

[[nodiscard]] constexpr KQuantDenseKernel SelectKQuantDenseKernel(
    bool prefill) noexcept {
  return prefill ? KQuantDenseKernel::kTiled : KQuantDenseKernel::kVector;
}

/// Projections of one input may be uploaded stacked along rows (one GEMV)
/// when they share one of these formats; rows of a block format concatenate
/// without re-encoding.
[[nodiscard]] constexpr bool IsStackableDense(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ8_0 || IsDenseKQuant(type);
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DENSE_FORMATS_HPP_
