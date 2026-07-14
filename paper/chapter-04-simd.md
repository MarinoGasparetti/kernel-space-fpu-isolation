# Chapter 4 — Vectorized Matrix Multiplication (SIMD) Inside the Kernel

## Abstract

Chapters 2 and 3 built and parallelized a kernel-resident language model
whose inner loop — the matrix-multiplication dot product — was scalar: one
multiply-add per iteration. This chapter vectorizes it, executing 8
multiply-adds per instruction (AVX2 on x86_64) or 4 (NEON on aarch64),
inside the existing per-thread kernel FPU sections. The result on real
hardware is a 3.6× single-core speedup and 5.3× combined with the
multi-core work of Chapter 3, over the scalar serial baseline — with
byte-identical output and zero crashes. The chapter documents the two real
obstacles to SIMD specifically in kernel space (the compiler emitting vector
instructions outside FPU-guarded regions, and the absence of the userspace
intrinsic headers) and how each was resolved, since both are the kind of
thing that turns a straightforward optimization into a kernel that fails to
build or, worse, corrupts userspace FPU state.

## 1. The Opportunity

A dot product `sum += a[i]*b[i]` executed one element at a time leaves most
of the CPU's floating-point throughput unused. Modern cores have wide vector
units: AVX2 processes 8 single-precision floats per instruction, NEON 4, and
fused multiply-add (FMA) does the multiply and the accumulate in one. Applied
to the dot product that dominates every matrix multiplication in the forward
pass, this is the single largest per-core speedup available, and it composes
with the multi-core parallelism of Chapter 3 (more cores each doing more per
instruction).

## 2. Two Kernel-Specific Obstacles

Vectorizing a userspace program is routine. Doing it in kernel space
surfaced two problems that do not exist in userspace, both anticipated
before writing the code and both load-bearing.

### 2.1 The Compiler Must Not Emit SIMD Outside FPU Sections

Kernel code may only touch the vector registers between `kernel_fpu_begin()`
and `kernel_fpu_end()` (x86_64) or `kernel_neon_begin()`/`kernel_neon_end()`
(aarch64); using them elsewhere silently corrupts the FPU state of whatever
userspace task was interrupted. But enabling vectorization for a translation
unit (`-mavx2`) gives the compiler license to emit vector instructions
*anywhere* in that file — including in code that runs outside any FPU
section. This is a real trap: the code compiles and mostly works, then
occasionally corrupts an unrelated process's floating-point state.

The resolution is to **isolate the vectorized code in its own translation
unit** (`appendix/andrea_llm_simd_kern.c`, containing only the dot-product
function `andrea_dot`), compiled with the SIMD flags, while the rest of the
module (`appendix/andrea_llm_kmod.c`) is compiled *without* them. The kernel
Makefile applies the flags per-object:

```make
andrea_llm-objs := andrea_llm_kmod.o andrea_llm_simd_kern.o
CFLAGS_andrea_llm_simd_kern.o += -mavx2 -mfma          # x86_64
CFLAGS_REMOVE_andrea_llm_simd_kern.o += -mgeneral-regs-only  # aarch64
```

`andrea_dot` is called only from `do_part`, which runs inside a kernel FPU
section (Chapter 3). This was verified, not assumed: `objdump` on the two
object files confirms the SIMD unit contains `vfmadd231ps`/`ymm` (x86) and
`fmla`/`v0.4s` (arm), and the core unit contains **no** AVX registers at all.
This is the same discipline the kernel's own AES-NI/crypto code uses, for the
same reason.

### 2.2 No Intrinsic Headers in the Kernel Build

The natural way to write AVX2 — `#include <immintrin.h>` and call
`_mm256_fmadd_ps` — fails outright in a kernel module: the build (`-nostdinc`
plus a kernel-only include path) does not provide `immintrin.h`
(`fatal error: immintrin.h: No such file or directory`). Rather than fall
back to inline assembly, the code uses **GCC vector extensions**, which are a
language feature of the compiler itself and need no headers:

```c
typedef float v8sf __attribute__((vector_size(32), aligned(4)));
...
acc0 += (*(const v8sf *)(a + i)) * (*(const v8sf *)(b + i));
```

Ordinary `+` and `*` on the vector type compile to packed AVX2 operations
under `-mavx2`, and the multiply-accumulate fuses to FMA under `-mfma`. The
`aligned(4)` attribute is essential and easy to miss: the model weights are
not 32-byte aligned, so the vector type must declare a 4-byte alignment
requirement, which makes the compiler emit unaligned loads (`vmovups`)
rather than alignment-faulting aligned ones. The aarch64 path is identical
in structure with a 16-byte (`v4sf`) type.

## 3. On Losing — and Not Losing — Bit-Identity

It was expected that SIMD would break the bit-identical property that
Chapters 2 and 3 relied on for verification: vectorized reduction sums the
products in a different order than scalar left-to-right accumulation, and
floating-point addition is not associative, so the final dot product can
differ in its last bit. In practice, the generated **text was byte-identical**
to the scalar version anyway. The reason is the decoder: next-token selection
is greedy `argmax`, and a discrepancy on the order of 10⁻⁶ in a logit almost
never changes *which* logit is the maximum, because the logits are
well-separated. So the intermediate arithmetic differs at the last bit while
the sequence of chosen tokens — and thus the output — does not. This is
stated with the honest caveat that it holds for this model under greedy
decoding; a model with near-tied top logits, or a sampling decoder, could
diverge. Verification for this chapter therefore checks output coherence and
equality-in-practice rather than claiming guaranteed bit-identity.

## 4. Results

Measured on the real deployment target (i3-1115G4, AVX2+FMA, 4 cores), over
ten full generations per configuration, scalar serial as the baseline:

| Configuration | time (10 gen) | speedup vs scalar serial |
|---|---|---|
| scalar, 1 thread | 32 s | 1× |
| scalar, 4 threads (Chapter 3) | 11 s | 2.9× |
| **SIMD, 1 thread** | 9 s | **3.6×** |
| **SIMD, 4 threads** | 6 s | **5.3×** |

![Optimization cascade](../docs/benchmark-01-optimization-cascade.png)

*The full path from the scalar single-threaded baseline through Chapter 3's
multi-core to this chapter's SIMD, on the same hardware: 5.3× end to end,
and SIMD on one core (9 s) beating scalar on all four (11 s).*

Three observations:

- **SIMD alone gives 3.6× on a single core** — real AVX2 on a modest CPU.
- **SIMD on one core (9 s) beats scalar on four cores (11 s).** On this small
  model, vectorization is a more effective use of the hardware than
  multi-core parallelism, because the model's small matmuls spend a large
  fraction of multi-core time in thread-synchronization overhead (a point
  already noted in Chapter 3) that SIMD sidesteps entirely.
- **Combined, 5.3× over scalar serial**, and 1.8× over Chapter 3's
  parallel-but-scalar configuration. Output byte-identical, zero crashes, no
  `invalid opcode` faults (the failure mode that would indicate SIMD being
  executed where it is not permitted).

The engine, with SIMD, was consolidated into the deployed system image and
verified again after reflash on real hardware: the 4-thread SIMD generation
time reproduced at 6 s, confirming the speedup is permanent, not a
measurement of a warm anomaly.

## 5. Honest Limitations

- **The small model masks the combined ceiling.** Because the 15M model's
  matmuls are small, SIMD+4-thread (6 s) is only modestly faster than SIMD
  alone (9 s) — the parallel dimension has little left to contribute once
  SIMD has made each matmul cheap. On a large model, where matmuls are big
  enough to keep all cores busy, SIMD and multi-core would compound more
  fully; that combined regime is argued, not yet measured here.
- **Memory bandwidth remains the ultimate ceiling** for large models on this
  hardware (Chapter 3, §4.1). SIMD accelerates the compute side; it does not
  widen the memory bus. On a large model streaming weights from RAM, SIMD's
  benefit is bounded by how fast the weights can be delivered, not how fast
  they can be multiplied.
- **AVX-512 unused.** This CPU (Tiger Lake) supports AVX-512 (16 floats per
  instruction), but AVX2 (8) was chosen for portability and safety. The
  additional 2× from AVX-512 is available but not taken here.
- **Bit-identity is empirical, not guaranteed** (§3): true for this model and
  greedy decoding, not a general property.
- **Still inference only.** As throughout, the model does not learn or grow.
- **aarch64 compile-verified only:** the SIMD unit compiles for arm64 with
  NEON vector extensions (`fmla`/`v0.4s` confirmed by `objdump`), but the
  arm64 runtime environment still cannot execute it (Chapters 1–3).

## 6. Next Steps

1. Measure the combined SIMD + multi-core scaling on a large model, where
   both dimensions have room to contribute.
2. AVX-512 on capable hardware, for a further ~2× per core.
3. Quantization (Q4), which attacks the memory-bandwidth ceiling that SIMD
   does not.
