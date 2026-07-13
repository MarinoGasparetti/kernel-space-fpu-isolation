# Chapter 2 — A Complete Transformer Language Model Running Inside the Linux Kernel

## Abstract

Chapter 1 established that sustained floating-point computation can run
inside the Linux kernel on a dedicated, isolated CPU core without starving
the scheduler, verified with a synthetic workload. This chapter closes the
gap between that mechanism and a real application: it documents a complete,
dependency-free transformer language model (Llama2 architecture) that
performs autoregressive text generation entirely in kernel space —
loading real trained weights, executing every transformer operation
(matmul, RMSNorm, RoPE, causal self-attention with KV-cache, SwiGLU
feed-forward), and emitting coherent text. The engine is not a port of an
existing userspace inference library; it is a purpose-written C
implementation with no external dependencies, not even libm, structured
from the outset to be kernel-resident. It was verified in three
environments — userspace (development host), an emulated kernel (QEMU), and
real physical deployment hardware — producing byte-identical output in all
three. On the physical target (a 4-core x86_64 machine, Linux 6.1.24) it
generates at approximately 45–64 tokens/second with a single core and no
SIMD, as a scalar baseline. Two supporting components verified along the
way — a bit-exact vector dot product from a real inference kernel, and a
model-scale memory loader — are documented first, as they close specific
open questions left by Chapter 1.

## 1. Motivation and Scope

Chapter 1's stress workload (`acc += 1.0001f; acc *= 0.9999f`) proved the
FPU-isolation mechanism but was explicitly a placeholder, not
inference-relevant code. Three questions were left open at the end of that
chapter, each of which had to be answered before a full engine could be
credible:

1. Does *real* numeric kernel code — the kind an inference engine actually
   executes — behave identically under the isolation-plus-pinned-thread
   scheme, or does the placeholder hide problems?
2. Can the kernel hold and correctly read a model-sized weight file
   (hundreds of MB to GB) in non-pageable memory?
3. The mainline path to an in-kernel LLM was assumed to be "port
   llama.cpp/ggml." Is that actually the right approach?

Sections 2 and 3 answer (1) and (2). Section 4 answers (3) in the negative
and documents the approach taken instead. Sections 5–7 present the engine
and its verification.

## 2. Bit-Exact Real Numeric Kernel Code

The placeholder loop of Chapter 1 was replaced with a real vector dot
product — the single most common operation in transformer inference,
appearing in every matrix multiplication — computed over two fixed
4096-element single-precision vectors inside `kernel_fpu_begin()` /
`kernel_fpu_end()` on the isolated core. Each iteration recomputes the same
dot product and compares its raw IEEE-754 bit pattern against a reference
captured on the first iteration.

On the physical target, **6,003,236 dot products** were computed with
**zero bit-level mismatches** — the numeric result was identical to the bit
across millions of repetitions under sustained kernel FPU use, with no
lockup or stall. This answers open question (1): real
inference-representative numeric code is bit-stable under the mechanism, not
merely the synthetic placeholder. (Source: the same
`appendix/andrea_mind_core.c` family from Chapter 1, with the dot-product
variant.)

## 3. Model-Scale Non-Pageable Memory

A separate module verified that the kernel can allocate a model-scale
buffer with `vmalloc` and populate it by reading a real file from disk with
`kernel_read` (chunked, no userspace-style `mmap`), then confirmed the
contents byte-exactly with a CRC32 over the whole buffer. A 256 MB buffer
was allocated, filled from a real 256 MB file, and its CRC32 matched the
reference computed independently.

One honest observation, recorded because it is load-bearing for scaling:
free memory dropped by the buffer size *twice* — once at `vmalloc`, once
during the actual read — consistent with `vmalloc` committing physical
pages lazily (on first write) rather than at allocation. At model scale
this matters: the pages backing a multi-gigabyte weight buffer are
non-pageable and remain resident for the lifetime of the load. On x86_64,
the `vmalloc` address space is roughly 32 TB, so the practical ceiling is
physical RAM, not address space. This answers open question (2). (Source:
the loader variant in the same module family.)

## 4. Why Not Port ggml: A Deliberate Rejection

The assumed path — porting `llama.cpp`/`ggml` into the kernel — was
investigated by reading the actual `ggml` source rather than reasoning
about it abstractly. The findings argue against that path:

- The CPU backend uses POSIX threads pervasively (37 `pthread` references in
  the core CPU backend alone); the kernel has no POSIX threads, only
  `kthread`/workqueue, so the entire parallelism layer would require
  rewriting, not merely recompiling.
- Weight loading is built on `mmap()` (`llama-mmap.cpp`), which is a
  userspace file-mapping model the kernel does not provide to its own code
  in the same form.
- `malloc`/`calloc` are used throughout, needing wholesale replacement with
  `kmalloc`/`vmalloc`/`kvmalloc` and attention to per-allocation size
  ceilings.
- The performance-critical kernels (e.g. `ggml_vec_dot_f32`) are tightly
  bound to intersecting SIMD dispatch macros (SVE/NEON/AVX in
  `ggml-cpu-impl.h`) and cannot be extracted cleanly.

Beyond the mechanical cost, porting a large userspace C++ library into the
kernel would produce exactly the outcome the broader project is trying to
avoid: a foreign component *bolted into* the kernel, with a visible seam
between "the system" and "the intelligence." The decision taken was to
write a minimal engine from scratch, in plain C, with no dependencies —
small enough to read in full, and kernel-native by construction rather than
by adaptation. This is feasible because a complete transformer forward pass
is arithmetically simple: the reference `llama2.c` project demonstrates that
a full Llama2 inference loop fits in a few hundred lines of dependency-free
C. The engine here follows that lead.

## 5. Engine Design

The engine implements the Llama2 architecture in full
(`appendix/andrea_llm_kmod.c`, the kernel module;
`appendix/andrea_llm_userspace.c`, the identical-logic userspace reference).
Per transformer layer, per generated token, it performs: RMSNorm; query /
key / value projections (`matmul`); rotary position embedding (RoPE)
applied to Q and K; causal self-attention over a key/value cache with
per-head softmax; the attention output projection; a second RMSNorm; and a
SwiGLU feed-forward block (`w2(silu(w1·x) ⊙ w3·x)`). A final RMSNorm and a
classifier projection produce logits over the vocabulary; the next token is
chosen by argmax (greedy decoding). Weights are memory-mapped from the
model file's layout by pointer arithmetic — in the kernel, into a
`vmalloc` buffer filled by `kernel_read` (Section 3), with no `mmap`.

### 5.1 No libm: Transcendental Functions by Hand

The kernel provides no math library. Every transcendental the engine needs
was reimplemented from scratch in a single header
(`appendix/andrea_llm_math.h`): `sqrtf` (Newton's method), `expf`
(range-reduction plus Taylor series), `logf` (atanh series), `powf`
(`exp(y·log x)`), and combined `sincosf` (Taylor series) for RoPE. These
were validated against the platform libm over their operating ranges
(`appendix/test_math.c`), with maximum errors on the order of 10⁻⁶–10⁻⁷ —
i.e. at the resolution limit of single precision. The decisive test was not
the microbenchmark but the end-to-end one: compiling the *entire* engine
against these hand-written functions instead of libm produced text output
**byte-identical** to the libm build. The hand-rolled math is
indistinguishable from the system library in the only context that matters.

### 5.2 Kernel Integration

The module allocates all state with `vmalloc`/`vzalloc`, loads model and
tokenizer with `kernel_read`, and runs the forward pass inside
`kernel_fpu_begin()`/`kernel_fpu_end()` on the isolated core, with
`cond_resched()` between tokens. It exposes generated text two ways: the
full output via a `/proc/andrea_llm` entry (`seq_file`), and — optionally —
live, token-by-token, by writing each decoded piece directly to
`/dev/console` (`filp_open` + `kernel_write`) as it is produced, including
decoding raw byte-tokens of the form `<0xNN>` back into their literal bytes
(e.g. `<0x0A>` → newline) so the on-screen text renders cleanly. The same
`kernel_fpu`/`kernel_neon` dual-architecture macros from Chapter 1 are used,
so the engine compiles for both x86_64 and aarch64.

## 6. Verification

The engine was verified in three environments, checked for byte-identical
output at each step rather than "looks plausible":

- **Userspace (development host):** generates coherent text from the real
  TinyStories 15M-parameter Llama2 checkpoint (public weights). This is the
  reference output.
- **Userspace, no libm:** the dependency-free build (Section 5.1) produces
  identical output to the libm build.
- **Emulated kernel (QEMU x86_64):** the module loads, reads the model with
  `kernel_read`, runs inference under kernel FPU, and `/proc/andrea_llm`
  returns text identical to userspace. Zero crashes or lockups.
- **Physical hardware (Asus, 4-core x86_64, Linux 6.1.24):** identical
  output again, this time on real silicon, generated inside the running
  server's kernel. Raw capture: `docs/asus-real-hardware-llm.log`.

### 6.1 Performance (Scalar Baseline)

On the physical target, a 256-step generation completed in approximately
4–5 seconds of wall-clock time, and the model emitted an end-of-sequence
token on its own at 222 tokens, yielding roughly **45–64 tokens/second**.
This is a deliberately unoptimized baseline: a single core, scalar
`matmul` (no SIMD), greedy decoding, 15M parameters. It is reported as a
starting point for the scaling discussion below, not as a competitive
figure.

### 6.2 What "The Same Engine Scales" Means Concretely

A 70B-parameter model uses the *identical* operations this 15M model uses —
`matmul`, RMSNorm, softmax, RoPE, attention. Nothing in the forward-pass
logic changes; only the numbers read from the weight file (dimensions,
layer count, head count) differ. The gap between this baseline and a
large-model deployment is therefore quantitative, not architectural, and
falls into three known, separable pieces of future work: multi-core
parallelism (one pinned `kthread` per isolated core — the extension of
Chapter 1's single-core mechanism), quantization (Q4 brings 70B to ~35 GB,
within `vmalloc`'s address space and bounded only by physical RAM per
Section 3), and hand-written SIMD in the hot `matmul` (the same work `ggml`
did in userspace, now inside `kernel_fpu`). None of these change the engine;
they accelerate it.

## 7. Honest Limitations

- **Greedy decoding only.** Next-token selection is argmax; there is no
  temperature, top-k, or top-p sampling. Output is deterministic — a
  strength for verification (byte-identical comparison across environments)
  but not representative of a general text generator.
- **Scalar, single-core.** No SIMD, no multi-core. The performance figure
  (Section 6.1) is a floor, not a representative production number, and is
  only meaningful as the baseline the scaling work (Section 6.2) starts
  from.
- **Inference only — the model does not learn or grow.** The weights are
  strictly read-only. There is no training, no fine-tuning, no structural
  growth. Making the model adapt or expand is a separate and much larger
  undertaking, out of scope here and deliberately not claimed.
- **Small model.** TinyStories 15M is a real Llama2 but a small one, chosen
  because it makes every component verifiable end-to-end at each step. Large
  models are argued to be reachable (Section 6.2) but are not demonstrated
  here; that argument is an engineering projection, not a result.
- **aarch64 is compile-verified, not run-verified,** for the same
  environmental reason as Chapter 1: the arm64 test boot environment does
  not yet reach a shell to load the module. The engine compiles cleanly for
  aarch64 with the `kernel_neon` path; it has not been executed there.

## 8. Next Steps

In the order dictated by the results:

1. Multi-core parallelism: distribute `matmul` rows across several isolated
   cores, one pinned `kthread` each — the direct extension of Chapter 1's
   mechanism and the single largest available speedup.
2. Hand-written SIMD (AVX-512 on x86_64, NEON on aarch64) in the `matmul`
   hot path, executed inside the existing `kernel_fpu`/`kernel_neon`
   sections.
3. Quantized weights (Q4) to bring large models within physical-RAM reach.
4. A larger model (TinyStories 110M, then a quantized general model) once
   the above make it practical.
