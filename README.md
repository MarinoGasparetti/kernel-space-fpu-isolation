# Kernel-Space FPU Isolation: An Experimental Case Study in Running Inference Compute — and a Full Language Model — Inside a Linux Kernel

This is a private, working case study documenting a progressive attempt to
run AI inference computation, and ultimately a complete transformer language
model, directly inside a Linux kernel — as kernel-resident code rather than
a userspace service. The repository is organized as a sequence of chapters,
each corresponding to one architectural or experimental milestone, written
and expanded as the work progresses rather than after the fact. Chapters are
added, not rewritten — a later chapter may revise an earlier chapter's
conclusions, but the earlier chapter's account of what was tried and found
stays intact as a historical record. (Chapter 3, for instance, deliberately
supersedes Chapter 1's core-isolation approach; Chapter 1 remains as the
record of what that approach was and why it was tried.)

## Chapters

- [`paper/chapter-01-kernel-fpu-isolation.md`](paper/chapter-01-kernel-fpu-isolation.md) — **The obstacle and the mechanism.** Sustained floating-point computation is structurally awkward inside the Linux kernel because `kernel_fpu_begin()`/`kernel_fpu_end()` disable preemption and were designed for short bursts. Resolution: dedicate one physical core via `isolcpus`/`nohz_full`/`rcu_nocbs`, pin a kernel thread to it, and hold the FPU open freely because nothing else is scheduled there. Verified in QEMU and on real hardware — ~2.1×10¹⁰ floating-point operations in-kernel, zero lockups.
- [`paper/chapter-02-llm-in-kernel.md`](paper/chapter-02-llm-in-kernel.md) — **A complete Llama2 language model running inside the kernel.** A dependency-free transformer engine written from scratch in plain C (no libm — transcendental functions reimplemented by hand and validated against the system library), loading real trained weights, executing every transformer operation in kernel space, and generating coherent text. Verified byte-identical across userspace, QEMU, and real physical hardware; ~45–64 tokens/second as a scalar single-core baseline. Includes a deliberate, documented rejection of the "port ggml" approach in favor of a kernel-native engine.
- [`paper/chapter-03-symbiotic-multicore.md`](paper/chapter-03-symbiotic-multicore.md) — **Symbiotic multi-core inference, superseding Chapter 1's isolation.** Static core isolation (`isolcpus`) is reconsidered and deliberately abandoned: it cannot be dynamic and it partitions the machine rather than sharing it. Replaced with kernel threads on shared cores, short bracketed FPU sections, and `cond_resched` — letting the scheduler decide moment-to-moment how much of the machine the inference occupies. Row-partitioned matrix multiplication across a kernel-thread pool, bit-identical to serial (hash-verified). Real-hardware results (2.67×–2.77× on 4 cores) plus an honest memory-bandwidth ceiling on large models. Consolidated into the deployed image with no bundled weights (engine permanent, model interchangeable).
- [`paper/chapter-04-simd.md`](paper/chapter-04-simd.md) — **Vectorizing the inner loop (SIMD) inside the kernel.** The dot product is vectorized to 8 multiply-adds per instruction (AVX2) / 4 (NEON) inside the existing per-thread FPU sections, for a 3.6× single-core and 5.3× combined speedup over scalar serial on real hardware, output byte-identical, zero crashes. Documents the two kernel-specific obstacles — the compiler emitting SIMD outside FPU-guarded regions (solved by isolating the vector code in its own translation unit, objdump-verified), and the absence of `immintrin.h` in the kernel build (solved with GCC vector extensions) — and why greedy decoding preserves byte-identical output despite non-associative vectorized reduction. Consolidated and re-verified after reflash.

## Benchmarks

[`docs/benchmarks.md`](docs/benchmarks.md) — graphs generated from the real
measurements across all chapters: the optimization cascade (scalar →
multi-core → SIMD, 5.3× on real hardware), the isolcpus-vs-symbiotic
scaling evidence, the memory-bandwidth ceiling on large matmuls, and
model-size scaling.

## Directory layout

- `paper/` — the chapters
- `appendix/` — full source: the Chapter 1 FPU-isolation module (`andrea_mind_core.*`), and the Chapter 2 inference engine (`andrea_llm_userspace.c` reference, `andrea_llm_kmod.c` kernel module, `andrea_llm_math.h` hand-written math, `test_math.c` validation, `andrea_llm.Makefile` dual-architecture build)
- `docs/` — raw logs from real runs: Chapter 1's QEMU and physical-hardware dmesg captures, and Chapter 2's physical-hardware LLM generation log (`asus-real-hardware-llm.log`)

## One-paragraph summary

The project starts from a narrow architectural obstacle — sustained
floating point in kernel space starving the scheduler — and resolves it by
borrowing Linux's own CPU-isolation mechanism (Chapter 1). It then closes
the distance from that mechanism to a real application by building, from
scratch and with no dependencies, a complete transformer language model that
runs entirely inside the kernel: reading real weights with `kernel_read`
into `vmalloc` memory, executing matmul / RMSNorm / RoPE / attention /
SwiGLU under kernel FPU on the isolated core, and emitting coherent text
verified byte-identical on real hardware (Chapter 2). The engine is
deliberately *not* a port of an existing userspace library — it is small
enough to read in full and kernel-native by construction, consistent with
the broader premise that the kernel should *be* the software rather than
host it. What is demonstrated is inference only; the model does not yet learn
or grow, and the scalar single-core performance is a baseline, not a
production figure — both stated plainly rather than obscured.
