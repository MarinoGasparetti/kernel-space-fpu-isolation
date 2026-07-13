# Chapter 3 — Symbiotic Multi-Core Inference, and Why It Supersedes Chapter 1's Isolation

## Abstract

Chapter 1 dedicated a physical CPU core to kernel-side computation by
isolating it from the scheduler (`isolcpus`), so that a kernel thread could
hold the FPU open indefinitely without starving anything. Chapter 2 built a
complete language model on top of that, running single-core. This chapter
reconsiders the isolation premise itself. Static core isolation is, by
construction, the opposite of what a system whose intelligence should grow
with its resources wants: `isolcpus` is a boot-time parameter, fixed until
reboot, and it partitions the machine rather than sharing it. This chapter
documents the deliberate replacement of that approach with a *symbiotic*
one — kernel threads running on the ordinary, shared cores, with short
bracketed FPU sections and voluntary rescheduling, letting the kernel's own
scheduler decide moment-to-moment how much of the machine the inference
work occupies. The matrix multiplication (the inference bottleneck) is
partitioned by rows across a pool of kernel threads. The partition is
bit-identical to the serial computation, verified by hash equality. On real
4-core hardware the mechanism reaches 2.67×–2.77× speedup, with the honest
finding that on memory-bandwidth-limited hardware a larger model's parallel
scaling is throttled by RAM bandwidth, not core count.

## 1. Why Isolation Was the Wrong Foundation

Chapter 1's mechanism is correct and its measurements stand. But its
premise — take a core away from the system and give it exclusively to the
computation — encodes a static, exclusive relationship between system and
computation. For the broader goal this project is built around (a machine
whose intelligence can scale up to occupy more of its own resources as they
are available, and yield them when the system needs them), that premise is
backwards on two counts:

1. **`isolcpus` cannot be dynamic.** It is a kernel command-line parameter
   read once at boot. An isolated core stays isolated until reboot. There
   is no runtime path to "give the mind three cores now, one core when a
   request arrives." The relationship is frozen at boot time.
2. **It partitions rather than shares.** An isolated core is unavailable to
   the rest of the system even when the inference workload is idle, and the
   inference workload cannot expand onto other cores even when they are
   idle. Both sides are worse off than they would be under a shared
   scheduler.

The design goal was restated explicitly during this work as *symbiosis* —
system and intelligence sharing the same substrate, the division of
resources emerging from moment-to-moment load rather than being declared in
advance. `isolcpus` cannot express that. This chapter therefore abandons it.

## 2. The Symbiotic Mechanism

The resolution is to stop isolating and let the scheduler do its job:

- Kernel worker threads are **not** pinned or isolated. They run on the
  ordinary shared cores. When the server is idle, the scheduler places the
  inference threads on all available cores; when an HTTP request or other
  system work arrives, the scheduler preempts and redistributes. Nobody
  decides how many cores the mind gets — it emerges from load.
- The FPU is held only in **short bracketed sections**. Chapter 1 could
  hold `kernel_fpu_begin()` open for a long computation precisely because
  its core was isolated (nothing else needed to run there). On a shared
  core that is illegal — a long FPU section disables preemption and stalls
  the scheduler. So here every FPU-using operation opens and closes its own
  short `kernel_fpu_begin()`/`kernel_fpu_end()` section, and the generation
  loop calls `cond_resched()` between tokens. This is the yield point where
  the symbiosis actually happens: between operations the inference work
  cedes the cores back to the scheduler.

The tradeoff is honest and the reverse of Chapter 1's: Chapter 1 paid with
static resource partitioning to get uninterrupted FPU; this chapter pays
with more frequent (and slightly costlier) FPU section boundaries to get
dynamic, shared use of the whole machine. For this project's goals, the
second tradeoff is the right one.

## 3. Parallelizing the Bottleneck

The dominant cost in transformer inference is matrix multiplication. It is
parallelized here by **row partition**: for `xout[d] = W[d×n]·x[n]`, the `d`
output rows are split into contiguous ranges, one per thread. The main
thread computes range 0; a pool of `pool_size` kernel worker threads
(`worker_fn` in `appendix/andrea_llm_kmod.c`) compute the rest. Each thread
brackets its own slice in its own per-CPU `kernel_fpu_begin`/`kernel_fpu_end`.

### 3.1 Thread Pool and Synchronization

The workers are persistent (created once at module load, not per matmul, to
avoid creation overhead dominating on a small model). Synchronization uses a
generation counter and a completion:

- A new matmul sets up the per-thread row ranges, increments an
  `atomic_t g_gen`, and wakes the workers via a wait queue.
- Each worker wakes on the generation change, computes its slice under its
  own FPU section, then `atomic_inc`s a done-counter; the last worker to
  finish signals a `struct completion`.
- The main thread computes its own slice (under its own FPU section, which
  it must end before sleeping), then waits on the completion — without
  holding the FPU, since a thread may not sleep with a kernel FPU section
  open.

Because only one matmul is in flight at a time (the main thread waits for
completion before issuing the next), there is no window for a worker to miss
or double-process a job. The mechanism was exercised heavily under QEMU and
on real hardware with **zero deadlocks, races, or lockups** observed.

### 3.2 A Necessary Restructuring: FPU Per Operation

Chapter 2's engine wrapped the entire forward pass in a single
`kernel_fpu_begin`/`kernel_fpu_end`. That is incompatible with the main
thread sleeping on a worker completion mid-forward (sleeping with the FPU
held is illegal). The forward pass was therefore restructured so that every
floating-point operation manages its own short FPU section — RMSNorm,
softmax, the RoPE loop, the attention score/softmax/weighted-sum blocks, the
residual adds, the SwiGLU block, and argmax each bracket their own FPU use;
the matmul brackets its own (serial path) or delegates to the parallel path
(each thread bracketing its slice). This multiplies the number of FPU
section boundaries and adds overhead, accepted here because the symbiotic
model requires it and the small model tolerates it.

### 3.3 Bit-Identical to Serial

Row partition does not reorder any summation: output row `i` is computed by
exactly the same sequence of operations as in the serial code, merely on a
different thread. The result is therefore **bit-identical** to the
single-threaded computation, which was verified directly — the hash of the
generated text is identical across 1, 2, 4, and 8 threads. (This is a
property of row partition specifically; the SIMD work planned next will
*not* preserve bit-identity, because vectorized reduction reorders
floating-point additions, which are not associative.)

## 4. Results

All figures are real measurements, bit-identical output confirmed, no
lockups.

- **Userspace reference (development host, pool of pthreads, identical
  partition logic):** the 15M model scales 2.6× on 4 threads; the 110M model
  scales **3.27×** on 4 threads. The larger model scales *better* — its
  matmuls are large enough that the per-matmul synchronization overhead is
  amortized, whereas the 15M's small matmuls spend a larger fraction of time
  in coordination.
- **Synthetic large matmul (8192×8192):** 3.7× on 4 threads, 5.5× on 8,
  6.7× on 16. The sub-linear scaling at high thread counts is **memory
  bandwidth**, not the code: the threads contend for RAM reading the weight
  matrix, which does not fit in cache. This is a physical ceiling, relevant
  to any large-model deployment.
- **Real hardware (4-core x86_64, `isolcpus` removed so all four cores are
  schedulable):** the 15M model reaches **2.67×** and the 110M model
  **2.77×** on 4 cores, monotonically increasing with thread count. Before
  removing `isolcpus`, scaling *degraded* past 2 threads (only 3 cores were
  schedulable and the 4th thread oversubscribed them) — direct evidence that
  the static-isolation configuration actively conflicts with the symbiotic
  design.

### 4.1 The Honest Ceiling: Memory Bandwidth

On this particular hardware (a low-end i3), the 110M model's parallel
advantage over the 15M was smaller than on the development host (2.77× vs
the host's 3.27×). The cause is memory bandwidth: the 110M's 418 MB of
weights do not fit in cache, so every matmul streams them from RAM, and four
cores reading simultaneously saturate the memory bus before they saturate
compute. This is not a defect of the parallelization — it is a property of
the machine, and it is exactly the ceiling the synthetic benchmark predicted
(Section 4). It improves with quantization (smaller weights, less bandwidth
per matmul) or with hardware that has more memory bandwidth — precisely the
regime a larger deployment target would occupy.

## 5. Deployment

The parallel engine was consolidated into the system image and verified on
real hardware: the kernel module lives permanently in the encrypted root
filesystem, the GRUB command line was permanently stripped of `isolcpus`
(making the symbiotic configuration the default rather than a live tweak),
and — a deliberate design choice — **no model weights are bundled in the
image**. Only the engine is permanent; model files are loaded on demand and
persist in the encrypted, writable root filesystem until the next full
reflash. This keeps the base image small and makes swapping the model a
matter of copying a file, consistent with the principle that the capacity to
compute is fixed infrastructure while the model it runs is interchangeable
content.

## 6. Honest Limitations

- **Symbiosis is real but coarse.** The scheduler does redistribute cores
  under load, but the granularity is per-operation (`cond_resched` between
  tokens and short FPU sections), not a fine-grained priority negotiation.
  Under heavy sustained system load the inference simply runs slower; there
  is no explicit quality-of-service contract between the two.
- **Small-model overhead.** The per-matmul thread wake/wait cycle is pure
  overhead on the 15M model's tiny matmuls, which is why it scales only
  2.67× on 4 cores rather than closer to 4×. The mechanism is aimed at large
  models; it is merely not harmful on small ones (past the removal of
  `isolcpus`).
- **Memory bandwidth, not compute, is the binding constraint** on this
  hardware for large models (Section 4.1). More cores will not help past the
  bandwidth ceiling; SIMD and quantization address the compute and bandwidth
  sides respectively and are the natural next steps.
- **Still inference only.** As in Chapter 2, the model does not learn or
  grow; this chapter accelerates inference, nothing more.
- **aarch64 compile-verified only,** unchanged from Chapters 1–2: the module
  compiles for arm64 with the `kernel_neon` path but the arm64 runtime boot
  environment is not yet complete enough to execute it.

## 7. Next Steps

1. Hand-written SIMD (AVX-512 / NEON) in the matmul inner loop, inside the
   existing per-thread FPU sections — the largest remaining single speedup
   (8–16× per core), at the cost of bit-identity (vectorized reduction is
   not bit-exact against scalar).
2. Quantization (Q4) to cut memory bandwidth, directly targeting the ceiling
   found in Section 4.1.
3. Larger models, once SIMD and quantization make them practical on
   bandwidth-limited hardware.
