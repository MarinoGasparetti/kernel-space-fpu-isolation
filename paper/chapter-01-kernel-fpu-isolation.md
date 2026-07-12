# Chapter 1 — Kernel-Space Floating Point via CPU Isolation

## Abstract

This chapter documents the first step toward embedding an LLM inference
engine (`llama.cpp`/`ggml`) as kernel-resident computation rather than a
userspace service: identifying the specific architectural obstacle that
makes sustained floating-point computation awkward inside the Linux
kernel, and verifying a mechanism — dedicated CPU isolation combined with
a pinned kernel thread — that resolves it. The mechanism is verified, not
merely proposed: it was run under QEMU (x86_64) and, the same day, on
physical deployment target hardware, executing 20,923 sustained
floating-point compute blocks (≈2.1×10¹⁰ operations) inside the kernel
with zero scheduler stalls. The chapter also documents three unrelated
engineering defects found and fixed during setup — a Buildroot Kconfig
dependency gap, a missing ARM64 PCI host-controller driver, and a stale
build-cache defect in the kernel's own Makefile-driven initramfs
regeneration — because each is a real, reusable finding independent of the
central result.

## 1. Motivation

An inference engine that lives inside the kernel, rather than as a
privileged userspace daemon, is architecturally consistent with a premise
explored elsewhere (the `AndreaV1.1.0` project this work is deployed
against): a system in which "the kernel is the software" has no removable
boundary between mechanism and intelligence. Realizing that premise for an
LLM inference engine specifically first requires answering a narrower,
concrete question: can the kind of sustained floating-point computation an
inference forward pass performs run correctly inside the Linux kernel at
all, and if so, under what constraint.

## 2. The Obstacle: Sustained FPU Use Inside a Kernel

Kernel code does not have unrestricted access to the CPU's floating-point
and vector register file. On x86_64, kernel code compiled with the
default kernel build flags cannot even emit floating-point-returning
instructions (`-mno-sse -mno-sse2`-class restrictions; the arm64 kernel
build carries an analogous restriction under `-mgeneral-regs-only`,
Section 4.2). Kernel code that legitimately needs the FPU — cryptographic
acceleration (AES-NI) and RAID6 XOR being the canonical examples already
present in the mainline kernel — must explicitly bracket that use with
`kernel_fpu_begin()`/`kernel_fpu_end()` on x86_64, or
`kernel_neon_begin()`/`kernel_neon_end()` on arm64.

The obstacle is not that this bracketing is hard to use — it is that it
was designed for **short** bursts. `kernel_fpu_begin()` disables kernel
preemption on the calling core for its entire duration, so that the
in-progress FPU/vector state belonging to whichever task the kernel
interrupted cannot be clobbered by the kernel's own temporary use of the
same physical register file. AES-NI and RAID6 hold this section open for a
single cache-line-sized operation. An LLM forward pass, even a small one,
performs sustained floating-point computation for whole seconds at a time.
Holding `kernel_fpu_begin()` open for that long would starve every other
task scheduled on that core — not a performance regression, a full
scheduler stall on that core for the duration of the computation.

## 3. The Verified Mechanism: CPU Isolation Plus a Pinned Kernel Thread

The resolution verified here does not attempt to shorten or restructure
the computation to fit inside a brief FPU section (an alternative,
rejected during design, would chunk the computation into small tiles with
`cond_resched()` between `kernel_fpu_begin`/`end` pairs — workable, but at
a real, repeated FPU-context-switch cost per tile). Instead, it removes the
constraint that motivates the "short section" rule in the first place:
**dedicate one physical CPU core entirely to kernel-side computation**, via
Linux's standard CPU-isolation boot parameters —

```
isolcpus=3 nohz_full=3 rcu_nocbs=3
```

`isolcpus` removes the core from the general-purpose scheduler's target
set (confirmed empirically: `nproc` on the deployment hardware reported 3
usable cores after this parameter was applied to a 4-core machine, not 4).
`nohz_full` stops the periodic scheduler timer tick on that core.
`rcu_nocbs` moves that core's RCU callback processing elsewhere. This exact
three-parameter combination is not novel in itself — it is the standard
mechanism used in real-time and high-throughput packet-processing
deployments (DPDK-class workloads) to obtain a core with no competing
kernel housekeeping. What this chapter verifies is that the *same*
mechanism, applied for a different purpose (sustained in-kernel floating
point rather than sustained in-kernel packet processing), removes the
FPU-section-length constraint as a side effect: since no userspace task and
no general kernel work is ever scheduled onto that core, a kernel thread
pinned there may hold `kernel_fpu_begin()`/`kernel_neon_begin()` open for
as long as it needs, because there is nothing else on that core whose FPU
state it could be clobbering or whose scheduling it could be starving.

### 3.1 Implementation

The verification module (`appendix/andrea_mind_core.c`, reproduced in full)
creates a single kernel thread at module load time, binds it to a
caller-specified CPU (`kthread_bind`), and runs a sustained
floating-point compute loop inside architecture-appropriate FPU brackets:

```c
#if defined(__x86_64__)
#include <asm/fpu/api.h>
#define ANDREA_FPU_BEGIN() kernel_fpu_begin()
#define ANDREA_FPU_END() kernel_fpu_end()
#elif defined(__aarch64__)
#include <asm/neon.h>
#define ANDREA_FPU_BEGIN() kernel_neon_begin()
#define ANDREA_FPU_END() kernel_neon_end()
#endif
```

```c
while (!kthread_should_stop()) {
    ANDREA_FPU_BEGIN();
    for (int i = 0; i < 1000000; i++) {
        acc += 1.0001f;
        acc *= 0.9999f;
    }
    ANDREA_FPU_END();

    iterations++;
    cond_resched();
}
```

`cond_resched()` between blocks is a courtesy yield, not a correctness
requirement of the isolation mechanism itself — since the isolated core
has no other runnable task in the general scheduler's view, the module
does not depend on it to avoid starving anything; it is included so the
thread itself remains cleanly stoppable (`kthread_should_stop()`) between
blocks rather than only at a coarser granularity.

### 3.2 A Genuine Compiler-Level Obstacle, Resolved Per Architecture

Compiling this module against each architecture's kernel build flags
surfaced the restriction described in Section 2 directly, as a build
failure rather than an abstract concern:

- On x86_64: `error: SSE register return with SSE disabled`, resolved by
  re-enabling SSE for this translation unit specifically —
  `CFLAGS_andrea_mind_core.o += -msse -msse2` — a standard, narrowly-scoped
  override, the same pattern used by in-kernel crypto/RAID glue code that
  needs FPU access in specific files without changing the kernel-wide
  default.
- On arm64: `error: '-mgeneral-regs-only' is incompatible with the use of
  floating-point types`, resolved with the Kbuild-specific "remove a global
  flag for one file" mechanism — `CFLAGS_REMOVE_andrea_mind_core.o +=
  -mgeneral-regs-only` — rather than an additive flag, since the arm64
  kernel build's restriction is a flag to be *removed* for this file, not a
  missing capability to be *added*.

Both fixes are correct and minimal specifically because the FPU-safety
contract (bracketing every use in `kernel_fpu_begin`/`kernel_neon_begin`)
is upheld by the module itself; the compiler-level restriction exists to
prevent accidental, unbracketed FPU use elsewhere in the kernel, not to
prevent this pattern generally.

## 4. Verification

### 4.1 QEMU (x86_64)

The module was loaded under an emulated x86_64 kernel and rootfs (QEMU,
`-smp 2`, `isolcpus=1 nohz_full=1 rcu_nocbs=1`), targeting the isolated
core. Raw console output: `docs/qemu-x86_64-boot-and-test.log`.

- The module loaded, the thread started on the target CPU, and reported
  periodic progress (`N blocchi FP completati`) with no gap or delay
  consistent with scheduler contention.
- A full-log search for `soft lockup`, `hard lockup`, `rcu_sched.*stall`,
  and `BUG:` — the standard kernel self-instrumentation for exactly this
  failure class — returned zero matches across the entire session.
- The system remained interactively responsive throughout (further shell
  commands queued during the run executed normally once their turn came).

### 4.2 Physical Hardware

The same module (recompiled against the exact production kernel build tree
already deployed — no kernel rebuild or reflash was required, since kernel
module loading is independent of the kernel binary itself) was deployed to
physical target hardware (a 4-core x86_64 server, kernel 6.1.24) via the
project's standard live-SSH-deployment pattern, with `isolcpus=3
nohz_full=3 rcu_nocbs=3` added to the GRUB kernel command line beforehand.
Raw dmesg capture: `docs/asus-real-hardware-dmesg.log`.

**Result**: `nproc` reported 3 available cores post-boot (down from 4),
confirming `isolcpus` took effect. The module ran for approximately one
minute of wall-clock time before being unloaded, completing **20,923**
compute blocks of 1,000,000 floating-point operations each —
approximately 2.1×10¹⁰ total floating-point operations executed inside the
kernel, on the isolated core — with:

- Zero soft-lockup, hard-lockup, or RCU-stall warnings in the kernel log.
- The rest of the system fully responsive throughout (SSH sessions,
  `dmesg`, and shell commands on the other three cores executed normally
  and without perceptible delay).
- A clean module unload (`rmmod`) at the end of the run, with no residual
  state and no warning on removal.

Real hardware executed at a substantially higher throughput than the
emulated run (20,923 blocks in roughly one minute of wall-clock time versus
a comparable emulated run over tens of minutes), consistent with QEMU's
software CPU emulation overhead rather than any difference in the
mechanism's correctness.

## 5. Three Unrelated Defects Found During Setup

None of the following affect the central result in Section 4; each is
recorded because it is a genuine, reusable finding encountered while
building the verification environment, not a property of the FPU-isolation
mechanism itself.

- **A Buildroot Kconfig dependency gap.** Selecting "use the same kernel
  headers as the kernel being built" (`BR2_KERNEL_HEADERS_AS_KERNEL`) does
  not, on inspection of the Buildroot source tree, `select` the
  corresponding `BR2_TOOLCHAIN_HEADERS_AT_LEAST_*` symbol the way the
  standard versioned-headers choices do — leaving the toolchain's
  self-reported minimum headers version at its oldest default ("2.6") and
  causing a hard build failure (`Incorrect selection of kernel headers:
  expected 2.6.x, got 6.1.x`) at the toolchain-validation step. Resolved by
  selecting the standard versioned choice (`BR2_KERNEL_HEADERS_6_1`)
  instead, which does carry the correct `select`.
- **A missing ARM64 PCI host-controller driver.** An arm64 kernel built
  with `CONFIG_PCI=y` and `CONFIG_VIRTIO_PCI=y` still enumerated zero PCI
  devices under QEMU's `virt` machine, because `CONFIG_PCI_HOST_GENERIC`
  (the driver for the generic ECAM-based PCIe host bridge that machine
  model exposes) was not enabled — a dependency x86 platforms do not have,
  since they support legacy PCI configuration-space access without a
  dedicated host-controller driver. Without it, `CONFIG_PCI=y` alone
  initializes the PCI subsystem but never finds an actual bus to probe.
- **A stale kernel-build-cache defect**, in the same family as a
  previously-documented Buildroot rootfs-staleness issue in the parent
  project: after adding real content to a previously-empty
  `CONFIG_INITRAMFS_SOURCE` directory and forcing a Buildroot-level package
  rebuild, the kernel's own embedded initramfs remained a stale, effectively
  empty 512-byte archive — confirmed by manually invoking the kernel's own
  `gen_initramfs.sh` outside the build system, which correctly picked up
  the new directory content, versus the actual kernel build, which did not.
  The kernel's internal build system (`kbuild`) tracks initramfs
  regeneration via its own dependency file
  (`usr/.initramfs_data.cpio.d`), separate from Buildroot's package-level
  stamp files; removing Buildroot's stamps was not sufficient to invalidate
  this internal, lower-level cache. Resolved by removing the kernel-internal
  initramfs build artifacts directly (`usr/initramfs_data.cpio*`) before
  rebuilding.

## 6. Honest Limitations

- **The compute performed is a placeholder, not inference.** The verified
  loop (`acc += 1.0001f; acc *= 0.9999f`) is a synthetic sustained-FPU
  workload chosen to stress the mechanism, not a real `ggml` compute
  kernel. Whether real `ggml` code (SIMD-dispatched dot products, matrix
  multiplication over quantized weights) behaves identically under the same
  isolation-plus-pinned-thread scheme — same absence of stalls, same
  clean unload — is the immediate next step, not yet verified.
- **ARM64 is verified at compile time only.** The module compiles cleanly
  against the arm64 kernel build tree with the architecture-appropriate
  `kernel_neon_begin`/`kernel_neon_end` substitution and the corresponding
  `-mgeneral-regs-only` removal (Section 3.2), but has not yet been run to
  completion under emulation or on real arm64 hardware: the arm64 test
  environment's boot process (a separate, minimal custom initramfs, not
  part of this module) is not yet complete enough to reach a shell and load
  the module — an unrelated, already-identified pending task, not a defect
  in the mechanism this chapter verifies.
- **Single-core, single-thread scope.** This chapter verifies that one
  isolated core can sustain kernel-space FPU use indefinitely. It does not
  address multi-core kernel-space inference (would require either multiple
  isolated cores and thread coordination, or a single-threaded inference
  loop, both unexplored here), nor does it address the separate obstacles
  already identified for a full inference engine port — kernel-appropriate
  memory allocation in place of `malloc`/`calloc`, `kthread`-based
  parallelism in place of `pthread`, and chunked `kernel_read`-based model
  loading in place of `mmap` — none of which this chapter's verification
  exercises.

## 7. Next Steps

1. Replace the placeholder compute loop with a real, minimal `ggml` compute
   primitive (a single vectorized dot-product kernel, e.g. the function
   family `ggml_vec_dot_f32`) under the same isolation-plus-pinned-thread
   scheme, verifying numeric correctness against a known expected result —
   the first test of real inference-relevant code under this mechanism,
   rather than a synthetic stress workload.
2. Complete arm64 runtime verification once the arm64 test boot
   environment reaches a working shell (a pending, unrelated task).
3. Address the remaining architectural obstacles to a full engine port
   identified but not yet resolved: kernel-appropriate memory allocation,
   `kthread`-based parallelism, and chunked model-weight loading without
   userspace-style `mmap`.
