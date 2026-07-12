# Kernel-Space FPU Isolation: An Experimental Case Study in Embedding Inference Compute Inside a Linux Kernel

This is a private, working case study documenting a progressive attempt to
run AI inference computation directly inside a Linux kernel, as a
kernel-resident thread rather than a userspace service. The repository is
organized as a sequence of chapters, each corresponding to one
architectural or experimental milestone, written and expanded as the work
progresses rather than after the fact. Chapters are added, not rewritten —
a later chapter may revise an earlier chapter's conclusions, but the
earlier chapter's account of what was tried and found stays intact as a
historical record.

## Contents

- [`paper/chapter-01-kernel-fpu-isolation.md`](paper/chapter-01-kernel-fpu-isolation.md) — the first chapter: identifying floating-point-in-kernel-space as the central obstacle to embedding an inference engine (`llama.cpp`/`ggml`) in the kernel, and a verified mechanism (CPU isolation plus a pinned kernel thread) that resolves it without starving the scheduler
- `appendix/` — full source of the experimental kernel module and its dual-architecture build file
- `docs/` — raw logs from both verification runs: a QEMU x86_64 boot-and-test session, and a real-hardware dmesg capture from physical deployment target hardware

## One-paragraph summary

The starting technical obstacle is architectural, not a matter of effort:
sustained floating-point computation (the kind an LLM forward pass
requires) is structurally awkward inside the Linux kernel, because
`kernel_fpu_begin()`/`kernel_fpu_end()` — the mechanism by which kernel
code may legitimately use the FPU/vector register file — disables
preemption for its entire duration, and was designed for short bursts (as
used by AES-NI or RAID6 acceleration), not for a computation that runs for
seconds at a time. Holding it open that long on a normally-scheduled core
would stall every other task sharing that core. This chapter documents a
verified resolution: dedicate one physical CPU core entirely to kernel-side
computation via Linux's own CPU-isolation mechanism (`isolcpus`,
`nohz_full`, `rcu_nocbs` — the same mechanism used in real-time and
packet-processing deployments), and pin a kernel thread to that core. Since
no userspace task is ever scheduled there, the thread may hold the FPU
context open indefinitely without starving anything, because there is
nothing else on that core to starve. The mechanism was verified first under
QEMU (x86_64) and then, the same day, on physical deployment target
hardware — 20,923 blocks of 1,000,000 floating-point operations each
(approximately 2.1×10¹⁰ operations total) executed inside the kernel, on
the isolated core, with zero soft-lockup or stall warnings and the rest of
the system fully responsive throughout.
