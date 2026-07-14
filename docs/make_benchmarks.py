import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

C_SCALAR = "#888888"
C_PAR = "#1f77b4"
C_SIMD = "#d62728"
C_GREEN = "#2ca02c"

# ---------------------------------------------------------------------------
# Figure 1 — the optimization cascade (the headline result)
# Real Asus i3-1115G4 measurements, 10 generations of 256 tokens each.
# ---------------------------------------------------------------------------
labels = ["scalar\n1 thread", "scalar\n4 threads\n(Ch.3)",
          "SIMD\n1 thread\n(Ch.4)", "SIMD\n4 threads\n(Ch.4)"]
times = [32, 11, 9, 6]
speedups = [32 / t for t in times]
colors = [C_SCALAR, C_PAR, "#ff9896", C_SIMD]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, times, color=colors, edgecolor="black", linewidth=0.6)
for b, t, s in zip(bars, times, speedups):
    ax.text(b.get_x() + b.get_width() / 2, t + 0.5,
            f"{t}s\n{s:.1f}×", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("time for 10 generations (s) — lower is better")
ax.set_title("Optimization cascade on real hardware (i3-1115G4, 4 cores)\n"
             "kernel-resident LLM inference, TinyStories 15M, greedy decoding")
ax.set_ylim(0, 37)
fig.tight_layout()
fig.savefig("benchmark-01-optimization-cascade.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 — why isolcpus was wrong: scaling with vs without it (15M, Asus)
# Real measurements: 10 generations each.
# ---------------------------------------------------------------------------
threads = [1, 2, 3, 4]
with_iso = [32, 18, 25, 29]      # isolcpus=3 -> only 3 cores schedulable
without_iso = [32, 18, 15, 12]   # isolcpus removed -> 4 cores

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threads, [32 / t for t in with_iso], "o--", color=C_SCALAR,
        label="with isolcpus=3 (static, Ch.1 approach)", linewidth=2, markersize=8)
ax.plot(threads, [32 / t for t in without_iso], "o-", color=C_GREEN,
        label="without isolcpus (symbiotic, Ch.3)", linewidth=2, markersize=8)
ax.plot(threads, threads, ":", color="#cccccc", label="ideal linear")
ax.set_xlabel("threads (cores given to the mind)")
ax.set_ylabel("speedup vs 1 thread")
ax.set_title("Why static isolation was the wrong foundation (15M, Asus)\n"
             "isolcpus caps at 2 threads then degrades; symbiotic scales to 4")
ax.set_xticks(threads)
ax.legend()
fig.tight_layout()
fig.savefig("benchmark-02-isolcpus-vs-symbiotic.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 — large-matmul scaling and the memory-bandwidth ceiling
# Real synthetic benchmark: W[8192x8192] . x, GFLOP/s by thread count.
# ---------------------------------------------------------------------------
t3 = [1, 2, 4, 8, 16]
gflops = [2.9, 5.6, 10.7, 16.0, 19.5]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t3, gflops, "o-", color=C_PAR, linewidth=2, markersize=8, label="measured")
ideal = [2.9 * n for n in t3]
ax.plot(t3, ideal, ":", color="#cccccc", label="ideal linear")
ax.set_xlabel("threads")
ax.set_ylabel("GFLOP/s")
ax.set_xscale("log", base=2)
ax.set_xticks(t3)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title("Large matmul (8192×8192) scaling — the memory-bandwidth ceiling\n"
             "near-linear to ~4 threads, then RAM bandwidth binds (not the code)")
ax.annotate("threads contend for RAM\n(weights don't fit in cache)",
            xy=(16, 19.5), xytext=(4.2, 24),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig("benchmark-03-bandwidth-ceiling.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 — small vs large model scaling (userspace reference, tok/s)
# Real: 15M and 110M, threads 1/2/4/8.
# ---------------------------------------------------------------------------
t4 = [1, 2, 4, 8]
m15 = [152, 252, 398, 382]
m110 = [15.1, 28.5, 49.4, 59.3]
s15 = [v / m15[0] for v in m15]
s110 = [v / m110[0] for v in m110]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t4, s15, "o-", color=C_PAR, linewidth=2, markersize=8, label="15M model")
ax.plot(t4, s110, "s-", color=C_SIMD, linewidth=2, markersize=8, label="110M model")
ax.plot(t4, t4, ":", color="#cccccc", label="ideal linear")
ax.set_xlabel("threads")
ax.set_ylabel("speedup vs 1 thread")
ax.set_xscale("log", base=2)
ax.set_xticks(t4)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title("Larger models parallelize better (userspace reference)\n"
             "big matmuls amortize thread-sync overhead the small model pays")
ax.legend()
fig.tight_layout()
fig.savefig("benchmark-04-model-size-scaling.png")
plt.close(fig)

print("4 grafici generati")
