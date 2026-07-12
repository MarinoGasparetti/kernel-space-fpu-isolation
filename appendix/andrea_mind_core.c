#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/delay.h>

#if defined(__x86_64__)
#include <asm/fpu/api.h>
#define ANDREA_FPU_BEGIN() kernel_fpu_begin()
#define ANDREA_FPU_END() kernel_fpu_end()
#elif defined(__aarch64__)
#include <asm/neon.h>
#define ANDREA_FPU_BEGIN() kernel_neon_begin()
#define ANDREA_FPU_END() kernel_neon_end()
#else
#error "andrea_mind_core: architettura non supportata (serve x86_64 o aarch64)"
#endif

static struct task_struct *andrea_mind_thread;
static int target_cpu = 3;
module_param(target_cpu, int, 0444);
MODULE_PARM_DESC(target_cpu, "CPU isolata dedicata al thread (deve combaciare con isolcpus=)");

static int andrea_mind_fn(void *data)
{
    volatile float acc = 0.0f;
    unsigned long iterations = 0;

    pr_info("andrea_mind_core: thread avviato su CPU %d\n", smp_processor_id());

    while (!kthread_should_stop()) {
        ANDREA_FPU_BEGIN();

        for (int i = 0; i < 1000000; i++) {
            acc += 1.0001f;
            acc *= 0.9999f;
        }

        ANDREA_FPU_END();

        iterations++;
        if (iterations % 100 == 0) {
            pr_info("andrea_mind_core: %lu blocchi FP completati, acc=%d (x1000)\n",
                    iterations, (int)(acc * 1000));
        }

        cond_resched();
    }

    pr_info("andrea_mind_core: thread fermato dopo %lu blocchi\n", iterations);
    return 0;
}

static int __init andrea_mind_init(void)
{
    pr_info("andrea_mind_core: caricamento, CPU target=%d\n", target_cpu);

    andrea_mind_thread = kthread_create(andrea_mind_fn, NULL, "andrea-mind-core");
    if (IS_ERR(andrea_mind_thread)) {
        pr_err("andrea_mind_core: creazione thread fallita\n");
        return PTR_ERR(andrea_mind_thread);
    }

    kthread_bind(andrea_mind_thread, target_cpu);
    wake_up_process(andrea_mind_thread);

    return 0;
}

static void __exit andrea_mind_exit(void)
{
    if (andrea_mind_thread) {
        kthread_stop(andrea_mind_thread);
    }
    pr_info("andrea_mind_core: scaricato\n");
}

module_init(andrea_mind_init);
module_exit(andrea_mind_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Prova: kthread fissato su CPU isolata, FPU sostenuta a blocchi");
