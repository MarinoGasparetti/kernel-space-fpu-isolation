#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/completion.h>
#include <linux/atomic.h>
#include <linux/cpumask.h>
#include <linux/sched.h>

#if defined(__x86_64__)
#include <asm/fpu/api.h>
#define ANDREA_FPU_BEGIN() kernel_fpu_begin()
#define ANDREA_FPU_END() kernel_fpu_end()
#elif defined(__aarch64__)
#include <asm/neon.h>
#define ANDREA_FPU_BEGIN() kernel_neon_begin()
#define ANDREA_FPU_END() kernel_neon_end()
#else
#error "architettura non supportata"
#endif

#include "andrea_llm_math.h"

static char *model_path = "/root/stories15M.bin";
module_param(model_path, charp, 0444);

static char *tokenizer_path = "/root/tokenizer.bin";
module_param(tokenizer_path, charp, 0444);

static int steps = 200;
module_param(steps, int, 0644);

static int threads;
module_param(threads, int, 0444);
MODULE_PARM_DESC(threads, "core dedicati al calcolo (0=auto=tutti i core online)");

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    float *token_embedding_table, *rms_att_weight, *rms_ffn_weight;
    float *wq, *wk, *wv, *wo, *w1, *w2, *w3, *rms_final_weight, *wcls;
} Weights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
    float *key_cache, *value_cache;
} RunState;

static void *model_data;
static long model_size;
static char **vocab;
static char *vocab_blob;
static Config config;
static Weights weights;
static RunState state;
static char *output_buf;
static int output_len;
static struct file *console_file;

/* ============ pool di worker per matmul parallela ============ */

#define MAX_WORKERS 63

static struct task_struct *g_workers[MAX_WORKERS];
static int g_pool_size;
static bool g_active;

static float *g_xout, *g_x, *g_w;
static int g_n;
static int g_part_start[MAX_WORKERS + 1];
static int g_part_end[MAX_WORKERS + 1];

static atomic_t g_gen = ATOMIC_INIT(0);
static atomic_t g_done = ATOMIC_INIT(0);
static DECLARE_WAIT_QUEUE_HEAD(g_work_wq);
static DECLARE_COMPLETION(g_all_done);

static void do_part(int p)
{
    int i, k;
    int r0 = g_part_start[p], r1 = g_part_end[p];
    for (i = r0; i < r1; i++) {
        float val = 0.0f;
        float *wr = g_w + (long)i * g_n;
        for (k = 0; k < g_n; k++) val += wr[k] * g_x[k];
        g_xout[i] = val;
    }
}

static int worker_fn(void *arg)
{
    long id = (long)arg;
    int last = 0;

    while (!kthread_should_stop()) {
        wait_event_interruptible(g_work_wq,
            atomic_read(&g_gen) != last || kthread_should_stop());
        if (kthread_should_stop())
            break;
        last = atomic_read(&g_gen);

        ANDREA_FPU_BEGIN();
        do_part(id + 1);
        ANDREA_FPU_END();

        if (atomic_inc_return(&g_done) == g_pool_size)
            complete(&g_all_done);
    }
    return 0;
}

static int pool_init(int nthreads)
{
    long i;
    int online = num_online_cpus();

    if (nthreads <= 0)
        nthreads = online;
    if (nthreads > online)
        nthreads = online;
    if (nthreads > MAX_WORKERS + 1)
        nthreads = MAX_WORKERS + 1;

    g_pool_size = nthreads - 1;
    g_active = (g_pool_size > 0);

    for (i = 0; i < g_pool_size; i++) {
        g_workers[i] = kthread_run(worker_fn, (void *)i, "andrea-llm-w%ld", i);
        if (IS_ERR(g_workers[i])) {
            g_pool_size = i;
            g_active = (g_pool_size > 0);
            return PTR_ERR(g_workers[i]);
        }
    }
    pr_info("andrea_llm: pool di %d parti (1 principale + %d worker) su %d core\n",
            nthreads, g_pool_size, online);
    return 0;
}

static void pool_shutdown(void)
{
    int i;
    for (i = 0; i < g_pool_size; i++)
        if (g_workers[i] && !IS_ERR(g_workers[i]))
            kthread_stop(g_workers[i]);
    g_pool_size = 0;
    g_active = false;
}

/* matmul: gestisce il proprio FPU. Parallela se attiva e la matrice è grande. */
static void matmul(float *xout, float *x, float *w, int n, int d)
{
    int parts, per, p;

    if (!g_active || d < (g_pool_size + 1) * 4) {
        int i, k;
        ANDREA_FPU_BEGIN();
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            float *wr = w + (long)i * n;
            for (k = 0; k < n; k++) val += wr[k] * x[k];
            xout[i] = val;
        }
        ANDREA_FPU_END();
        return;
    }

    parts = g_pool_size + 1;
    per = d / parts;
    for (p = 0; p < parts; p++) {
        g_part_start[p] = p * per;
        g_part_end[p] = (p == parts - 1) ? d : (p + 1) * per;
    }
    g_xout = xout; g_x = x; g_w = w; g_n = n;

    atomic_set(&g_done, 0);
    reinit_completion(&g_all_done);
    atomic_inc(&g_gen);
    wake_up_all(&g_work_wq);

    ANDREA_FPU_BEGIN();
    do_part(0);
    ANDREA_FPU_END();

    wait_for_completion(&g_all_done);
}

/* ============ operazioni leggere (ognuna gestisce il proprio FPU) ============ */

static long read_file(const char *path, void **out)
{
    struct file *f;
    loff_t pos = 0;
    long size;
    void *buf;
    ssize_t n;

    f = filp_open(path, O_RDONLY, 0);
    if (IS_ERR(f))
        return PTR_ERR(f);
    size = i_size_read(file_inode(f));
    buf = vmalloc(size);
    if (!buf) { filp_close(f, NULL); return -ENOMEM; }
    n = kernel_read(f, buf, size, &pos);
    filp_close(f, NULL);
    if (n != size) { vfree(buf); return -EIO; }
    *out = buf;
    return size;
}

static void map_weights(float *ptr, int shared)
{
    int head_size = config.dim / config.n_heads;
    unsigned long long nl = config.n_layers;
    weights.token_embedding_table = ptr; ptr += config.vocab_size * config.dim;
    weights.rms_att_weight = ptr; ptr += nl * config.dim;
    weights.wq = ptr; ptr += nl * config.dim * (config.n_heads * head_size);
    weights.wk = ptr; ptr += nl * config.dim * (config.n_kv_heads * head_size);
    weights.wv = ptr; ptr += nl * config.dim * (config.n_kv_heads * head_size);
    weights.wo = ptr; ptr += nl * (config.n_heads * head_size) * config.dim;
    weights.rms_ffn_weight = ptr; ptr += nl * config.dim;
    weights.w1 = ptr; ptr += nl * config.dim * config.hidden_dim;
    weights.w2 = ptr; ptr += nl * config.hidden_dim * config.dim;
    weights.w3 = ptr; ptr += nl * config.dim * config.hidden_dim;
    weights.rms_final_weight = ptr; ptr += config.dim;
    ptr += config.seq_len * head_size / 2;
    ptr += config.seq_len * head_size / 2;
    weights.wcls = shared ? weights.token_embedding_table : ptr;
}

static int alloc_state(void)
{
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    state.x = vzalloc(config.dim * sizeof(float));
    state.xb = vzalloc(config.dim * sizeof(float));
    state.xb2 = vzalloc(config.dim * sizeof(float));
    state.hb = vzalloc(config.hidden_dim * sizeof(float));
    state.hb2 = vzalloc(config.hidden_dim * sizeof(float));
    state.q = vzalloc(config.dim * sizeof(float));
    state.att = vzalloc(config.n_heads * config.seq_len * sizeof(float));
    state.logits = vzalloc(config.vocab_size * sizeof(float));
    state.key_cache = vzalloc((long)config.n_layers * config.seq_len * kv_dim * sizeof(float));
    state.value_cache = vzalloc((long)config.n_layers * config.seq_len * kv_dim * sizeof(float));
    if (!state.x || !state.xb || !state.xb2 || !state.hb || !state.hb2 ||
        !state.q || !state.att || !state.logits || !state.key_cache || !state.value_cache)
        return -ENOMEM;
    return 0;
}

static void free_state(void)
{
    vfree(state.x); vfree(state.xb); vfree(state.xb2);
    vfree(state.hb); vfree(state.hb2); vfree(state.q);
    vfree(state.att); vfree(state.logits);
    vfree(state.key_cache); vfree(state.value_cache);
}

static void rmsnorm(float *o, float *x, float *w, int size)
{
    float ss = 0.0f;
    int j;
    ANDREA_FPU_BEGIN();
    for (j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / k_sqrtf(ss);
    for (j = 0; j < size; j++) o[j] = w[j] * (ss * x[j]);
    ANDREA_FPU_END();
}

static void softmax(float *x, int size)
{
    float mx = x[0], sum = 0.0f;
    int i;
    ANDREA_FPU_BEGIN();
    for (i = 1; i < size; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < size; i++) { x[i] = k_expf(x[i] - mx); sum += x[i]; }
    for (i = 0; i < size; i++) x[i] /= sum;
    ANDREA_FPU_END();
}

static float *forward(int token, int pos)
{
    int dim = config.dim;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    int kv_mul = config.n_heads / config.n_kv_heads;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;
    float *content_row = weights.token_embedding_table + token * dim;
    int l, i, h, t;

    memcpy(state.x, content_row, dim * sizeof(float));

    for (l = 0; l < config.n_layers; l++) {
        float *k_row, *v_row;
        int loff = l * config.seq_len * kv_dim;

        rmsnorm(state.xb, state.x, weights.rms_att_weight + l * dim, dim);

        k_row = state.key_cache + loff + pos * kv_dim;
        v_row = state.value_cache + loff + pos * kv_dim;

        matmul(state.q, state.xb, weights.wq + l * dim * dim, dim, dim);
        matmul(k_row, state.xb, weights.wk + l * dim * kv_dim, dim, kv_dim);
        matmul(v_row, state.xb, weights.wv + l * dim * kv_dim, dim, kv_dim);

        ANDREA_FPU_BEGIN();
        for (i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / k_powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr, fci;
            int rotn = i < kv_dim ? 2 : 1;
            int vi;
            k_sincosf(val, &fci, &fcr);
            for (vi = 0; vi < rotn; vi++) {
                float *vec = vi == 0 ? state.q : k_row;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        ANDREA_FPU_END();

        for (h = 0; h < config.n_heads; h++) {
            float *q = state.q + h * head_size;
            float *att = state.att + h * config.seq_len;
            float *xb;
            ANDREA_FPU_BEGIN();
            for (t = 0; t <= pos; t++) {
                float *k = state.key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (i = 0; i < head_size; i++) score += q[i] * k[i];
                score /= k_sqrtf(head_size);
                att[t] = score;
            }
            ANDREA_FPU_END();
            softmax(att, pos + 1);
            xb = state.xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            ANDREA_FPU_BEGIN();
            for (t = 0; t <= pos; t++) {
                float *v = state.value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
            ANDREA_FPU_END();
        }

        matmul(state.xb2, state.xb, weights.wo + l * dim * dim, dim, dim);
        ANDREA_FPU_BEGIN();
        for (i = 0; i < dim; i++) state.x[i] += state.xb2[i];
        ANDREA_FPU_END();

        rmsnorm(state.xb, state.x, weights.rms_ffn_weight + l * dim, dim);
        matmul(state.hb, state.xb, weights.w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(state.hb2, state.xb, weights.w3 + l * dim * hidden_dim, dim, hidden_dim);

        ANDREA_FPU_BEGIN();
        for (i = 0; i < hidden_dim; i++) {
            float val = state.hb[i];
            val *= (1.0f / (1.0f + k_expf(-val)));
            val *= state.hb2[i];
            state.hb[i] = val;
        }
        ANDREA_FPU_END();

        matmul(state.xb, state.hb, weights.w2 + l * dim * hidden_dim, hidden_dim, dim);
        ANDREA_FPU_BEGIN();
        for (i = 0; i < dim; i++) state.x[i] += state.xb[i];
        ANDREA_FPU_END();
    }

    rmsnorm(state.x, state.x, weights.rms_final_weight, dim);
    matmul(state.logits, state.x, weights.wcls, dim, config.vocab_size);
    return state.logits;
}

static int argmax(float *v, int n)
{
    int mi = 0, i;
    float mp;
    ANDREA_FPU_BEGIN();
    mp = v[0];
    for (i = 1; i < n; i++) if (v[i] > mp) { mp = v[i]; mi = i; }
    ANDREA_FPU_END();
    return mi;
}

static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void console_write(const char *s, int len)
{
    loff_t pos = 0;
    if (console_file && len > 0)
        kernel_write(console_file, s, len, &pos);
}

static void emit_piece(const char *piece)
{
    char byte;
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        hexval(piece[3]) >= 0 && hexval(piece[4]) >= 0 && piece[5] == '>') {
        byte = (char)(hexval(piece[3]) * 16 + hexval(piece[4]));
        console_write(&byte, 1);
        if (output_len < 8000) output_buf[output_len++] = byte;
        return;
    }
    console_write(piece, strlen(piece));
    while (*piece && output_len < 8000)
        output_buf[output_len++] = *piece++;
}

static int build_tokenizer(void)
{
    void *tokdata;
    long size = read_file(tokenizer_path, &tokdata);
    char *p;
    int i;

    if (size < 0)
        return size;
    vocab = kmalloc_array(config.vocab_size, sizeof(char *), GFP_KERNEL);
    if (!vocab) { vfree(tokdata); return -ENOMEM; }
    vocab_blob = vmalloc(size);
    if (!vocab_blob) { kfree(vocab); vfree(tokdata); return -ENOMEM; }
    memcpy(vocab_blob, tokdata, size);
    vfree(tokdata);
    p = vocab_blob + sizeof(int);
    for (i = 0; i < config.vocab_size; i++) {
        int len;
        p += sizeof(float);
        len = *(int *)p;
        p += sizeof(int);
        vocab[i] = p;
        p += len;
        *p = '\0';
    }
    return 0;
}

static void generate(void)
{
    int token = 1, pos = 0;
    output_len = 0;

    console_write("\n--- Andrea genera ---\n", 22);

    while (pos < steps && output_len < 8000) {
        float *logits;
        int next;
        char *piece;

        logits = forward(token, pos);
        next = argmax(logits, config.vocab_size);

        piece = vocab[next];
        if (token == 1 && piece[0] == ' ') piece++;
        emit_piece(piece);

        token = next;
        pos++;
        if (next == 1) break;
        cond_resched();
    }
    output_buf[output_len] = '\0';
    console_write("\n--- fine ---\n", 14);
    pr_info("andrea_llm: generati %d token\n", pos);
}

static int andrea_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s\n", output_buf);
    return 0;
}

static int andrea_open(struct inode *inode, struct file *file)
{
    return single_open(file, andrea_show, NULL);
}

static const struct proc_ops andrea_proc_ops = {
    .proc_open = andrea_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int __init andrea_llm_init(void)
{
    float *wptr;
    int shared, ret;

    pr_info("andrea_llm: caricamento modello da %s\n", model_path);

    model_size = read_file(model_path, &model_data);
    if (model_size < 0) {
        pr_err("andrea_llm: lettura modello fallita: %ld\n", model_size);
        return model_size;
    }

    memcpy(&config, model_data, sizeof(Config));
    shared = config.vocab_size > 0 ? 1 : 0;
    if (config.vocab_size < 0) config.vocab_size = -config.vocab_size;

    pr_info("andrea_llm: dim=%d layers=%d heads=%d vocab=%d seq_len=%d\n",
            config.dim, config.n_layers, config.n_heads, config.vocab_size, config.seq_len);

    wptr = (float *)model_data + sizeof(Config) / sizeof(float);
    map_weights(wptr, shared);

    ret = alloc_state();
    if (ret) { pr_err("andrea_llm: alloc stato fallita\n"); goto err_model; }

    ret = build_tokenizer();
    if (ret) { pr_err("andrea_llm: tokenizer fallito: %d\n", ret); goto err_state; }

    output_buf = vmalloc(8192);
    if (!output_buf) { ret = -ENOMEM; goto err_tok; }

    if (steps > config.seq_len) steps = config.seq_len;

    pool_init(threads);

    console_file = filp_open("/dev/console", O_WRONLY, 0);
    if (IS_ERR(console_file)) console_file = NULL;

    generate();

    if (console_file) { filp_close(console_file, NULL); console_file = NULL; }

    pool_shutdown();

    proc_create("andrea_llm", 0444, NULL, &andrea_proc_ops);
    pr_info("andrea_llm: output pronto in /proc/andrea_llm\n");
    return 0;

err_tok:
    vfree(vocab_blob);
    kfree(vocab);
err_state:
    free_state();
err_model:
    vfree(model_data);
    return ret;
}

static void __exit andrea_llm_exit(void)
{
    remove_proc_entry("andrea_llm", NULL);
    vfree(output_buf);
    vfree(vocab_blob);
    kfree(vocab);
    free_state();
    vfree(model_data);
    pr_info("andrea_llm: scaricato\n");
}

module_init(andrea_llm_init);
module_exit(andrea_llm_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Motore LLM (Llama2) nel kernel, matmul parallela multi-core simbiotica");
