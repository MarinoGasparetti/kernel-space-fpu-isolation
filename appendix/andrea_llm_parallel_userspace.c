#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/time.h>

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    float *token_embedding_table, *rms_att_weight, *rms_ffn_weight;
    float *wq, *wk, *wv, *wo, *w1, *w2, *w3, *rms_final_weight, *wcls;
} Weights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache;
} RunState;

/* ---- thread pool per matmul parallela ---- */

#define MAX_THREADS 64

typedef struct {
    float *xout, *x, *w;
    int n;
    int row_start, row_end;
} MatmulJob;

static int g_nthreads = 1;
static pthread_t g_workers[MAX_THREADS];
static MatmulJob g_jobs[MAX_THREADS];
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_work_done = PTHREAD_COND_INITIALIZER;
static int g_work_gen = 0;
static int g_done_count = 0;
static int g_shutdown = 0;

static void do_rows(MatmulJob *j)
{
    for (int i = j->row_start; i < j->row_end; i++) {
        float val = 0.0f;
        float *wr = j->w + (long)i * j->n;
        for (int k = 0; k < j->n; k++) val += wr[k] * j->x[k];
        j->xout[i] = val;
    }
}

static void *worker_fn(void *arg)
{
    int id = (int)(long)arg;
    int last_gen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mtx);
        while (g_work_gen == last_gen && !g_shutdown)
            pthread_cond_wait(&g_work_ready, &g_mtx);
        if (g_shutdown) { pthread_mutex_unlock(&g_mtx); return NULL; }
        last_gen = g_work_gen;
        pthread_mutex_unlock(&g_mtx);

        do_rows(&g_jobs[id]);

        pthread_mutex_lock(&g_mtx);
        g_done_count++;
        if (g_done_count == g_nthreads - 1)
            pthread_cond_signal(&g_work_done);
        pthread_mutex_unlock(&g_mtx);
    }
}

static void pool_init(int nthreads)
{
    g_nthreads = nthreads;
    for (int i = 1; i < nthreads; i++)
        pthread_create(&g_workers[i], NULL, worker_fn, (void *)(long)i);
}

static void pool_shutdown(void)
{
    pthread_mutex_lock(&g_mtx);
    g_shutdown = 1;
    pthread_cond_broadcast(&g_work_ready);
    pthread_mutex_unlock(&g_mtx);
    for (int i = 1; i < g_nthreads; i++)
        pthread_join(g_workers[i], NULL);
}

static void matmul(float *xout, float *x, float *w, int n, int d)
{
    if (g_nthreads == 1 || d < g_nthreads * 4) {
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            float *wr = w + (long)i * n;
            for (int k = 0; k < n; k++) val += wr[k] * x[k];
            xout[i] = val;
        }
        return;
    }

    int rows_per = d / g_nthreads;
    for (int t = 0; t < g_nthreads; t++) {
        g_jobs[t].xout = xout; g_jobs[t].x = x; g_jobs[t].w = w; g_jobs[t].n = n;
        g_jobs[t].row_start = t * rows_per;
        g_jobs[t].row_end = (t == g_nthreads - 1) ? d : (t + 1) * rows_per;
    }

    pthread_mutex_lock(&g_mtx);
    g_done_count = 0;
    g_work_gen++;
    pthread_cond_broadcast(&g_work_ready);
    pthread_mutex_unlock(&g_mtx);

    do_rows(&g_jobs[0]);

    pthread_mutex_lock(&g_mtx);
    while (g_done_count < g_nthreads - 1)
        pthread_cond_wait(&g_work_done, &g_mtx);
    pthread_mutex_unlock(&g_mtx);
}

/* ---- resto del transformer (identico all'originale) ---- */

static void malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

static void map_weights(Weights *w, Config *p, float *ptr, int shared) {
    int head_size = p->dim / p->n_heads;
    unsigned long long nl = p->n_layers;
    w->token_embedding_table = ptr; ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr; ptr += nl * p->dim;
    w->wq = ptr; ptr += nl * p->dim * (p->n_heads * head_size);
    w->wk = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr; ptr += nl * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr; ptr += nl * p->dim;
    w->w1 = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->w2 = ptr; ptr += nl * p->hidden_dim * p->dim;
    w->w3 = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr; ptr += p->dim;
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared ? w->token_embedding_table : ptr;
}

static void rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float *x, int size) {
    float mx = x[0];
    for (int i = 1; i < size; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static float *forward(Config *p, Weights *w, RunState *s, int token, int pos) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    memcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int hd = i % head_size;
            float freq = 1.0f / powf(10000.0f, hd / (float)head_size);
            float val = pos * freq, fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int vi = 0; vi < rotn; vi++) {
                float *vec = vi == 0 ? s->q : s->k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

static char **vocab;
static void build_tokenizer(char *path, int vocab_size) {
    vocab = malloc(vocab_size * sizeof(char *));
    FILE *f = fopen(path, "rb");
    int mtl, len; float sc;
    if (fread(&mtl, sizeof(int), 1, f) != 1) exit(1);
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&sc, sizeof(float), 1, f) != 1) exit(1);
        if (fread(&len, sizeof(int), 1, f) != 1) exit(1);
        vocab[i] = malloc(len + 1);
        if (fread(vocab[i], len, 1, f) != 1) exit(1);
        vocab[i][len] = '\0';
    }
    fclose(f);
}

static int argmax(float *v, int n) {
    int mi = 0; float mp = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mp) { mp = v[i]; mi = i; }
    return mi;
}

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static unsigned long hash_str(const char *s) {
    unsigned long h = 5381; int c;
    while ((c = *s++)) h = ((h << 5) + h) + c;
    return h;
}

int main(int argc, char **argv) {
    char *checkpoint = argc > 1 ? argv[1] : "stories15M.bin";
    char *tokpath = argc > 2 ? argv[2] : "tokenizer.bin";
    int steps = argc > 3 ? atoi(argv[3]) : 256;
    int nthreads = argc > 4 ? atoi(argv[4]) : 1;

    Config config; Weights weights;
    int fd = open(checkpoint, O_RDONLY);
    struct stat st; fstat(fd, &st);
    long fsz = st.st_size;
    float *data = malloc(fsz);
    if (read(fd, data, fsz) != fsz) return 1;
    close(fd);

    memcpy(&config, data, sizeof(Config));
    int shared = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    map_weights(&weights, &config, data + sizeof(Config) / sizeof(float), shared);
    build_tokenizer(tokpath, config.vocab_size);

    if (nthreads > 1) pool_init(nthreads);

    RunState state; malloc_run_state(&state, &config);
    if (steps > config.seq_len) steps = config.seq_len;

    char out[16384]; int olen = 0;
    int token = 1, pos = 0;
    double t0 = now_ms();
    while (pos < steps) {
        float *logits = forward(&config, &weights, &state, token, pos);
        int next = argmax(logits, config.vocab_size);
        char *piece = vocab[next];
        if (token == 1 && piece[0] == ' ') piece++;
        int pl = strlen(piece);
        if (olen + pl < 16000) { memcpy(out + olen, piece, pl); olen += pl; }
        token = next; pos++;
        if (next == 1) break;
    }
    double t1 = now_ms();
    out[olen] = '\0';

    if (nthreads > 1) pool_shutdown();

    double dt = (t1 - t0) / 1000.0;
    fprintf(stderr, "threads=%d  token=%d  tempo=%.3fs  tok/s=%.1f  hash_output=%lu\n",
            nthreads, pos, dt, pos / dt, hash_str(out));
    if (argc > 5 && strcmp(argv[5], "-v") == 0)
        printf("%s\n", out);
    return 0;
}
