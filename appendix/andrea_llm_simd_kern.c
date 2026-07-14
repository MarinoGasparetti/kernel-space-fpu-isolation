#include <linux/kernel.h>

#if defined(__x86_64__)

typedef float v8sf __attribute__((vector_size(32), aligned(4)));

float andrea_dot(const float *a, const float *b, int n)
{
    v8sf acc0 = {0}, acc1 = {0};
    int i = 0, j;
    float sum = 0.0f;

    for (; i + 16 <= n; i += 16) {
        acc0 += (*(const v8sf *)(a + i))     * (*(const v8sf *)(b + i));
        acc1 += (*(const v8sf *)(a + i + 8)) * (*(const v8sf *)(b + i + 8));
    }
    for (; i + 8 <= n; i += 8)
        acc0 += (*(const v8sf *)(a + i)) * (*(const v8sf *)(b + i));

    acc0 += acc1;
    for (j = 0; j < 8; j++)
        sum += acc0[j];
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

#elif defined(__aarch64__)

typedef float v4sf __attribute__((vector_size(16), aligned(4)));

float andrea_dot(const float *a, const float *b, int n)
{
    v4sf acc0 = {0}, acc1 = {0};
    int i = 0, j;
    float sum = 0.0f;

    for (; i + 8 <= n; i += 8) {
        acc0 += (*(const v4sf *)(a + i))     * (*(const v4sf *)(b + i));
        acc1 += (*(const v4sf *)(a + i + 4)) * (*(const v4sf *)(b + i + 4));
    }
    for (; i + 4 <= n; i += 4)
        acc0 += (*(const v4sf *)(a + i)) * (*(const v4sf *)(b + i));

    acc0 += acc1;
    for (j = 0; j < 4; j++)
        sum += acc0[j];
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

#else

float andrea_dot(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    int i;
    for (i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

#endif
