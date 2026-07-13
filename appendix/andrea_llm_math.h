#ifndef ANDREA_LLM_MATH_H
#define ANDREA_LLM_MATH_H

static inline float k_fabsf(float x) { return x < 0 ? -x : x; }

static inline float k_sqrtf(float x)
{
    if (x <= 0.0f) return 0.0f;
    float guess = x;
    for (int i = 0; i < 20; i++)
        guess = 0.5f * (guess + x / guess);
    return guess;
}

static inline float k_expf(float x)
{
    if (x < -60.0f) return 0.0f;
    if (x > 60.0f) x = 60.0f;

    int k = (int)(x * 1.4426950408889634f + (x < 0 ? -0.5f : 0.5f));
    float r = x - (float)k * 0.6931471805599453f;

    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i < 14; i++) {
        term *= r / (float)i;
        sum += term;
    }

    float p2 = 1.0f;
    float base = (k >= 0) ? 2.0f : 0.5f;
    int n = (k >= 0) ? k : -k;
    for (int i = 0; i < n; i++) p2 *= base;

    return sum * p2;
}

static inline float k_logf(float x)
{
    if (x <= 0.0f) return -1e30f;
    int e = 0;
    while (x > 1.5f) { x *= 0.5f; e++; }
    while (x < 0.75f) { x *= 2.0f; e--; }
    float t = (x - 1.0f) / (x + 1.0f);
    float t2 = t * t;
    float sum = 0.0f;
    float term = t;
    for (int i = 1; i < 20; i += 2) {
        sum += term / (float)i;
        term *= t2;
    }
    return 2.0f * sum + (float)e * 0.6931471805599453f;
}

static inline float k_powf(float base, float exp)
{
    if (base <= 0.0f) return 0.0f;
    return k_expf(exp * k_logf(base));
}

static inline void k_sincosf(float x, float *s, float *c)
{
    const float twopi = 6.283185307179586f;
    while (x > 3.14159265f) x -= twopi;
    while (x < -3.14159265f) x += twopi;

    float x2 = x * x;
    float sn = x;
    float term = x;
    for (int i = 1; i < 8; i++) {
        term *= -x2 / ((2 * i) * (2 * i + 1));
        sn += term;
    }

    float cs = 1.0f;
    term = 1.0f;
    for (int i = 1; i < 8; i++) {
        term *= -x2 / ((2 * i - 1) * (2 * i));
        cs += term;
    }

    *s = sn;
    *c = cs;
}

#endif
