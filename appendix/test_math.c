#include <stdio.h>
#include <math.h>
#include "andrea_llm_math.h"

int main(void)
{
    float maxerr = 0;
    const char *worst = "";

    for (float x = 0.001f; x < 100.0f; x += 0.1f) {
        float e = k_fabsf(k_sqrtf(x) - sqrtf(x));
        if (e > maxerr) { maxerr = e; worst = "sqrtf"; }
    }
    printf("sqrtf  max err: %g\n", maxerr); maxerr = 0;

    for (float x = -30.0f; x < 30.0f; x += 0.05f) {
        float rel = k_fabsf(k_expf(x) - expf(x)) / (expf(x) + 1e-9f);
        if (rel > maxerr) maxerr = rel;
    }
    printf("expf   max rel err: %g\n", maxerr); maxerr = 0;

    for (float x = 0.01f; x < 100.0f; x += 0.05f) {
        float e = k_fabsf(k_logf(x) - logf(x));
        if (e > maxerr) maxerr = e;
    }
    printf("logf   max err: %g\n", maxerr); maxerr = 0;

    for (float x = 1.0f; x < 20.0f; x += 0.1f) {
        float e = k_fabsf(k_powf(10000.0f, x/64.0f) - powf(10000.0f, x/64.0f)) / powf(10000.0f, x/64.0f);
        if (e > maxerr) maxerr = e;
    }
    printf("powf   max rel err: %g\n", maxerr); maxerr = 0;

    for (float x = -6.0f; x < 6.0f; x += 0.01f) {
        float s, c;
        k_sincosf(x, &s, &c);
        float es = k_fabsf(s - sinf(x));
        float ec = k_fabsf(c - cosf(x));
        if (es > maxerr) maxerr = es;
        if (ec > maxerr) maxerr = ec;
    }
    printf("sincosf max err: %g\n", maxerr);

    return 0;
}
