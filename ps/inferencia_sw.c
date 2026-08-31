// inferencia em software no ARM, usada como referencia.

#include <stdint.h>
#include <string.h>

#include "pesos.h"

static int32_t acc_buf[MAX_ATIV];
static int8_t  at_a[MAX_ATIV];
static int8_t  at_b[MAX_ATIV];

static inline int8_t requant(int32_t acc, int32_t mult, int shift)
{
    int64_t p = (int64_t)acc * mult + ((int64_t)1 << (shift - 1));
    int32_t q = (int32_t)(p >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static void camada(const int8_t *ent, int nif, int len,
                   const int8_t *w, const int32_t *b,
                   int nof, int k, int pool,
                   const int32_t *mult, uint32_t n_mult,
                   int8_t *sai)
{
    const int pad = (k - 1) / 2;

    for (int oc = 0; oc < nof; oc++) {
        int32_t *a = acc_buf;
        const int32_t bias = b[oc];
        const int32_t mult_oc = mult[n_mult > 1 ? oc : 0];
        for (int p = 0; p < len; p++) a[p] = bias;

        for (int ic = 0; ic < nif; ic++) {
            const int8_t *x = ent + ic * len;
            const int8_t *ww = w + (oc * nif + ic) * k;
            for (int t = 0; t < k; t++) {
                const int32_t wt = ww[t];
                if (!wt) continue;
                const int desl = t - pad;
                int p0 = desl < 0 ? -desl : 0;
                int p1 = desl > 0 ? len - desl : len;
                for (int p = p0; p < p1; p++)
                    a[p] += (int32_t)x[p + desl] * wt;
            }
        }

        int8_t *s = sai + oc * (len / pool);
        for (int p = 0; p < len; p += pool) {
            int32_t m = a[p] > 0 ? a[p] : 0;
            for (int j = 1; j < pool; j++) {
                int32_t v = a[p + j] > 0 ? a[p + j] : 0;
                if (POOL_AVG) m += v;
                else if (v > m) m = v;
            }
            if (POOL_AVG && pool > 1) {
                int lg = 0;
                while ((1 << lg) < pool) lg++;
                m >>= lg;
            }
            s[p / pool] = requant(m, mult_oc, 16);
        }
    }
}

int inferencia_sw(const int8_t *entrada, int32_t *logits)
{
    const int8_t *ent = entrada;
    int8_t *sai = at_a;
    for (int i = 0; i < N_CONV; i++) {
        camada(ent, NIF_TAB[i], LEN_TAB[i], W_TAB[i], B_TAB[i],
               NOF_TAB[i], K_TAB[i], POOL_TAB[i],
               MULT_TAB[i], N_MULT_TAB[i], sai);
        ent = sai;
        sai = (sai == at_a) ? at_b : at_a;
    }

    const int ult = N_CONV - 1;
    const int npos = LEN_TAB[ult] / POOL_TAB[ult];
    const int nch  = NOF_TAB[ult];
#if GAP
    for (int c = 0; c < N_CLASSES; c++) {
        int32_t s = B_TAB[N_CONV][c];
        const int8_t *ww = W_TAB[N_CONV] + (size_t)c * nch;
        for (int ch = 0; ch < nch; ch++) {
            const int32_t wt = ww[ch];
            const int8_t *x = ent + (size_t)ch * npos;
            for (int p = 0; p < npos; p++) s += (int32_t)x[p] * wt;
        }
        logits[c] = s;
    }
#else
    const int nflat = nch * npos;
    for (int c = 0; c < N_CLASSES; c++) {
        int32_t s = B_TAB[N_CONV][c];
        const int8_t *ww = W_TAB[N_CONV] + (size_t)c * nflat;
        for (int i = 0; i < nflat; i++) s += (int32_t)ent[i] * ww[i];
        logits[c] = s;
    }
#endif

    int melhor = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (logits[c] > logits[melhor]) melhor = c;
    return melhor;
}
