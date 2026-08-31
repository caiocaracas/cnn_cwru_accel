// linha de base em FLUXO CONTINUO no ARM, para a comparacao ser justa.

#include <stdint.h>
#include <string.h>
#include "pesos.h"

#define MAXCH   256
#define MAXK      8

typedef struct {
    int      nif, nof, k, pool, npos_saida;
    int8_t   lin[MAXCH][MAXK];
    int      preenchido;
    int32_t  pool_acc[MAXCH];
    int      pool_ph;
} camada_t;

static camada_t cam[N_CONV];
static int8_t   sai[N_CONV][MAXCH];

static int8_t   anel[1024][MAXCH];
static int32_t  soma[MAXCH];
static int      anel_pos, anel_cheio;

void fluxo_sw_inicia(void)
{
    memset(cam, 0, sizeof(cam));
    memset(anel, 0, sizeof(anel));
    memset(soma, 0, sizeof(soma));
    anel_pos = anel_cheio = 0;
    for (int i = 0; i < N_CONV; i++) {
        cam[i].nif  = NIF_TAB[i];
        cam[i].nof  = NOF_TAB[i];
        cam[i].k    = K_TAB[i];
        cam[i].pool = POOL_TAB[i];
        cam[i].pool_ph = 0;
        cam[i].preenchido = 0;
        for (int c = 0; c < cam[i].nof; c++)
            cam[i].pool_acc[c] = POOL_AVG ? 0 : -2147483647;
    }
}

int fluxo_sw_amostra(int8_t x, int32_t *logits, int *classe)
{
    int8_t ent[MAXCH];
    ent[0] = x;
    int nent = 1, tem = 1;

    for (int i = 0; i < N_CONV && tem; i++) {
        camada_t *c = &cam[i];
        for (int ch = 0; ch < nent; ch++) {
            for (int t = c->k - 1; t > 0; t--) c->lin[ch][t] = c->lin[ch][t-1];
            c->lin[ch][0] = ent[ch];
        }
        if (c->preenchido < c->k) { c->preenchido++; }

        for (int oc = 0; oc < c->nof; oc++) {
            int32_t acc = B_TAB[i][oc];
            const int8_t *w = W_TAB[i] + (size_t)oc * c->nif * c->k;
            for (int ic = 0; ic < c->nif; ic++)
                for (int t = 0; t < c->k; t++)
                    acc += (int32_t)w[ic*c->k + t] * (int32_t)c->lin[ic][t];
            if (acc < 0) acc = 0;
            if (POOL_AVG) c->pool_acc[oc] += acc;
            else if (c->pool_ph == 0 || acc > c->pool_acc[oc]) c->pool_acc[oc] = acc;
        }

        if (++c->pool_ph < c->pool) { tem = 0; break; }
        c->pool_ph = 0;
        const int32_t mult = MULT_TAB[i][0];
        for (int oc = 0; oc < c->nof; oc++) {
            int32_t v = c->pool_acc[oc];
            if (POOL_AVG) v /= c->pool;
            int64_t q = ((int64_t)v * mult + (1 << 15)) >> 16;
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            sai[i][oc] = (int8_t)q;
            c->pool_acc[oc] = POOL_AVG ? 0 : -2147483647;
        }
        memcpy(ent, sai[i], c->nof);
        nent = c->nof;
    }
    if (!tem) return 0;

    const int npos = IN_LEN / AMOSTRAS_POR_DECISAO;
    for (int ch = 0; ch < nent; ch++) {
        soma[ch] += ent[ch] - (anel_cheio ? anel[anel_pos][ch] : 0);
        anel[anel_pos][ch] = ent[ch];
    }
    if (++anel_pos >= npos) { anel_pos = 0; anel_cheio = 1; }

    const int8_t *wfc = W_TAB[N_CONV];
    int melhor = 0;
    for (int o = 0; o < N_CLASSES; o++) {
        int32_t acc = B_TAB[N_CONV][o];
        for (int ch = 0; ch < nent; ch++)
            acc += (int32_t)wfc[o*nent + ch] * soma[ch];
        logits[o] = acc;
        if (acc > logits[melhor]) melhor = o;
    }
    *classe = melhor;
    return 1;
}
