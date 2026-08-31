// le o pacote gerado pelo fluxo: janelas int8 (v5) ou sinal continuo (v6)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pacote.h"

#define MAGIC 0x434E4E58u

int pacote_abre(const char *caminho, pacote_t *p)
{
    FILE *f = fopen(caminho, "rb");
    if (!f) { perror(caminho); return -1; }

    uint32_t cab[4];
    if (fread(cab, sizeof(uint32_t), 4, f) != 4) goto ruim;
    if (cab[0] != MAGIC) {
        fprintf(stderr, "%s: identificacao inesperada %08x\n", caminho, cab[0]);
        goto ruim;
    }
    p->n_vec  = cab[1];
    p->in_len = cab[2];
    p->versao = cab[3];
    if (p->versao != PACOTE_JANELA && p->versao != PACOTE_FLUXO) {
        fprintf(stderr, "%s: pacote versao %u, esperado %u ou %u\n",
                caminho, p->versao, PACOTE_JANELA, PACOTE_FLUXO);
        goto ruim;
    }
    if (fread(&p->escala, sizeof(double), 1, f) != 1) goto ruim;

    // v5: n_vec janelas de in_len amostras, e uma decisao por janela
    // v6: n_vec amostras no total, e in_len decisoes
    size_t n_am  = (p->versao == PACOTE_FLUXO)
                 ? (size_t)p->n_vec
                 : (size_t)p->n_vec * p->in_len;
    size_t n_dec = (p->versao == PACOTE_FLUXO) ? p->in_len : p->n_vec;

    p->jan     = malloc(n_am);
    p->modelo  = malloc(n_dec * sizeof(int32_t));
    p->verdade = malloc(n_dec * sizeof(int32_t));
    if (!p->jan || !p->modelo || !p->verdade) goto ruim;

    if (fread(p->jan,     1, n_am, f) != n_am)                    goto ruim;
    if (fread(p->modelo,  sizeof(int32_t), n_dec, f) != n_dec)    goto ruim;
    if (fread(p->verdade, sizeof(int32_t), n_dec, f) != n_dec)    goto ruim;

    fclose(f);
    return 0;
ruim:
    fprintf(stderr, "%s: pacote incompleto ou corrompido\n", caminho);
    fclose(f);
    return -1;
}

void pacote_fecha(pacote_t *p)
{
    free(p->jan);  free(p->modelo);  free(p->verdade);
    memset(p, 0, sizeof(*p));
}
