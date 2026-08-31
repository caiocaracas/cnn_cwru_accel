// formato do pacote de entrada lido pela aplicacao do ARM.

#ifndef PACOTE_H
#define PACOTE_H

#include <stdint.h>

#define PACOTE_JANELA 5
#define PACOTE_FLUXO  6

typedef struct {
    uint32_t  versao;
    uint32_t  n_vec;
    uint32_t  in_len;
    double    escala;
    int8_t   *jan;
    int32_t  *modelo;
    int32_t  *verdade;
} pacote_t;

int  pacote_abre(const char *caminho, pacote_t *p);
void pacote_fecha(pacote_t *p);

#endif
