// formato do pacote de entrada lido pela aplicacao do ARM

#ifndef PACOTE_H
#define PACOTE_H

#include <stdint.h>

// versao 5: janelas independentes em int8, uma decisao por janela.
// versao 6: FLUXO CONTINUO - um sinal so', sem fronteira de janela, e uma
//           decisao a cada AMOSTRAS_POR_DECISAO amostras. A verdade vem -1
//           quando a janela da decisao atravessa duas gravacoes.
#define PACOTE_JANELA 5
#define PACOTE_FLUXO  6

typedef struct {
    uint32_t  versao;
    uint32_t  n_vec;      // janelas (v5) ou amostras do fluxo (v6)
    uint32_t  in_len;     // amostras por janela (v5) ou decisoes (v6)
    double    escala;     // escala da entrada, fixada no treino
    int8_t   *jan;        // janelas (v5) ou o sinal continuo (v6)
    int32_t  *modelo;     // classe prevista pelo modelo quantizado
    int32_t  *verdade;    // classe real; -1 = decisao que nao conta
} pacote_t;

int  pacote_abre(const char *caminho, pacote_t *p);
void pacote_fecha(pacote_t *p);

#endif
