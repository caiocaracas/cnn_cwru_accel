// orquestra o acelerador pelo ARM, mede tempo e confere o resultado

#define _POSIX_C_SOURCE 200809L

#define _GNU_SOURCE   // CPU_ZERO/CPU_SET
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>

#include "pesos.h"
#include "pacote.h"

#ifndef CAMINHO_PACOTE
#define CAMINHO_PACOTE "/usr/share/acelerador/entrada_ps.bin"
#endif

#define REG_CTRL     0x00
#define REG_STATUS   0x04
#define REG_CLASSE   0x08
#define REG_CICLOS   0x0C
#define REG_LOGIT0   0x10
#define REG_LD_CTRL  0x20
#define REG_LD_W     0x24
#define REG_LD_B     0x28
#define REG_IN_DATA  0x2C
#define REG_IN_DATA4 0x30
#define REG_OCUP     0x3C
#define REG_LD_M     0x40
#define REG_LEDS     0x44
// so' em fluxo continuo: a PL empurra decisao numa fila e o ARM drena. Ler
// 0x48 TIRA uma; bit 8 diz se havia. 0x58 e' a ocupacao, 0x5C o que se perdeu
#define REG_RES      0x48
#define REG_RES_OCUP 0x58
#define REG_PERDIDA  0x5C
#define REG_NDONE    0x4C
#define REG_TIQUE    0x50
#define REG_LAT      0x54
// jitter medido pela propria PL: intervalo entre decisoes em ciclos, e os
// ciclos em que o motor aceitaria amostra e nao havia (culpa do alimentador)
#define REG_IV_MIN   0x60
#define REG_IV_MAX   0x64
#define REG_IV_N     0x68
#define REG_FOME     0x6C

#define ST_BUSY      (1u << 0)
#define ST_DONE      (1u << 1)
#define ST_PRONTA    (1u << 5)   // janela inteira na fila
#define ST_TRUNC     (0xFFu << 8)

#define MM2S_DMACR   0x00
#define MM2S_DMASR   0x04
#define MM2S_SA      0x18
#define MM2S_LENGTH  0x28

#define DMACR_RS     (1u << 0)
#define DMACR_RESET  (1u << 2)
#define DMACR_IOC_EN (1u << 12)
#define DMASR_HALTED (1u << 0)
#define DMASR_IDLE   (1u << 1)
#define DMASR_ERRO   (0x70u)
#define DMASR_IOC    (1u << 12)

static volatile uint32_t *acel;
static volatile uint32_t *dma;
static volatile uint32_t *buf;

static inline void  wr(uint32_t off, uint32_t v) { acel[off >> 2] = v; }
static inline uint32_t rd(uint32_t off)          { return acel[off >> 2]; }
static inline void  dwr(uint32_t off, uint32_t v) { dma[off >> 2] = v; }
static inline uint32_t drd(uint32_t off)          { return dma[off >> 2]; }

#define GTC_BASE 0xF8F00200u
static volatile uint32_t *gtc;

static uint64_t agora(void)
{
    uint32_t hi, lo, hi2;
    do {
        hi  = gtc[1];
        lo  = gtc[0];
        hi2 = gtc[1];
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

static double us_por_tique = 0.0;
static long long afere_ns = 0;
static uint64_t  afere_tiques = 0;

static void afere_tique(void)
{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    const uint64_t t0 = agora();
    long long ns;
    do {
        clock_gettime(CLOCK_MONOTONIC, &b);
        ns = (long long)(b.tv_sec - a.tv_sec) * 1000000000LL
           + (long long)(b.tv_nsec - a.tv_nsec);
    } while (ns < 200000000LL);
    const uint64_t dt = agora() - t0;
    afere_ns     = ns;
    afere_tiques = dt;
    us_por_tique = dt ? ((double)ns / 1000.0) / (double)dt : 2.0 / ARM_CLK_MHZ;
}

static void gtc_inicia(void)
{

    if (!(gtc[2] & 1u)) gtc[2] = 1u;
}

// relogio da PL por medida direta: um contador livre lido em dois instantes
// cronometrados pelo processador. Dividir os ciclos da inferencia pelo tempo
// de espera do ARM superestimaria o relogio, por causa da sobreposicao.
static double afere_clock_pl(void)
{
    const uint32_t a = rd(REG_TIQUE);
    const uint64_t ta = agora();
    const uint64_t alvo = (uint64_t)(0.2e6 / us_por_tique);   // ~200 ms
    while (agora() - ta < alvo) ;
    const uint32_t b = rd(REG_TIQUE);
    const uint64_t tb = agora();
    const double us = (double)(tb - ta) * us_por_tique;
    return (double)(b - a) / us;
}

static uint64_t carrega_tudo(void)
{
    uint64_t t0 = agora();
    for (int i = 0; i < N_CONV; i++) {
        wr(REG_LD_CTRL, ((uint32_t)i & 15u) | (1u << 6));
        for (uint32_t c = 0; c < N_MULT_TAB[i]; c++)
            wr(REG_LD_M, (uint32_t)MULT_TAB[i][c] & 0x3FFFFu);
    }
    return agora() - t0;
}

// a janela chega do pacote ja' normalizada e em int8: a estatistica por
// janela e a divisao pelo desvio acontecem no preparo dos dados, junto com o
// treino. O processador nao toca mais na amostra - so' a transfere.

static int dma_inicia(void)
{
    dwr(MM2S_DMACR, DMACR_RESET);
    for (int i = 0; i < 1000000 && (drd(MM2S_DMACR) & DMACR_RESET); i++) ;
    if (drd(MM2S_DMACR) & DMACR_RESET) return -1;

    dwr(MM2S_DMACR, DMACR_RS | DMACR_IOC_EN);
    for (int i = 0; i < 1000000 && (drd(MM2S_DMASR) & DMASR_HALTED); i++) ;
    return (drd(MM2S_DMASR) & DMASR_HALTED) ? -1 : 0;
}

// quatro amostras int8 por palavra de 32 bits, na mesma ordem em que a PL
// as desempacota do barramento
static void copia_para_buf(const int8_t *x)
{
    const uint32_t *s = (const uint32_t *)(const void *)x;
    for (uint32_t i = 0; i < IN_LEN / 4u; i++) buf[i] = s[i];
}

// arrancar e esperar sao separados de proposito: em fluxo o ARM enche o outro
// buffer e drena as decisoes DENTRO da sombra desta transferencia. Enquanto as
// duas coisas eram um bloco so', o motor ficava sem dado no intervalo entre um
// lote e o proximo - medido: 0,41% dos ciclos, esticando o intervalo entre
// decisoes de 80 para ate' 642 ciclos
static void dma_arranca(uint32_t off, uint32_t n)
{
    dwr(MM2S_DMASR, DMASR_IOC);
    dwr(MM2S_SA, BUF_DMA_FIS + off);
    dwr(MM2S_LENGTH, n);
}

static int dma_espera(void)
{
    uint64_t lim = agora() + (uint64_t)ARM_CLK_MHZ * 1000ull;
    for (;;) {
        uint32_t s = drd(MM2S_DMASR);
        if (s & DMASR_ERRO) return -1;
        if (s & DMASR_IOC) {
            dwr(MM2S_DMASR, DMASR_IOC);
            return 0;
        }
        if (agora() > lim)  return -2;
    }
}

static int dma_dispara(uint32_t n)
{
    dma_arranca(0, n);
    return dma_espera();
}

typedef struct {
    int32_t  logits[N_CLASSES];
    uint32_t classe;
    uint32_t ciclos_pl;
    uint64_t t_envio;
    uint64_t t_espera;
    uint64_t t_leitura;
    uint64_t t_e2e;
    uint32_t trunc;
    uint32_t ocup;
    uint32_t lat_hw;
} resultado_t;

typedef struct {
    uint64_t t_envio;
    uint32_t ocup;
} envio_t;

// transfere a janela para a fila da PL. Uma amostra e' um byte, entao a
// rajada e' de IN_LEN bytes e cada palavra do barramento leva quatro amostras
static int envia(const int8_t *x, envio_t *e, int por_rajada)
{
    uint64_t te0 = agora();
    if (por_rajada) {
        copia_para_buf(x);
        int err = dma_dispara(IN_LEN);
        if (err) {
            printf("  DMA falhou (%d): dmasr=%08lx\n", err,
                   (unsigned long)drd(MM2S_DMASR));
            return -1;
        }
    } else {
        uint32_t i = 0;
        for (; i + 4u <= IN_LEN; i += 4u)
            wr(REG_IN_DATA4, (uint32_t)(uint8_t)x[i]
                           | ((uint32_t)(uint8_t)x[i+1] <<  8)
                           | ((uint32_t)(uint8_t)x[i+2] << 16)
                           | ((uint32_t)(uint8_t)x[i+3] << 24));
        for (; i < IN_LEN; i++)
            wr(REG_IN_DATA, (uint32_t)(uint8_t)x[i]);
    }
    e->t_envio = agora() - te0;
    e->ocup    = rd(REG_OCUP);
    return 0;
}

// arma a partida: o motor so' arranca quando a janela inteira estiver na fila,
// entao armar antes de a transferencia terminar tambem vale
static void arma(void) { wr(REG_CTRL, 1u); }

static int colhe(resultado_t *r)
{
    uint64_t tw0 = agora();
    uint64_t lim = tw0 + (uint64_t)ARM_CLK_MHZ * 1000ull;
    while (!(rd(REG_STATUS) & ST_DONE)) {
        if (agora() > lim) {
            printf("  TIMEOUT: done nao chegou. status=%08lx ocup=%lu\n",
                   (unsigned long)rd(REG_STATUS), (unsigned long)rd(REG_OCUP));
            r->classe = 0xFFu;
            return -1;
        }
    }
    r->t_espera = agora() - tw0;

    uint64_t tl0 = agora();
    for (int c = 0; c < N_CLASSES; c++)
        r->logits[c] = (int32_t)rd(REG_LOGIT0 + 4u * c);
    r->classe    = rd(REG_CLASSE) & 15u;
    r->ciclos_pl = rd(REG_CICLOS);
    r->lat_hw    = rd(REG_LAT);
    r->trunc     = (rd(REG_STATUS) & ST_TRUNC) >> 8;
    r->t_leitura = agora() - tl0;
    return 0;
}

extern int inferencia_sw(const int8_t *entrada, int32_t *logits);

static void compara_com_arm(const int8_t *q, const int32_t *logits_pl,
                            uint64_t *ticks, int *iguais)
{
    int32_t logits_sw[N_CLASSES];
    uint64_t t0 = agora();
    inferencia_sw(q, logits_sw);
    *ticks = agora() - t0;

    *iguais = 1;
    for (int c = 0; c < N_CLASSES; c++)
        if (logits_sw[c] != logits_pl[c]) *iguais = 0;
}

static void carrega_camada(uint32_t sel,
                           const int8_t *w, uint32_t nw,
                           const int32_t *b, uint32_t nb)
{
    wr(REG_LD_CTRL, (sel & 15u) | (1u << 4) | (1u << 5));
    for (uint32_t i = 0; i < nw; i++) wr(REG_LD_W, (uint32_t)(uint8_t)w[i]);
    for (uint32_t i = 0; i < nb; i++) wr(REG_LD_B, (uint32_t)b[i]);
}

static uint64_t carrega_pelo_barramento(uint32_t mascara)
{
    uint64_t t0 = agora();
    for (int i = 0; i < N_CAMADAS; i++)
        if (mascara & (1u << i))
            carrega_camada((uint32_t)i, W_TAB[i], N_W_TAB[i], B_TAB[i], N_B_TAB[i]);
    return agora() - t0;
}

// os LEDs sao do hardware: acendem os quatro quando a inferencia termina, sem
// depender do processador. Aqui o ARM os toma emprestado para sinalizar que o
// fluxo inteiro acabou, e devolve em seguida
static void leds_pisca(int vezes)
{
    struct timespec meio = {0, 200000000L};   // 200 ms
    for (int i = 0; i < vezes; i++) {
        wr(REG_LEDS, 0x1Fu);          // bit 4 toma o controle, 0xF acende tudo
        nanosleep(&meio, NULL);
        wr(REG_LEDS, 0x10u);          // ainda no controle, tudo apagado
        nanosleep(&meio, NULL);
    }
    wr(REG_LEDS, 0x1Fu);              // termina aceso
    nanosleep(&meio, NULL);
    wr(REG_LEDS, 0x00u);              // devolve os LEDs ao hardware
}

static double g_prazo_us = 0.0;   // prazo por janela, se informado
static void percentis(const char *rot, uint64_t *v, uint32_t n, double us_tick,
                      double prazo_us);

static uint32_t campanha(const char *rotulo, pacote_t *p, int com_ab,
                         uint32_t n_max)
{
    printf("\n########## %s ##########\n", rotulo);
    const uint32_t n_vec = (n_max && n_max < p->n_vec) ? n_max : p->n_vec;

    uint32_t acertos = 0, concorda = 0, n_trunc = 0;
    // se a classe do hardware bate com a da janela vizinha, a discordancia e'
    // escorregao de sincronismo e nao erro de aritmetica
    uint32_t bate_anterior = 0, bate_seguinte = 0;
    int32_t primeira_dif = -1;
    uint64_t soma_e2e = 0, soma_env = 0, soma_pl = 0, soma_sw = 0, soma_diag = 0;
    uint64_t soma_esp = 0, soma_lei = 0, soma_sis = 0;
    uint32_t pl_min = 0xFFFFFFFFu, pl_max = 0;
    uint32_t lat_min = 0xFFFFFFFFu, lat_max = 0;
    uint64_t e2e_min = ~0ull, e2e_max = 0, sis_min = ~0ull, sis_max = 0;
    // guarda janela a janela: a garantia de prazo se caracteriza pela cauda,
    // e min/max nao distinguem estouro isolado de jitter sistematico
    uint64_t *lat_e2e = (uint64_t *)malloc((size_t)n_vec * sizeof(uint64_t));
    uint32_t *lat_pl  = (uint32_t *)malloc((size_t)n_vec * sizeof(uint32_t));
    uint32_t logits_iguais = 0;
    const uint32_t n_sw = n_vec < 32 ? n_vec : 32;
    uint32_t vistos_sw = 0;

    const uint32_t n_ab = n_vec < 200 ? n_vec : 200;
    uint64_t soma_env_reg = 0;
    uint32_t vistos_ab = 0, ab_iguais = 0;
    uint32_t ocup_min = 0xFFFFFFFFu;

    const double pl_mhz_aferido = afere_clock_pl();

    if (com_ab) {
        wr(REG_CTRL, 2u);
        for (uint32_t v = 0; v < n_ab; v++) {
            const int8_t *c2 = &p->jan[(size_t)v * p->in_len];
            envio_t e1, e2; resultado_t r1, r2;
            if (envia(c2, &e1, 1)) break;
            arma();
            if (colhe(&r1)) break;
            if (envia(c2, &e2, 0)) break;
            arma();
            if (colhe(&r2)) break;
            int ig = 1;
            for (int c = 0; c < N_CLASSES; c++)
                if (r1.logits[c] != r2.logits[c]) ig = 0;
            ab_iguais    += (uint32_t)ig;
            soma_env_reg += e2.t_envio;
            vistos_ab++;
        }
    }

    // a fila guarda duas janelas: a seguinte e' transferida enquanto a atual
    // e' processada, e por isso o envio sai do caminho critico
    wr(REG_CTRL, 2u);
    envio_t env, prox;
    if (envia(&p->jan[0], &env, 1)) return 0;
    arma();

    const uint64_t t_campanha = agora();
    for (uint32_t v = 0; v < n_vec; v++) {
        const int8_t *jan = &p->jan[(size_t)v * p->in_len];
        int tem_prox = 0;

        if (v + 1 < n_vec)
            tem_prox = (envia(&p->jan[(size_t)(v+1) * p->in_len], &prox, 1) == 0);

        resultado_t r;
        r.t_envio  = env.t_envio;
        r.ocup     = env.ocup;
        if (colhe(&r)) { printf("  janela %u: sem resposta\n", v); break; }
        r.t_e2e = r.t_envio + r.t_espera + r.t_leitura;
        if (tem_prox) { env = prox; arma(); }
        if (r.ocup < ocup_min) ocup_min = r.ocup;

        if ((int32_t)r.classe == p->verdade[v]) acertos++;
        if ((int32_t)r.classe == p->modelo[v]) {
            concorda++;
        } else {
            if (primeira_dif < 0) primeira_dif = (int32_t)v;
            if (v > 0 && (int32_t)r.classe == p->modelo[v - 1]) bate_anterior++;
            if (v + 1 < n_vec && (int32_t)r.classe == p->modelo[v + 1])
                bate_seguinte++;
        }
        if (r.trunc) n_trunc++;

        soma_e2e += r.t_e2e;  soma_env += r.t_envio;  soma_pl += r.ciclos_pl;
        soma_esp += r.t_espera; soma_lei += r.t_leitura;
        uint64_t t_sis = r.t_e2e;
        soma_sis += t_sis;
        if (r.lat_hw && r.lat_hw < lat_min) lat_min = r.lat_hw;
        if (r.lat_hw > lat_max) lat_max = r.lat_hw;
        if (r.ciclos_pl < pl_min) pl_min = r.ciclos_pl;
        if (r.ciclos_pl > pl_max) pl_max = r.ciclos_pl;
        if (r.t_e2e < e2e_min) e2e_min = r.t_e2e;
        if (r.t_e2e > e2e_max) e2e_max = r.t_e2e;
        if (t_sis < sis_min) sis_min = t_sis;
        if (t_sis > sis_max) sis_max = t_sis;
        if (lat_e2e) lat_e2e[v] = r.t_e2e;
        if (lat_pl)  lat_pl[v]  = r.ciclos_pl;

        // conferencia contra o ARM: nao faz parte da inferencia, mas roda
        // dentro do laco, entao vai cronometrada e descontada
        const uint64_t td0 = agora();
        if (vistos_sw < n_sw) {
            uint64_t t_sw; int ig;
            compara_com_arm(jan, r.logits, &t_sw, &ig);
            soma_sw += t_sw;
            logits_iguais += (uint32_t)ig;
            vistos_sw++;
        }
        soma_diag += agora() - td0;
        if (v == 0) {

            int32_t sw[N_CLASSES];
            inferencia_sw(jan, sw);
            printf("  janela 0, PL x ARM por classe:\n");
            for (int c = 0; c < N_CLASSES; c++)
                printf("    classe %d: PL %11ld   ARM %11ld   dif %11ld\n",
                       c, (long)r.logits[c], (long)sw[c],
                       (long)(r.logits[c] - sw[c]));
        }
        if (n_vec > 200 && (v % 200) == 0)
            printf("  ... %u/%u\n", v, n_vec);
    }

    const double us_tick = us_por_tique;
    const double pl_mhz = (pl_mhz_aferido > 1.0) ? pl_mhz_aferido
                                                : (double)PL_CLK_MHZ;
    double pl_us   = (double)(soma_pl / n_vec) / pl_mhz;
    double e2e_us  = (double)(soma_e2e / n_vec) * us_tick;

    printf("\n=== RESULTADO ===\n");
    printf("  acuracia            : %.2f%%  (%u/%u)\n",
           100.0 * acertos / n_vec, acertos, n_vec);
    printf("  concorda com o modelo: %.2f%%  (%u/%u)\n",
           100.0 * concorda / n_vec, concorda, n_vec);
    printf("  estouro de acumulador: %u\n", n_trunc);
    {
        // o contador zera no ultimo 'limpa', que vem logo antes do laco
        const uint32_t nd = rd(REG_NDONE);
        printf("  inferencias: a PL concluiu %u, o processador colheu %u%s\n",
               nd, n_vec, (nd == n_vec) ? "" : "   <-- ESCORREGOU");
    }
    if (concorda < n_vec) {
        const uint32_t difs = n_vec - concorda;
        printf("  DIVERGENCIA: %u janelas. primeira em %d. "
               "bate com a anterior em %u (%.1f%%), com a seguinte em %u (%.1f%%)\n",
               difs, (int)primeira_dif,
               bate_anterior, 100.0 * bate_anterior / difs,
               bate_seguinte, 100.0 * bate_seguinte / difs);
    }

    // com duas janelas na fila a ocupacao minima depois do envio fica acima
    // de IN_LEN: e' a prova de que a janela seguinte chegou antes da hora
    // minimo entre as janelas: o aviso do DMA chega antes de a ultima rajada
    // assentar na fila, e por isso o motor espera a janela inteira (jan_pronta)
    // em vez de confiar no aviso
    printf("  fila apos o envio   : %u amostras, minimo (janela de %u)\n",
           ocup_min, IN_LEN);

    const uint64_t t_parede = agora() - t_campanha;
    double sis_us  = (double)(t_parede / n_vec) * us_tick;
    double diag_us = (double)(soma_diag / n_vec) * us_tick;
    double fases_us = (double)(soma_sis / n_vec) * us_tick;
    printf("\n=== TEMPO ===\n");
    printf("  computo na PL       : %u..%u ciclos, jitter %u  (%.2f us)\n",
           pl_min, pl_max, pl_max - pl_min, pl_us);
    if (lat_max)
        printf("  janela -> resultado : %u..%u ciclos, jitter %u  (%.2f us)"
               "  [so' hardware]\n", lat_min, lat_max, lat_max - lat_min,
               (double)((lat_min + lat_max) / 2) / pl_mhz);
    printf("  envio da janela     : %.2f us  (rajada pelo DMA)\n",
           (double)(soma_env / n_vec) * us_tick);
    if (vistos_ab)
        printf("  envio por registrador: %.2f us  (referencia, %u janelas, "
               "%u/%u com logits identicos)\n",
               (double)(soma_env_reg / vistos_ab) * us_tick, vistos_ab,
               ab_iguais, vistos_ab);
    printf("  espera pelo done    : %.2f us\n",
           (double)(soma_esp / n_vec) * us_tick);
    printf("  leitura do resultado: %.2f us\n",
           (double)(soma_lei / n_vec) * us_tick);
    printf("  acelerador          : %.2f us  (%.2f..%.2f, jitter %.2f)\n",
           e2e_us, (double)e2e_min * us_tick, (double)e2e_max * us_tick,
           (double)(e2e_max - e2e_min) * us_tick);
    printf("  relogio da PL       : %.2f MHz aferido  (a sintese fechou em %u; "
           "desvio %+.1f%%)\n", pl_mhz_aferido, PL_CLK_MHZ,
           100.0 * (pl_mhz_aferido / (double)PL_CLK_MHZ - 1.0));
    printf("  conferencias        : %.2f us  (sobrepostas ao computo)\n",
           diag_us);
    printf("  sistema completo    : %.2f us  (relogio de parede / janelas)\n",
           sis_us);
    printf("  soma das fases      : %.2f us  (%.2f..%.2f, jitter %.2f) - sem "
           "sobreposicao seria isto\n",
           fases_us, (double)sis_min * us_tick, (double)sis_max * us_tick,
           (double)(sis_max - sis_min) * us_tick);
    // o envio da janela se sobrepoe ao computo, entao o tempo medido pelo ARM
    // pode ser menor que o da PL; supor que um contenha o outro daria
    // porcentagem negativa
    if (e2e_us <= pl_us)
        printf("  fora do computo     : 0.0%%  (o ARM cabe inteiro dentro do "
               "computo da PL: %.2f contra %.2f us)\n", e2e_us, pl_us);
    else
        printf("  fora do computo     : %.1f%%\n",
               100.0 * (1.0 - pl_us / e2e_us));

    // separa a garantia do circuito da garantia do sistema em volta, que e'
    // a fraca, por o orquestrador rodar em Linux
    if (lat_pl) {
        uint32_t k;
        uint64_t *v64 = (uint64_t *)malloc((size_t)n_vec * sizeof(uint64_t));
        if (v64) {
            for (k = 0; k < n_vec; k++) v64[k] = lat_pl[k];
            printf("  --- garantia do circuito (ciclos da PL) ---\n");
            percentis("PL                  ", v64, n_vec,
                      1.0 / (double)PL_CLK_MHZ, g_prazo_us);
            free(v64);
        }
    }
    if (lat_e2e) {
        printf("  --- garantia do sistema (ARM em Linux) ---\n");
        percentis("sistema             ", lat_e2e, n_vec, us_tick, g_prazo_us);
    }
    free(lat_e2e); free(lat_pl);
    printf("  vazao               : %.0f inferencias/s\n", 1e6 / sis_us);

    if (vistos_sw && soma_sw) {
        double sw_us = (double)(soma_sw / vistos_sw) * us_tick;
        printf("\n=== CONTRA O PROCESSADOR DA MESMA PLACA ===\n");
        printf("  logits identicos    : %u/%u\n", logits_iguais, vistos_sw);
        printf("  ARM                 : %.1f us\n", sw_us);
        printf("  ganho no computo    : %.1fx\n", sw_us / pl_us);
        printf("  ganho de sistema    : %.1fx\n", sw_us / e2e_us);
    }

    return acertos;
}

// ------------------------------------------------------------------------
// campanha em FLUXO CONTINUO
//
// Nao ha' janela: o sinal entra sem parar e a decisao sai a cada
// AMOSTRAS_POR_DECISAO amostras. O ARM alterna empurrar um pedaco e drenar a
// fila; o pedaco e' dimensionado para a fila nunca encher, porque decisao
// perdida invalidaria a medida em silencio.
// ------------------------------------------------------------------------
static uint32_t campanha_fluxo(pacote_t *p, uint32_t n_max)
{
    const uint32_t n_dec = (n_max && n_max < p->in_len) ? n_max : p->in_len;
    const uint32_t dec_lote = 256;
    const uint32_t am_lote  = dec_lote * AMOSTRAS_POR_DECISAO;
    uint32_t vistas = 0, acertos = 0, contadas = 0, concorda = 0;
    int32_t  primeira_dif = -1;
    // a decisao vem NUMERADA pelo hardware: se a sequencia pular, alguma se
    // perdeu entre a fila e aqui, e todas as seguintes ficam deslocadas em
    // relacao a' referencia. Sem esta conferencia o deslocamento aparecia como
    // erro de aritmetica nas transicoes de classe
    uint32_t saltos = 0, perdidas_seq = 0;
    int32_t  seq_ant = -1, primeiro_salto = -1;

    printf("\n########## fluxo continuo ##########\n");
    printf("  %u amostras, %u decisoes esperadas, uma a cada %u amostras\n",
           p->n_vec, p->in_len, AMOSTRAS_POR_DECISAO);

    const double pl_mhz = afere_clock_pl();

    // linha de base: o MESMO trabalho em fluxo, no ARM. Comparar contra a
    // versao por janela inflaria o ganho em 128x de graca, porque o ARM
    // estaria recomputando o que o circuito nao recomputa
    extern void fluxo_sw_inicia(void);
    extern int  fluxo_sw_amostra(int8_t x, int32_t *logits, int *classe);
    const uint32_t n_sw = (p->n_vec < 20000u) ? p->n_vec : 20000u;
    uint64_t t_sw = 0;
    {
        int32_t lg[N_CLASSES]; int cl; uint32_t dec_sw = 0;
        fluxo_sw_inicia();
        const uint64_t s0 = agora();
        for (uint32_t i = 0; i < n_sw; i++)
            dec_sw += (uint32_t)fluxo_sw_amostra(p->jan[i], lg, &cl);
        t_sw = agora() - s0;
        printf("  linha de base em fluxo no ARM: %u amostras, %u decisoes, "
               "%.1f ms -> %.0f amostras/s\n", n_sw, dec_sw,
               (double)t_sw * us_por_tique / 1000.0,
               1e6 * n_sw / ((double)t_sw * us_por_tique));
    }

    wr(REG_CTRL, 2u);                       // limpa fila e contadores

    // arma ANTES de alimentar. Em fluxo o motor parte com a primeira amostra
    // e nunca mais para; armar depois do primeiro envio trava, porque o lote e'
    // maior que a fila de entrada: ela enche, o DMA espera vaga e o motor, que
    // nunca partiu, nao consome. Na arquitetura por janela isso nao aparecia
    // porque uma janela cabia inteira na fila
    arma();

    uint64_t enviadas = 0;
    const uint64_t t0 = agora();
    const uint32_t tq0 = rd(REG_TIQUE);

    // dois buffers alternando. Enquanto o DMA drena um deles - e ele demora o
    // que o motor levar para consumir, porque a fila de entrada faz
    // contrapressao -, o ARM enche o outro e colhe as decisoes. O unico
    // trabalho entre um lote e o proximo passa a ser escrever dois
    // registradores do DMA, e a fila de entrada nunca seca.
    const uint32_t lado_bytes = (am_lote + 63u) & ~63u;
    uint32_t lado = 0;
    uint32_t n_voo = 0;

    // enche e arranca o primeiro lote
    {
        uint32_t n = (uint32_t)((p->n_vec < am_lote) ? p->n_vec : am_lote) & ~3u;
        if (n) {
            const uint32_t *s = (const uint32_t *)(const void *)p->jan;
            volatile uint32_t *d = buf + (lado * lado_bytes) / 4u;
            for (uint32_t i = 0; i < n / 4u; i++) d[i] = s[i];
            dma_arranca(lado * lado_bytes, n);
            n_voo = n;
            enviadas = n;
        }
    }

    while (n_voo && vistas < n_dec) {
        // ---- dentro da sombra do DMA em voo ----
        uint32_t n_prox = 0;
        if (enviadas < p->n_vec) {
            n_prox = (uint32_t)((p->n_vec - enviadas) < am_lote
                                ? (p->n_vec - enviadas) : am_lote) & ~3u;
            if (n_prox) {
                const uint32_t *s =
                    (const uint32_t *)(const void *)(p->jan + enviadas);
                volatile uint32_t *d = buf + ((1u - lado) * lado_bytes) / 4u;
                for (uint32_t i = 0; i < n_prox / 4u; i++) d[i] = s[i];
            }
        }

        for (;;) {                           // drena o que ja' saiu
            uint32_t r = rd(REG_RES);
            if (!(r & 0x100u)) break;
            const int32_t seq = (int32_t)((r >> 4) & 15u);
            if (seq_ant >= 0) {
                const int32_t d = (seq - seq_ant + 16) & 15;
                if (d != 1) {
                    saltos++;
                    perdidas_seq += (uint32_t)((d + 15) & 15);
                    if (primeiro_salto < 0) primeiro_salto = (int32_t)vistas;
                }
            }
            seq_ant = seq;
            if (vistas < n_dec) {
                int32_t c = (int32_t)(r & 15u);
                if (c == p->modelo[vistas]) concorda++;
                else if (primeira_dif < 0)   primeira_dif = (int32_t)vistas;
                if (p->verdade[vistas] >= 0) {
                    contadas++;
                    if (c == p->verdade[vistas]) acertos++;
                }
            }
            vistas++;
        }

        // ---- so' agora se paga a espera; o proximo lote ja' esta' pronto ----
        int err = dma_espera();
        if (err) {
            printf("  DMA falhou (%d) apos %llu amostras: dmasr=%08lx "
                   "ocup=%lu\n", err, (unsigned long long)enviadas,
                   (unsigned long)drd(MM2S_DMASR),
                   (unsigned long)rd(REG_OCUP));
            break;
        }
        if (!n_prox) { n_voo = 0; break; }
        lado ^= 1u;
        dma_arranca(lado * lado_bytes, n_prox);
        enviadas += n_prox;
        n_voo = n_prox;
    }

    // o cano ainda tem decisao dentro: deixa escoar antes de fechar a conta
    uint64_t lim = agora() + (uint64_t)ARM_CLK_MHZ * 200ull;
    while (agora() < lim) {
        uint32_t r = rd(REG_RES);
        if (!(r & 0x100u)) continue;
        {
            const int32_t seq = (int32_t)((r >> 4) & 15u);
            if (seq_ant >= 0) {
                const int32_t d = (seq - seq_ant + 16) & 15;
                if (d != 1) {
                    saltos++;
                    perdidas_seq += (uint32_t)((d + 15) & 15);
                    if (primeiro_salto < 0) primeiro_salto = (int32_t)vistas;
                }
            }
            seq_ant = seq;
        }
        if (vistas < n_dec) {
            int32_t c = (int32_t)(r & 15u);
            if (c == p->modelo[vistas]) concorda++;
            else if (primeira_dif < 0)   primeira_dif = (int32_t)vistas;
            if (p->verdade[vistas] >= 0) {
                contadas++;
                if (c == p->verdade[vistas]) acertos++;
            }
        }
        vistas++;
    }
    const uint64_t t_parede = agora() - t0;
    const uint32_t tq1 = rd(REG_TIQUE);
    const uint32_t perdidas = rd(REG_PERDIDA);

    const double us = (double)t_parede * us_por_tique;
    printf("\n=== RESULTADO ===\n");
    // as ultimas decisoes dependem de amostras que estao ALEM do fim do sinal:
    // elas nao existem, nao foram perdidas. Quem prova isso e' a sequencia, que
    // e' contigua; o que ficaria de fora seria um salto, e nao um deficit no
    // fim. Por isso o rodape so' alarma se houve salto
    printf("  decisoes: %u colhidas de %u esperadas%s\n", vistas, n_dec,
           (vistas == n_dec) ? ""
           : (saltos ? "   <-- FALTOU"
                     : "   (as ultimas dependem de amostras alem do sinal)"));
    printf("  decisoes perdidas por fila cheia: %u%s\n", perdidas,
           perdidas ? "   <-- a medida nao vale" : "");
    printf("  sequencia: %u saltos, %u decisoes perdidas na colheita%s",
           saltos, perdidas_seq, saltos ? "" : "\n");
    if (saltos)
        printf(", primeiro em %d   <-- desloca tudo a partir dai\n",
               (int)primeiro_salto);
    printf("  concorda com o modelo: %.2f%%  (%u/%u)\n",
           100.0 * concorda / (vistas ? vistas : 1), concorda, vistas);
    if (primeira_dif >= 0)
        printf("  DIVERGENCIA: primeira em %d\n", (int)primeira_dif);
    printf("  acuracia            : %.2f%%  (%u/%u janelas inteiras)\n",
           100.0 * acertos / (contadas ? contadas : 1), acertos, contadas);
    printf("  estouro de acumulador: %u\n", (rd(REG_STATUS) & ST_TRUNC) >> 8);

    printf("\n=== TEMPO ===\n");
    printf("  relogio da PL       : %.2f MHz aferido  (a sintese fechou em %u)\n",
           pl_mhz, PL_CLK_MHZ);
    printf("  ciclos da PL        : %lu para %llu amostras -> %.2f ciclos por "
           "amostra\n", (unsigned long)(tq1 - tq0),
           (unsigned long long)enviadas,
           (double)(tq1 - tq0) / (double)(enviadas ? enviadas : 1));
    printf("  sistema completo    : %.1f ms para %llu amostras\n",
           us / 1000.0, (unsigned long long)enviadas);
    printf("  vazao               : %.0f amostras/s, %.0f decisoes/s\n",
           1e6 * (double)enviadas / us, 1e6 * (double)vistas / us);
    printf("  vazao da PL         : %.0f amostras/s  (so' o circuito)\n",
           pl_mhz * 1e6 / ((double)(tq1 - tq0) / (double)(enviadas ? enviadas : 1)));

    // o intervalo entre decisoes e' a promessa da arquitetura. Medido pelo ARM
    // seria o Linux; estes contadores sao da PL, no ciclo em que a decisao sai
    {
        const uint32_t iv_min = rd(REG_IV_MIN), iv_max = rd(REG_IV_MAX);
        const uint32_t iv_n = rd(REG_IV_N), fome = rd(REG_FOME);
        printf("\n=== DETERMINISMO (contado na PL) ===\n");
        printf("  intervalo entre decisoes: %lu..%lu ciclos, jitter %lu  "
               "(%lu intervalos)\n", (unsigned long)iv_min,
               (unsigned long)iv_max, (unsigned long)(iv_max - iv_min),
               (unsigned long)iv_n);
        printf("  ciclos sem amostra para consumir: %lu de %lu (%.4f%%)"
               "  <- alimentador, nao circuito\n", (unsigned long)fome,
               (unsigned long)(tq1 - tq0),
               100.0 * (double)fome / (double)((tq1 - tq0) ? (tq1 - tq0) : 1));
        if (iv_max != iv_min && !fome)
            printf("  ATENCAO: intervalo variou sem fome de dado - a garantia "
                   "de taxa nao vale\n");
    }
    if (t_sw) {
        const double sw_am_s = 1e6 * n_sw / ((double)t_sw * us_por_tique);
        const double pl_am_s = pl_mhz * 1e6
            / ((double)(tq1 - tq0) / (double)(enviadas ? enviadas : 1));
        printf("\n=== CONTRA O PROCESSADOR DA MESMA PLACA (fluxo x fluxo) ===\n");
        printf("  ARM em fluxo        : %.0f amostras/s\n", sw_am_s);
        printf("  ganho no computo    : %.1fx\n", pl_am_s / sw_am_s);
    }
    return acertos;
}

// a PL tem jitter zero por construcao; a incerteza vem do orquestrador, que
// roda em Linux e pode ser preemptado. Prioridade de tempo real, memoria
// travada e afinidade de nucleo reduzem isso. Falhar nao e' fatal, mas entra
// no relatorio: jitter medido sem tempo real nao vale como garantia.
static void tempo_real(void)
{
    struct sched_param sp;
    cpu_set_t cpus;
    int ok_prio, ok_mem, ok_cpu;

    sp.sched_priority = 80;
    ok_prio = (sched_setscheduler(0, SCHED_FIFO, &sp) == 0);
    ok_mem  = (mlockall(MCL_CURRENT | MCL_FUTURE) == 0);
    CPU_ZERO(&cpus);
    CPU_SET(1, &cpus);
    ok_cpu  = (sched_setaffinity(0, sizeof(cpus), &cpus) == 0);

    printf("  tempo real          : prioridade %s, memoria %s, nucleo %s\n",
           ok_prio ? "SCHED_FIFO 80" : "NAO (sem privilegio)",
           ok_mem  ? "travada"       : "NAO travada",
           ok_cpu  ? "1 dedicado"    : "NAO fixado");
    if (!ok_prio || !ok_mem || !ok_cpu)
        printf("  AVISO: sem tempo real completo o jitter medido e do Linux, "
               "nao do sistema\n");
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// minimo e maximo nao caracterizam um prazo: um estouro isolado em 10 mil
// janelas e' diferente de jitter sistematico, e quem decide e' a cauda
static void percentis(const char *rot, uint64_t *v, uint32_t n, double us_tick,
                      double prazo_us)
{
    uint32_t i, perdidas = 0;
    if (!n) return;
    qsort(v, n, sizeof(uint64_t), cmp_u64);
    printf("  %s: p50 %.2f  p99 %.2f  p99.9 %.2f  max %.2f us\n", rot,
           (double)v[(n * 50) / 100] * us_tick,
           (double)v[(n * 99) / 100] * us_tick,
           (double)v[(n * 999) / 1000] * us_tick,
           (double)v[n - 1] * us_tick);
    if (prazo_us > 0.0) {
        for (i = 0; i < n; i++)
            if ((double)v[i] * us_tick > prazo_us) perdidas++;
        printf("  fora do prazo %s: %u de %u janelas (%.4f%%), prazo %.1f us\n",
               rot, perdidas, n, 100.0 * perdidas / n, prazo_us);
    }
}

int main(int argc, char **argv)
{
    // o script passa "" no lugar do caminho quando so' quer informar o prazo
    // no segundo argumento
    const char *caminho = (argc > 1 && argv[1][0]) ? argv[1]
                                                   : CAMINHO_PACOTE;
    if (argc > 2) g_prazo_us = atof(argv[2]);

    extern volatile uint32_t *mapeia(uint32_t base, uint32_t tam);
    acel = mapeia(ACEL_BASE, 0x1000);
    dma  = mapeia(DMA_BASE,  0x100);
    buf  = mapeia(BUF_DMA_FIS, BUF_DMA_TAM);
    gtc  = mapeia(GTC_BASE,  0x100);
    gtc_inicia();
    afere_tique();

    if (dma_inicia()) {
        printf("ERRO: o DMA nao saiu do repouso (dmasr=%08lx)\n",
               (unsigned long)drd(MM2S_DMASR));
        return 1;
    }

    pacote_t pac;
    if (pacote_abre(caminho, &pac)) return 1;

    // depois do cabecalho de proposito: o fluxo recorta a saida a partir dele,
    // e o relatorio precisa da confirmacao de tempo real junto das medidas.
    // Impresso antes, o campo sumia calado do metricas.json
    printf("acelerador CNN 1D - Arty Z7-20, PL a %u MHz\n", PL_CLK_MHZ);
    tempo_real();
    printf("rodada: %s  pesos: %s\n", RUN_ID, PESOS_SHA);
    printf("contador do ARM: %llu tiques em %lld ns -> %.4f ns por tique "
           "(%.1f MHz; a constante ARM_CLK_MHZ=%u supunha %.1f MHz)\n",
           (unsigned long long)afere_tiques, afere_ns,
           us_por_tique * 1000.0, 1.0 / us_por_tique,
           ARM_CLK_MHZ, ARM_CLK_MHZ / 2.0);
    if ((pac.versao == PACOTE_FLUXO) != (MODO_FLUXO != 0)) {
        printf("ERRO: pacote versao %u e imagem %s\n", pac.versao,
               MODO_FLUXO ? "de fluxo continuo" : "por janela");
        return 1;
    }
    if (pac.versao == PACOTE_FLUXO)
        printf("conjunto de teste: %u amostras em fluxo continuo, %u decisoes\n",
               pac.n_vec, pac.in_len);
    else
        printf("conjunto de teste: %u janelas de %u amostras, ja' em int8\n",
               pac.n_vec, pac.in_len);
    // a escala vem do treino e vai gravada nos dois lados; se divergirem, a
    // janela do pacote nao e' a que a rede viu e a acuracia medida nao vale
    if (fabs(pac.escala - INPUT_SCALE) > 1e-12 * fabs(INPUT_SCALE))
        printf("  AVISO: escala do pacote %.9e != da imagem %.9e\n",
               pac.escala, (double)INPUT_SCALE);
    // in_len muda de significado entre os dois formatos: em v5 e' o tamanho da
    // janela, em v6 e' o numero de decisoes. So' o primeiro se compara com
    // IN_LEN da imagem
    if (pac.versao == PACOTE_JANELA && pac.in_len != IN_LEN) {
        printf("ERRO: pacote com janela de %u, imagem construida para %u\n",
               pac.in_len, IN_LEN);
        return 1;
    }

    uint64_t t_cfg = carrega_tudo();
    printf("config dos multiplicadores: %u constantes, %.3f ms\n",
           N_MULT_TOTAL, (double)t_cfg * us_por_tique / 1000.0);

    const uint32_t n_conf = 2000;
    uint32_t a1, a2;

    if (pac.versao == PACOTE_FLUXO) {
        uint64_t t_carga = carrega_pelo_barramento((1u << N_CAMADAS) - 1u);
        printf("  carga dos %u pesos + %u bias: %.3f ms\n",
               N_W_TOTAL, N_B_TOTAL, (double)t_carga * us_por_tique / 1000.0);
        a1 = campanha_fluxo(&pac, 0);
        a2 = a1;
    } else if (PESO_EMBARCADO) {
        a1 = campanha("pesos embarcados na sintese", &pac, 1, 0);
        carrega_pelo_barramento((1u << N_CAMADAS) - 1u);
        a2 = campanha("depois de escrever pelo barramento", &pac, 0, n_conf);
        printf("\n=== DE ONDE VEM O PESO ===\n");
        printf("  embarcado, conjunto inteiro : %.2f%%  (%u/%u)\n",
               100.0 * a1 / pac.n_vec, a1, pac.n_vec);
        printf("  apos escrita, %u janelas    : %.2f%%\n",
               n_conf, 100.0 * a2 / n_conf);
    } else {
        uint64_t t_carga = carrega_pelo_barramento((1u << N_CAMADAS) - 1u);
        a1 = campanha("peso escrito pelo barramento", &pac, 1, 0);
        a2 = a1;
        printf("\n=== DE ONDE VEM O PESO ===\n");
        printf("  pelo barramento, conjunto inteiro : %.2f%%  (%u/%u)\n",
               100.0 * a1 / pac.n_vec, a1, pac.n_vec);
        printf("  carga dos %u pesos + %u bias      : %.3f ms\n",
               N_W_TOTAL, N_B_TOTAL,
               (double)t_carga * us_por_tique / 1000.0);
    }

    printf("\n=== FIM DO FLUXO ===\n");
    printf("  LEDs: tres piscadas e volta ao hardware\n");
    fflush(stdout);
    leds_pisca(3);

    pacote_fecha(&pac);
    return (a1 == 0) ? 1 : 0;
}
