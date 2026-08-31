"""deriva a microarquitetura e o rtl a partir da especificacao da rede."""

from __future__ import annotations

import argparse
import json
import itertools
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

ENTRADA   = 1024
NCLASSES  = 4
DSP_TOTAL  = 220
BRAM_TOTAL = 140
LUT_TOTAL  = 53200
FF_TOTAL   = 106400
DIR_MEM    = str(Path(__file__).resolve().parent.parent / "results/gen")

LUT_POR_MAC = 117
FF_POR_MAC  = 54

FRACAO_LUT  = 0.50

B18_MIN_BITS = 512
B18_MIN_PROF = 64
BRAM_RESERVA = 1

def dsp_por_via(pk: int) -> int:
    return pk if pk == 1 else pk + 1

def prof_fila(pox: int) -> int:
    return max(4, 2 * pox)

def prof_b18(larg: int) -> int:
    for w, d in ((1, 16384), (2, 8192), (4, 4096), (9, 2048)):
        if larg <= w:
            return d
    return 1024

def b18(prof: int, larg: int) -> int:
    if prof < B18_MIN_PROF or prof * larg < B18_MIN_BITS:
        return 0
    col = max(1, math.ceil(larg / 18))
    return col * math.ceil(prof / prof_b18(math.ceil(larg / col)))

def memorias(camadas: list, entrada: int, head: str,
             fc_vias: int) -> list[tuple[str, int]]:
    cs = [c if isinstance(c, dict) else asdict(c) for c in camadas]
    itens = [("entrada", b18(entrada, 8))]
    for i, c in enumerate(cs):
        og = c["nof"] // c["pof"]
        itens.append((f"{c['nome']}.peso",
                      c["pof"] * c["pk"] * b18(og * c["nif"] * c["k"], 8)))
        itens.append((f"{c['nome']}.bias", c["pof"] * b18(og, c["acc_w"])))
        if i:
            itens.append((f"{c['nome']}.fila",
                          b18(prof_fila(c["pox"]), c["nif"] * 8)))
    u = cs[-1]
    nflat = u["nof"] * (u["comp"] // u["pool"])
    npeso = u["nof"] if head == "gap" else nflat
    itens.append(("densa", NCLASSES * fc_vias * b18(npeso // fc_vias, 8)))
    return [(k, v) for k, v in itens if v]

def blocos_bram(camadas: list, entrada: int, head: str, fc_vias: int) -> int:
    n18 = sum(v for _, v in memorias(camadas, entrada, head, fc_vias))
    return math.ceil(n18 / 2)

def p2_abaixo(n: int) -> int:
    return 1 << (max(n, 1).bit_length() - 1)

def p2_acima(n: int) -> int:
    n = max(n, 1)
    return 1 << (n - 1).bit_length() if n > 1 else 1

@dataclass
class Camada:
    nome: str
    nif: int
    nof: int
    k: int
    comp: int
    pool: int
    pool_avg: int
    acc_w: int
    pox: int
    pk: int
    pof: int
    nrqu: int
    ndsp: int
    rq_w: int
    dsp: int
    ciclos: int
    pesos: int
    bias: int

@dataclass
class Plano:
    nome: str
    num_layers: int
    num_filters_first: int
    kernel_size: int
    pool_type: str
    head: str
    camadas: list
    fc_nflat: int
    fc_nstream: int
    fc_acc_w: int
    fc_vias: int
    fc_dsp: int
    fc_bram: int
    dsp_total: int
    mac_logica: int
    lut_mac: int
    ii: int
    macs: int
    eficiencia_dsp: float
    pesos_total: int
    bram: int
    cabe: bool
    motivo: str
    entrada: int = ENTRADA
    escrita_de_peso: bool = True
    rq_por_canal: bool = False
    dsp_blocos: int = 0
    fluxo: bool = False
    dec_intervalo: int = 0
    dec_por_pooling: int = 0
    dec_por_dobramento: int = 0
    prazo: int = 0
    dsp_reserva: int = 0
    ciclos_ociosos: int = 0
    bram_reserva: int = 0
    frente: list = None

def largura_acumulador(nif: int, k: int) -> int:
    return math.ceil(math.log2(nif * k * 127 * 127)) + 1

def geometria(L: int, F: int, K: int, pool_type: str, head: str = "flatten",
              entrada: int = ENTRADA):
    pool = 2 if pool_type in ("max", "avg") else 1
    cams, nif, comp = [], 1, entrada
    for i in range(L):
        nof = F * (2 ** i)
        cams.append({"nome": f"conv{i+1}", "nif": nif, "nof": nof,
                     "k": K, "comp": comp, "pool": pool,
                     "pool_avg": 1 if pool_type == "avg" else 0,
                     "rq_w": largura_acumulador(nif, K),
                     "acc_w": largura_acumulador(nif, K) + 1})
        nif = nof
        comp //= pool
    return cams, (nif if head == "gap" else nif * comp)

def fluxo_final(cams: list) -> int:
    return cams[-1]["nof"] * (cams[-1]["comp"] // cams[-1]["pool"])

def vias_saida(c: dict, p: int, pk: int = 1, pof: int | None = None) -> int:
    pof     = c["nof"] if pof is None else pof
    grp     = (c["nif"] * c["k"]) // pk
    pool_in = min(p, c["pool"])
    nq      = p // pool_in
    gpp     = c["pool"] // pool_in
    folga   = grp if pof < c["nof"] else gpp * grp
    preciso = math.ceil(pof * nq / folga)
    return min(pof, p2_acima(preciso))

def orcamento_efetivo(dsp: int) -> int:
    return dsp + int(LUT_TOTAL * FRACAO_LUT / LUT_POR_MAC)

def pox_maximo(c: dict, pk: int = 1) -> int:
    grp = (c["nif"] * c["k"]) // pk
    p = c["comp"]
    if grp > 1:
        p = min(p, p2_abaixo(grp - 1))
    if c["pool"] > 1 and 1 < p < c["pool"]:
        p = 1
    return max(1, p2_abaixo(p))

def ciclos_camada(c: dict, pox: int, pof: int, pk: int = 1) -> int:
    return (c["comp"] // pox) * (c["nof"] // pof) * ((c["nif"] * c["k"]) // pk)

def legal(c: dict, pox: int, pof: int, pk: int) -> bool:
    if pox < 1 or pof < 1 or pk < 1:
        return False
    if c["comp"] % pox or c["nof"] % pof or c["k"] % pk:
        return False
    if pof & (pof - 1):
        return False
    if pox > pox_maximo(c, pk):
        return False
    pool_in = min(pox, c["pool"])
    if pox % pool_in or c["pool"] % pool_in:
        return False
    grp = (c["nif"] * c["k"]) // pk
    nrqu = vias_saida(c, pox, pk, pof)
    if pof % nrqu:
        return False
    folga = grp if pof < c["nof"] else (c["pool"] // pool_in) * grp
    return pof * (pox // pool_in) <= nrqu * folga

def b18_camada(c: dict, pox: int, pof: int, pk: int) -> int:
    og = c["nof"] // pof
    fila = 0 if c["nome"] == "conv1" else b18(prof_fila(pox), c["nif"] * 8)
    return (pof * pk * b18(og * c["nif"] * c["k"], 8)
            + pof * b18(og, c["acc_w"])
            + fila)

def ff_camada(c: dict, pox: int, pof: int, pk: int) -> int:
    pool_in = min(pox, c["pool"])
    nq = pox // pool_in
    jan = pox + c["k"] - 1
    return (2 * c["nif"] * jan * 8
            + (c["nof"] + pof) * nq * c["acc_w"])

def configs_camada(c: dict) -> list[dict]:
    saida = []
    for pk in [q for q in (1, 2, 4, 8, 16) if q <= c["k"] and c["k"] % q == 0]:
        pox = 1
        while pox <= c["comp"]:
            pof = 1
            while pof <= c["nof"]:
                if legal(c, pox, pof, pk):
                    vias = vias_saida(c, pox, pk, pof)
                    saida.append({
                        "pox": pox, "pof": pof, "pk": pk, "nrqu": vias,
                        "ciclos": ciclos_camada(c, pox, pof, pk),
                        "mult": pof * pox * dsp_por_via(pk) + vias,
                        "b18": b18_camada(c, pox, pof, pk),
                        "ff": ff_camada(c, pox, pof, pk)})
                pof *= 2
            pox *= 2
    return saida

def com_densa(d: dict, nstream: int, nflat: int) -> dict:
    vias = d["nrqu"]
    e = dict(d)
    e["ciclos"] = max(d["ciclos"], nstream // vias)
    e["mult"] = d["mult"] + NCLASSES * vias
    e["b18"] = d["b18"] + NCLASSES * vias * b18(nflat // vias, 8)
    e["ff"] = d["ff"] + NCLASSES * 32
    return e

RECURSOS = ("mult", "b18", "ff")

def nao_dominados(cands: list[dict]) -> list[dict]:
    frente = []
    for d in sorted(cands, key=lambda x: tuple(x[r] for r in RECURSOS)):
        if not any(all(o[r] <= d[r] for r in RECURSOS)
                   and any(o[r] < d[r] for r in RECURSOS) for o in frente):
            frente.append(d)
    return frente

def frente_pareto(cams: list, nflat: int, nstream: int, entrada: int,
                  orcamento: int, fluxo: bool = False) -> list[dict]:
    teto_mult = orcamento_efetivo(orcamento)
    teto_b18 = 2 * (BRAM_TOTAL - BRAM_RESERVA) - b18(entrada, 8)
    teto_ff = FF_TOTAL

    por_camada = [configs_camada(c) for c in cams]
    if not all(por_camada):
        return []
    por_camada[-1] = [com_densa(d, nstream, nflat) for d in por_camada[-1]]

    def pressao(b: int, f: int) -> float:
        return max(b / teto_b18, f / teto_ff)

    bruta = []
    for alvo in sorted({d["ciclos"] for lista in por_camada for d in lista}):
        opcoes = []
        for lista in por_camada:
            viaveis = [d for d in lista if d["ciclos"] <= alvo]
            if not viaveis:
                opcoes = None
                break
            opcoes.append(nao_dominados(viaveis))
        if opcoes is None:
            continue
        melhor = None
        for combo in itertools.product(*opcoes):
            mult = sum(d["mult"] for d in combo)
            bram = sum(d["b18"] for d in combo)
            ff = sum(d["ff"] for d in combo)
            if mult > teto_mult or bram > teto_b18 or ff > teto_ff:
                continue
            ult = combo[-1]
            nq = ult["pox"] // min(ult["pox"], cams[-1]["pool"])
            chave = ((mult, nq, pressao(bram, ff)) if fluxo
                     else (mult, pressao(bram, ff)))
            if melhor is None or chave < melhor[0]:
                melhor = (chave, mult, bram, ff, combo)
        if melhor is not None:
            bruta.append({"ii": max(d["ciclos"] for d in melhor[4]),
                          "mult": melhor[1], "b18": melhor[2], "ff": melhor[3],
                          "escolha": list(melhor[4])})

    frente, visto = [], None
    for q in bruta:
        if visto is None or q["mult"] < visto:
            frente.append(q)
            visto = q["mult"]
    return frente

EFIC_MINIMA = 0.60

def aloca(cams: list, orcamento: int, nflat: int, nstream: int, entrada: int,
          prazo: int | None = None,
          fluxo: bool = False) -> tuple[dict | None, list, str]:
    frente = frente_pareto(cams, nflat, nstream, entrada, orcamento, fluxo)
    if not frente:
        return None, [], ("nenhuma reparticao desta topologia cabe na peca "
                          "(multiplicadores ou blocos de memoria)")
    so_dsp = [q for q in frente if q["mult"] <= orcamento]
    melhor = (so_dsp or frente)[0]
    if prazo is not None and melhor["ii"] > prazo:
        return None, frente, (
            f"nao cumpre o prazo nem no maximo paralelismo: {melhor['ii']} "
            f"ciclos contra os {prazo} disponiveis")
    return melhor, frente, ""

def reparte(cams: list, escolha: list, orcamento: int) -> list:
    lanes = [d["pof"] * d["pox"] * dsp_por_via(d["pk"]) for d in escolha]
    fixo = sum(d["mult"] for d in escolha) - sum(lanes)
    livre = max(0, orcamento - fixo)
    total = sum(lanes)
    if total <= livre:
        return lanes
    return [min(l, int(l * livre / total)) for l in lanes]

def planeja(L: int, F: int, K: int, pool_type: str, head: str = "flatten",
            orcamento: int = DSP_TOTAL, escrita_de_peso: bool = True,
            sufixo: str = "",
            rq_por_canal: bool = False, prazo: int | None = None,
            entrada: int = ENTRADA, fluxo: bool = False) -> Plano:
    cams, nflat = geometria(L, F, K, pool_type, head, entrada)
    nstream = fluxo_final(cams)
    if fluxo and prazo is not None:
        prazo = prazo * entrada
    escolha, frente, motivo = aloca(cams, orcamento, nflat, nstream, entrada,
                                    prazo, fluxo)
    if fluxo and head != "gap":
        escolha = None
        motivo = ("fluxo continuo exige cabeca gap: com flatten o peso da densa "
                  "depende da posicao na janela e a saida nao e' incremental")
    if escolha is None:
        minima = []
        for c in cams:
            vias = vias_saida(c, 1, 1, 1)
            minima.append({"pox": 1, "pof": 1, "pk": 1, "nrqu": vias,
                           "ciclos": ciclos_camada(c, 1, 1, 1),
                           "mult": 1 + vias, "b18": b18_camada(c, 1, 1, 1),
                           "ff": ff_camada(c, 1, 1, 1)})
        escolha = {"escolha": minima}
    ndsp = reparte(cams, escolha["escolha"], orcamento)

    camadas, macs, pesos_tot = [], 0, 0
    for c, d, nd in zip(cams, escolha["escolha"], ndsp):
        m = c["nof"] * c["nif"] * c["k"] * c["comp"]
        pesos = c["nof"] * c["nif"] * c["k"]
        p, f, pk, vias = d["pox"], d["pof"], d["pk"], d["nrqu"]
        camadas.append(Camada(
            nome=c["nome"], nif=c["nif"], nof=c["nof"], k=c["k"],
            comp=c["comp"], pool=c["pool"],
            pool_avg=c["pool_avg"],
            acc_w=c["acc_w"],
            pox=p, pk=pk, pof=f,
            nrqu=vias, ndsp=nd,
            rq_w=c["rq_w"],
            dsp=f * p * dsp_por_via(pk) + vias,
            ciclos=ciclos_camada(c, p, f, pk),
            pesos=pesos, bias=c["nof"],
        ))
        macs += m
        pesos_tot += pesos

    fc_pesos = nflat * NCLASSES
    macs += nstream * NCLASSES
    pesos_tot += fc_pesos

    fc_bram = NCLASSES if nflat > 512 else 0
    fc_acc_w = math.ceil(math.log2(nstream * 127 * 127)) + 1

    ii_conv = max(c.ciclos for c in camadas)
    fc_vias = camadas[-1].nrqu
    fc_ciclos = nstream // fc_vias
    ii = max(ii_conv, fc_ciclos)

    dsp_total = sum(c.dsp for c in camadas) + NCLASSES * fc_vias
    lanes = sum(c.pof * c.pox * c.pk for c in camadas) + NCLASSES * fc_vias
    efic = macs / (lanes * ii) if lanes and ii else 0.0

    em_logica = sum(c.pof * c.pox * dsp_por_via(c.pk) for c in camadas) - sum(ndsp)
    lut_mac   = em_logica * LUT_POR_MAC
    if not motivo and fc_acc_w > 32:
        motivo = (f"acumulador da densa precisa de {fc_acc_w} bits e o caminho "
                  f"de dados tem 32 ({nstream} termos no fluxo)")
    bram = blocos_bram(camadas, entrada, head, fc_vias)
    if not motivo and bram > BRAM_TOTAL - BRAM_RESERVA:
        motivo = (f"precisa de {bram} blocos BRAM e sobram "
                  f"{BRAM_TOTAL - BRAM_RESERVA} ({BRAM_RESERVA} reservados "
                  f"para DMA e interconexao)")
    if not motivo and prazo is not None and ii > prazo:
        motivo = (f"nao cumpre o prazo: {ii} ciclos contra os {prazo} "
                  f"disponiveis por janela")

    dsp_blocos = dsp_total - em_logica
    cabe = (not motivo) and lut_mac <= LUT_TOTAL * FRACAO_LUT \
        and dsp_blocos <= orcamento
    if cabe and efic < EFIC_MINIMA:
        cabe = False
        motivo = (f"circuito serializado demais: {efic:.1%} de eficiencia "
                  f"aritmetica, abaixo do piso de {EFIC_MINIMA:.0%}. "
                  f"O custo fixo domina quando o arranjo de MAC encolhe.")
    if not cabe and not motivo:
        motivo = (f"{dsp_blocos} blocos DSP > {orcamento}"
                  if dsp_blocos > orcamento else
                  f"{em_logica} MAC em logica = {lut_mac} LUT, acima do teto")

    u = camadas[-1]
    _npos = u.comp // u.pool
    _pool = entrada // _npos
    _npg  = u.pox // min(u.pox, u.pool)

    return Plano(
        dec_por_pooling=_pool,
        dec_por_dobramento=_npg,
        dec_intervalo=(_pool * _npg) if fluxo else entrada,
        nome=(f"L{L}_F{F:02d}_K{K}_P{pool_type}_H{head}"
              + ("" if entrada == ENTRADA else f"_W{entrada}")
              + ("_fluxo" if fluxo else "") + sufixo),
        fluxo=fluxo,
        entrada=entrada,
        dsp_blocos=dsp_blocos,
        prazo=prazo or 0,
        dsp_reserva=max(0, orcamento - dsp_blocos),
        ciclos_ociosos=max(0, prazo - ii) if prazo else 0,
        bram_reserva=max(0, BRAM_TOTAL - BRAM_RESERVA - bram),
        escrita_de_peso=escrita_de_peso,
        rq_por_canal=rq_por_canal,
        num_layers=L, num_filters_first=F, kernel_size=K, pool_type=pool_type,
        head=head,
        camadas=[asdict(c) for c in camadas],
        fc_nflat=nflat, fc_nstream=nstream, fc_acc_w=fc_acc_w,
        fc_vias=fc_vias,
        fc_dsp=NCLASSES * fc_vias,
        fc_bram=fc_bram,
        dsp_total=dsp_total, mac_logica=em_logica, lut_mac=lut_mac,
        ii=ii, macs=macs,
        eficiencia_dsp=efic, pesos_total=pesos_tot, bram=bram,
        cabe=cabe, motivo=motivo,
        frente=[{"ii": q["ii"], "dsp": q["mult"], "ff": q["ff"],
                 "bram": math.ceil((q["b18"] + b18(entrada, 8)) / 2)}
                for q in frente],
    )

CAB = ("// gerado por ferramentas/gerador.py para {nome}: {L} camadas, "
       "{F} filtros, kernel {K}, pool {pool}, {ii} ciclos, {dsp} DSP\n")

def emite_rtl(p: Plano) -> str:
    L = p.num_layers
    t = [CAB.format(nome=p.nome, L=L, F=p.num_filters_first, K=p.kernel_size,
                    pool=p.pool_type, ii=p.ii, dsp=p.dsp_total,
                    efic=p.eficiencia_dsp)]
    t.append("`default_nettype none\n")
    t.append("module acelerador_gen #(\n"
             "    parameter NCAM = 8,\n    parameter NCLS = 4,\n"
             f'    parameter MEMDIR = "{DIR_MEM}/{p.nome}/mem",\n'
             f"    parameter WR_PESO = {1 if p.escrita_de_peso else 0},\n"
             "    parameter EST_CARGA = 3\n)(")
    t.append("""    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [3:0]               ld_sel,
    input  wire                     ld_w_en,
    input  wire                     ld_w_valid,
    input  wire signed [7:0]        ld_w_data,
    input  wire                     ld_b_en,
    input  wire                     ld_b_valid,
    input  wire signed [31:0]       ld_b_data,
    input  wire                     ld_m_en,
    input  wire                     ld_m_valid,
    input  wire signed [17:0]       ld_m_data,
    input  wire                     start,
    output wire                     busy,
    output wire                     done,
    output wire [NCAM-1:0]          trunc_err,
    // a janela chega ja' normalizada e em int8; a PL so' requantiza entre
    // camadas. A normalizacao por janela e' feita no preparo dos dados, antes
    // do treino, e a rede foi treinada exatamente sobre este tensor.
    input  wire                     in_valid,
    input  wire signed [7:0]        in_data,
    output wire                     in_ready,
    output wire [NCLS*32-1:0]       logits,
    output wire [3:0]               classe,
    output wire                     classe_valid,
    input  wire                     ativ_reinicia,
    input  wire                     ativ_avanca,
    output wire [31:0]              ativ_data
);
""")
    t.append(f"    localparam NCONV = {L};\n")
    t.append(f"    wire [NCONV-1:0] trunc_conv;")
    t.append(f"    reg  [NCONV-1:0] estouro;")
    t.append("    initial estouro = 0;")
    t.append("    wire [$clog2(NCLS)-1:0] classe_fc;")
    t.append("    assign classe = {{{{(4-$clog2(NCLS)){{1'b0}}}}, classe_fc}};")
    if p.fluxo:
        t.append(f"    assign trunc_err = {{{{(NCAM-NCONV-1){{1'b0}}}}, "
                 f"desvio_cabeca, trunc_conv | estouro}};")
    else:
        t.append(f"    assign trunc_err = {{{{(NCAM-NCONV){{1'b0}}}}, "
                 f"trunc_conv | estouro}};")
    t.append("")

    for i, c in enumerate(p.camadas):
        n = i + 1
        t.append(f"    wire c{n}_ov, c{n}_busy, c{n}_ready;")
        t.append(f"    wire [{c['nrqu']*8-1}:0] c{n}_od;")
        t.append(f"    wire c{n}_done;")
        t.append(f"    wire [{max(1, (c['nof']-1).bit_length())-1}:0] c{n}_oc;")
        opos = c["comp"] // c["pool"]
        t.append(f"    wire [{max(1,(opos-1).bit_length())-1}:0] c{n}_op;")
    t.append("")

    t.append("""    // o start chega do envelope so' depois que a janela inteira esta' na
    // fila de entrada, entao o motor nunca espera dado e o contador de ciclos
    // mede computo puro. E' o que mantem o jitter da PL em zero.
    wire arranca = start;
    assign in_ready = c1_ready;
""")

    for i, c in enumerate(p.camadas):
        n = i + 1
        t.append(f"    // ---------------- conv{n}: {c['nif']}x{c['nof']} "
                 f"canais, kernel {c['k']}, {c['comp']} posicoes, "
                 f"POX={c['pox']} POF={c['pof']} PK={c['pk']} "
                 f"({c['dsp']} DSP, {c['ciclos']} ciclos) ----------------")
        ent = ("in_valid), .in_data(in_data), .in_ready(c1_ready" if i == 0
               else f"!f{i}_vazia), .in_data(f{i}_rd_d), .in_ready(c{n}_ready")
        ultima = (i == len(p.camadas) - 1)
        saida_pronta = "1'b1" if ultima else f"!f{n}_cheia"
        if not ultima:
            t.append(f"    wire f{n}_cheia;\n")
        t.append(f"""    wire p{n}_w_en, p{n}_w_v, p{n}_b_en, p{n}_b_v, p{n}_m_en, p{n}_m_v;
    wire signed [7:0]  p{n}_w_d;
    wire signed [31:0] p{n}_b_d;
    wire signed [17:0] p{n}_m_d;
    carga_pipe #(.EST(EST_CARGA)) u_carga_conv{n} (
        .clk(clk),
        .i_w_en(ld_w_en && ld_sel == {i}), .i_w_valid(ld_w_valid && ld_sel == {i}),
        .i_w_data(ld_w_data),
        .i_b_en(ld_b_en && ld_sel == {i}), .i_b_valid(ld_b_valid && ld_sel == {i}),
        .i_b_data(ld_b_data),
        .i_m_en(ld_m_en && ld_sel == {i}), .i_m_valid(ld_m_valid && ld_sel == {i}),
        .i_m_data(ld_m_data),
        .o_w_en(p{n}_w_en), .o_w_valid(p{n}_w_v), .o_w_data(p{n}_w_d),
        .o_b_en(p{n}_b_en), .o_b_valid(p{n}_b_v), .o_b_data(p{n}_b_d),
        .o_m_en(p{n}_m_en), .o_m_valid(p{n}_m_v), .o_m_data(p{n}_m_d)
    );

    conv1d_engine #(
        .NIF({c['nif']}), .NOF({c['nof']}), .K({c['k']}), .LEN({c['comp']}),
        .POOL({c['pool']}), .POOL_AVG({c['pool_avg']}),
        .POX({c['pox']}), .POF({c['pof']}), .PK({c['pk']}), .NRQU({c['nrqu']}),
        .NDSP({c['ndsp']}), .DATA_W(8), .ACC_W(32),
        .RQ_W({c['rq_w']}), .MULT_W(18), .SHIFT(16),
        .RQ_POR_CANAL({1 if p.rq_por_canal else 0}), .FLUXO({1 if p.fluxo else 0}),
        .MEM({{MEMDIR, "/c{n}"}}), .WR_PESO(WR_PESO)
    ) u_conv{n} (
        .clk(clk), .rst_n(rst_n),
        .ld_w_en(p{n}_w_en), .ld_w_valid(p{n}_w_v), .ld_w_data(p{n}_w_d),
        .ld_b_en(p{n}_b_en), .ld_b_valid(p{n}_b_v), .ld_b_data(p{n}_b_d),
        .ld_m_en(p{n}_m_en), .ld_m_valid(p{n}_m_v), .ld_m_data(p{n}_m_d),
        .start(arranca), .busy(c{n}_busy), .done(c{n}_done),
        .trunc_err(trunc_conv[{i}]),
        .out_ready({saida_pronta}),
        .in_valid({ent}),
        .out_valid(c{n}_ov), .out_data(c{n}_od), .out_ch(c{n}_oc), .out_pos(c{n}_op),
        .dbg_acc()
    );
""")
        if i < L - 1:
            prox = p.camadas[i + 1]
            n1 = n + 1
            prof = prof_fila(prox["pox"])
            npg_c = c["pox"] // min(c["pox"], c["pool"])
            t.append(f"""    // profundidade pela rajada de quem consome: a c{n+1} puxa
    // {prox['pox']} posicoes por grupo, entao a fila guarda {prof}
    localparam PROF_F{n} = {prof};
    wire f{n}_wr, f{n}_vazia;
    wire [{c['nof']*8-1}:0] f{n}_wd, f{n}_rd_d;
    wire [$clog2(PROF_F{n}+1)-1:0] f{n}_ocup;
    agrupador #(.NCAN({c['nof']}), .DATA_W(8), .NVIA({c['nrqu']}),
                .NPOSG({npg_c}), .NPOS({c['comp'] // c['pool']})) u_agrupa_conv{n} (
        .clk(clk), .rst_n(rst_n),
        .in_valid(c{n}_ov), .in_data(c{n}_od), .in_ch(c{n}_oc),
        .in_pos(c{n}_op),
        .out_valid(f{n}_wr), .out_data(f{n}_wd)
    );
    // a fila nao pode mais transbordar: o motor para quando ela enche. O bit
    // fica como rede de seguranca, e reprovar por ele passou a ser defeito.
    // Sem este bit o transbordo passaria calado, como ja passou antes.
    always @(posedge clk)
        if (!rst_n || start)            estouro[{i}] <= 1'b0;
        else if (f{n}_wr && f{n}_cheia) estouro[{i}] <= 1'b1;
    fifo_sinc #(.W({c['nof']*8}), .PROF(PROF_F{n})) u_fila_conv{n}_conv{n1} (
        .clk(clk), .rst_n(rst_n), .limpa(arranca),
        .wr(f{n}_wr), .wdata(f{n}_wd), .cheia(f{n}_cheia),
        .rd(c{n+1}_ready && !f{n}_vazia), .rdata(f{n}_rd_d),
        .vazia(f{n}_vazia), .ocupacao(f{n}_ocup)
    );
""")

    ult = p.camadas[-1]
    L_ult = len(p.camadas)
    if p.fluxo:
        t.append(f"""    assign ativ_data = 32'd0;

""")
    else:
        t.append(f"""    ativa_buf #(
        .NCH({ult['nof']}), .NPOS({ult['comp'] // ult['pool']}),
        .NVIA({ult['nrqu']}), .DATA_W(8)
    ) u_buf_ativacao (
        .clk(clk), .rst_n(rst_n),
        .in_valid(c{L_ult}_ov), .in_data(c{L_ult}_od),
        .in_ch(c{L_ult}_oc), .in_pos(c{L_ult}_op),
        .rd_reinicia(ativ_reinicia), .rd_avanca(ativ_avanca),
        .rd_data(ativ_data)
    );

""")
    if p.fluxo:
        npg = ult["pox"] // min(ult["pox"], ult["pool"])
        t.append(f"""    wire pf_w_en, pf_w_v, pf_b_en, pf_b_v;
    wire signed [7:0]  pf_w_d;
    wire signed [31:0] pf_b_d;
    carga_pipe #(.EST(EST_CARGA)) u_carga_densa (
        .clk(clk),
        .i_w_en(ld_w_en && ld_sel == {L}), .i_w_valid(ld_w_valid && ld_sel == {L}),
        .i_w_data(ld_w_data),
        .i_b_en(ld_b_en && ld_sel == {L}), .i_b_valid(ld_b_valid && ld_sel == {L}),
        .i_b_data(ld_b_data),
        .i_m_en(1'b0), .i_m_valid(1'b0), .i_m_data(18'sd0),
        .o_w_en(pf_w_en), .o_w_valid(pf_w_v), .o_w_data(pf_w_d),
        .o_b_en(pf_b_en), .o_b_valid(pf_b_v), .o_b_data(pf_b_d),
        .o_m_en(), .o_m_valid(), .o_m_data()
    );

    cabeca_gap_fluxo #(
        .NCLS({NCLASSES}), .NCH({ult['nof']}), .NPOS({ult['comp'] // ult['pool']}),
        .DATA_W(8), .ACC_W(32), .NVIA({ult['nrqu']}), .NPG({npg}),
        .MEM({{MEMDIR, "/fc"}}), .WR_PESO(WR_PESO)
    ) u_densa (
        .clk(clk), .rst_n(rst_n), .limpa(start),
        .ld_w_en(pf_w_en), .ld_w_valid(pf_w_v), .ld_w_data(pf_w_d),
        .ld_b_en(pf_b_en), .ld_b_valid(pf_b_v), .ld_b_data(pf_b_d),
        .in_valid(c{L}_ov), .in_data(c{L}_od), .in_ch(c{L}_oc), .in_pos(c{L}_op),
        .logits(logits), .classe(classe_fc), .classe_valid(classe_valid),
        .desvio(desvio_cabeca)
    );

    // o desvio da cabeca entra no mesmo registrador de erro da truncagem: quem
    // le o estado ja' olha esse campo, e assim nenhum modo de falha fica sem
    // caminho ate' o processador
    wire desvio_cabeca;
    assign done = classe_valid;
    reg correndo;
    always @(posedge clk)
        if (!rst_n)     correndo <= 1'b0;
        else if (start) correndo <= 1'b1;
    assign busy = correndo;

endmodule

`default_nettype wire
""")
        return "\n".join(t)

    t.append(f"""    wire pf_w_en, pf_w_v, pf_b_en, pf_b_v;
    wire signed [7:0]  pf_w_d;
    wire signed [31:0] pf_b_d;
    carga_pipe #(.EST(EST_CARGA)) u_carga_densa (
        .clk(clk),
        .i_w_en(ld_w_en && ld_sel == {L}), .i_w_valid(ld_w_valid && ld_sel == {L}),
        .i_w_data(ld_w_data),
        .i_b_en(ld_b_en && ld_sel == {L}), .i_b_valid(ld_b_valid && ld_sel == {L}),
        .i_b_data(ld_b_data),
        .i_m_en(1'b0), .i_m_valid(1'b0), .i_m_data(18'sd0),
        .o_w_en(pf_w_en), .o_w_valid(pf_w_v), .o_w_data(pf_w_d),
        .o_b_en(pf_b_en), .o_b_valid(pf_b_v), .o_b_data(pf_b_d),
        .o_m_en(), .o_m_valid(), .o_m_data()
    );

    fc_engine #(
        .NCLS({NCLASSES}), .NCH({ult['nof']}), .NPOS({ult['comp'] // ult['pool']}),
        .GAP({1 if p.head == "gap" else 0}),
        .DATA_W(8), .ACC_W(32), .NVIA({ult['nrqu']}), .MEM({{MEMDIR, "/fc"}}),
        .WR_PESO(WR_PESO)
    ) u_densa (
        .clk(clk), .rst_n(rst_n),
        .ld_w_en(pf_w_en), .ld_w_valid(pf_w_v), .ld_w_data(pf_w_d),
        .ld_b_en(pf_b_en), .ld_b_valid(pf_b_v), .ld_b_data(pf_b_d),
        .start(arranca), .busy(fc_busy), .done(done),
        .in_valid(c{L}_ov), .in_data(c{L}_od), .in_ch(c{L}_oc), .in_pos(c{L}_op),
        .logits(logits), .classe(classe_fc), .classe_valid(classe_valid)
    );

    wire fc_busy;
    assign busy = {' || '.join(f'c{i+1}_busy' for i in range(L))} || fc_busy;

endmodule

`default_nettype wire
""")
    return "\n".join(t)

def salva(p: Plano, raiz: Path) -> Path:
    d = raiz / p.nome
    d.mkdir(parents=True, exist_ok=True)
    (d / "plano.json").write_text(json.dumps(asdict(p), indent=2))
    (d / "acelerador_gen.v").write_text(emite_rtl(p))
    tcl = [f"set NOME {p.nome}", f"set II {p.ii}", f"set DSP {p.dsp_total}"]
    for i, c in enumerate(p.camadas):
        tcl.append(f"dict set CAM {i} [list {c['nif']} {c['nof']} {c['k']} "
                   f"{c['comp']} {c['pool']} 16 {c['rq_w']} {c['pox']} "
                   f"{c['nrqu']} {c['ndsp']} {c['pool_avg']} {c['pk']}]")
    (d / "params.tcl").write_text("\n".join(tcl) + "\n")
    return d

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/gen"))
    ap.add_argument("--orcamento", type=int, default=DSP_TOTAL)
    ap.add_argument("--sufixo", default="",
                    help="acrescenta ao nome do plano; dois orcamentos da "
                         "mesma rede sao hardwares diferentes e nao podem "
                         "escrever na mesma pasta")
    ap.add_argument("--taxa", type=float, default=None, metavar="JANELAS/S",
                    help="janelas por segundo a sustentar; o gerador verifica "
                         "o prazo antes da sintese")
    ap.add_argument("--clock", type=float, default=83.3, metavar="MHZ",
                    help="clock previsto da PL, para converter taxa em ciclos")
    ap.add_argument("--rq-por-canal", action="store_true",
                    help="uma constante de requantizacao por canal de saida em "
                         "vez de uma por camada; exige escala de peso por canal "
                         "no modelo e custa uma tabela por motor")
    ap.add_argument("--fluxo", action="store_true",
                    help="deriva o acelerador em fluxo continuo: o motor nunca "
                         "para, a convolucao nao tem fronteira de janela e a "
                         "media global vira soma corrente. Exige cabeca gap")
    ap.add_argument("--sem-escrita-de-peso", dest="escrita_de_peso",
                    action="store_false",
                    help="desliga a porta de escrita de peso; a rede fica fixa "
                         "no bitstream")
    a = ap.parse_args()

    if not a.spec:
        ap.error("informe --spec")
    cfg = yaml.safe_load(a.spec.read_text())["model"]
    p = planeja(cfg["num_layers"], cfg["num_filters_first"],
                cfg["kernel_size"], cfg["pool_type"],
                cfg.get("head", "flatten"), a.orcamento,
                a.escrita_de_peso, a.sufixo, a.rq_por_canal,
                prazo=(int(a.clock * 1e6 / a.taxa) if a.taxa else None),
                entrada=int(cfg.get("input_len", ENTRADA)), fluxo=a.fluxo)
    d = salva(p, a.out)
    print(f"{p.nome}: {p.dsp_total} DSP, II={p.ii}, "
          f"eficiencia={p.eficiencia_dsp:.1%}")
    if a.taxa:
        ciclos_disp = a.clock * 1e6 / a.taxa
        livre = planeja(cfg["num_layers"], cfg["num_filters_first"],
                        cfg["kernel_size"], cfg["pool_type"],
                        cfg.get("head", "flatten"), a.orcamento,
                        a.escrita_de_peso, a.sufixo, a.rq_por_canal, prazo=None,
                        entrada=int(cfg.get("input_len", ENTRADA)),
                        fluxo=a.fluxo)
        ef = p.macs / (p.dsp_total * p.ii)
        ef_livre = livre.macs / (livre.dsp_total * livre.ii)
        print(f"  eficiencia aritmetica: {ef:.3f} MAC por DSP-ciclo")
        if livre.cabe and ef_livre > 1.5 * ef:
            print(f"  AVISO: dimensionado pelo prazo, este ponto usa "
                  f"{p.dsp_total} DSP com {ef:.3f} MAC por DSP-ciclo.")
            print(f"         O ponto mais paralelo que cabe usa "
                  f"{livre.dsp_total} DSP com {ef_livre:.3f} "
                  f"({ef_livre/ef:.1f}x mais eficiente) e {livre.ii} ciclos.")
            print(f"         Sem --taxa o gerador entrega esse. O prazo so' "
                  f"vale a pena quando area ou energia sao o criterio.")
        print(f"  prazo: {a.taxa:.0f} {'amostras' if a.fluxo else 'janelas'}/s "
              f"a {a.clock:.1f} MHz = {ciclos_disp:.0f} ciclos por "
              f"{'amostra' if a.fluxo else 'janela'}")
        gasto = (p.ii / p.entrada) if a.fluxo else p.ii
        print(f"  o circuito leva {gasto:.4g} ciclos por "
              f"{'amostra' if a.fluxo else 'janela'} -> "
              f"{100.0 * gasto / ciclos_disp:.2f}% do prazo"
              + ("" if gasto <= ciclos_disp else "   NAO CUMPRE"))
        print(f"  reserva: {p.dsp_reserva} DSP, {p.ciclos_ociosos} ciclos por "
              f"janela, {p.bram_reserva} blocos BRAM")
    print(f"  requantizacao: "
          + ("uma constante por canal de saida"
             if p.rq_por_canal else "uma constante por camada"))
    for c in p.camadas:
        print(f"  {c['nome']}: NIF={c['nif']:3d} NOF={c['nof']:3d} "
              f"LEN={c['comp']:5d} POX={c['pox']:2d} PK={c['pk']:2d} "
              f"vias={c['nrqu']:2d} "
              f"DSP={c['dsp']:4d} ciclos={c['ciclos']}")
    if p.fluxo:
        print(f"  decisao a cada {p.dec_intervalo} amostras "
              f"({p.dec_por_pooling} do pooling x {p.dec_por_dobramento} do "
              f"dobramento da ultima camada)")
    print(f"  fc: pesos/classe={p.fc_nflat}, fluxo={p.fc_nstream}, "
          f"acumulador de {p.fc_acc_w} bits, "
          f"{p.fc_vias} vias, {p.fc_dsp} DSP, "
          f"{p.fc_bram} BRAM36")
    print(f"em {d}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
