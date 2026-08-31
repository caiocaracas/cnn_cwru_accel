"""gera o cabecalho em c com pesos enderecos e constantes do sistema."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from ferramentas.memorias import digesto

def le_mem(p: Path, bits: int) -> np.ndarray:
    vals = []
    for linha in p.read_text().splitlines():
        linha = linha.split("//")[0].strip()
        if linha:
            vals.append(int(linha, 16))
    a = np.array(vals, dtype=np.int64)
    return np.where(a >= (1 << (bits - 1)), a - (1 << bits), a)

def le_pacote(p: Path):
    dados = p.read_bytes()
    magic, n, w, versao = struct.unpack_from("<IIII", dados, 0)

    if magic != 0x434E4E58 or versao not in (5, 6):
        raise ValueError(f"pacote {magic:#x} versao {versao}, esperado "
                         f"0x434E4E58 versao 5 ou 6 - refaca a quantizacao")
    (escala,) = struct.unpack_from("<d", dados, 16)
    o = 24
    if versao == 6:
        jan = np.frombuffer(dados, np.int8, n, o).reshape(1, n)
        cls = np.frombuffer(dados, "<i4", w, o + n)
        return escala, jan, cls
    jan = np.frombuffer(dados, np.int8, n * w, o).reshape(n, w); o += n * w
    cls = np.frombuffer(dados, "<i4", n, o)
    return escala, jan, cls

def vetor_c(nome: str, tipo: str, a: np.ndarray, por_linha: int = 16) -> str:
    txt = [f"static const {tipo} {nome}[{a.size}] = {{"]
    plano = a.reshape(-1)
    for i in range(0, plano.size, por_linha):
        txt.append("    " + ", ".join(str(int(v)) for v in plano[i:i + por_linha]) + ",")
    txt.append("};")
    return "\n".join(txt)

def matriz_c(nome: str, tipo: str, a: np.ndarray, por_linha: int = 12) -> str:
    n, m = a.shape
    txt = [f"static const {tipo} {nome}[{n}][{m}] = {{"]
    for lin in a:
        txt.append("  {")
        for i in range(0, m, por_linha):

            txt.append("    " + ", ".join(repr(float(v)) if tipo == "double"
                                          else str(int(v))
                                          for v in lin[i:i + por_linha]) + ",")
        txt.append("  },")
    txt.append("};")
    return "\n".join(txt)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/hw_data"))
    ap.add_argument("--out", type=Path, default=Path("ps/pesos.h"))
    ap.add_argument("--base", default="0x43C00000")
    ap.add_argument("--base-dma", default="0x40400000",
                    help="registradores do DMA, atribuidos pelo bloco")
    ap.add_argument("--buf-dma", default="0x1FF00000",
                    help="area de DDR reservada no device tree para a janela")
    ap.add_argument("--buf-tam", default="0x100000")
    ap.add_argument("--clk", type=int, default=100, help="clock da PL em MHz")
    ap.add_argument("--arm", type=int, default=667, help="clock do ARM em MHz")
    ap.add_argument("--run", default="", help="rodada dona destes pesos")
    ap.add_argument("--plano", type=Path, default=None,
                    help="plano da microarquitetura; diz se a porta de escrita "
                         "de peso existe, e portanto de onde o peso vem")
    a = ap.parse_args()

    d = a.dir
    escala, jan, cls = le_pacote(d / "entrada_ps.bin")
    plano = json.loads(a.plano.read_text()) if a.plano else {}
    escrita = bool(plano.get("escrita_de_peso"))
    modo_fluxo = bool(plano.get("fluxo"))
    nvec, inlen = jan.shape
    if modo_fluxo:
        u = plano["camadas"][-1]
        npg = u["pox"] // min(u["pox"], u["pool"])
        npos = u["comp"] // u["pool"]
        am_por_dec = npg * (plano["entrada"] // npos)
        inlen = plano["entrada"]
    else:
        am_por_dec = 0

    n_gold = len((d / "gold_classes.txt").read_text().split())
    ncls = le_mem(d / "gold_logits.mem", 32).size // n_gold

    mult = {}
    for f in sorted(d.glob("mult_conv*.mem")):
        mult[f.stem.replace("mult_", "")] = le_mem(f, 18)

    manifesto = json.loads((d / "manifest.json").read_text())
    nomes = [L["name"] for L in manifesto["layers"]]
    conv = [n for n in nomes if n.startswith("conv")]
    pesos = [le_mem(d / f"pesos_{n}.mem", 8) for n in nomes]
    bias  = [le_mem(d / f"bias_{n}.mem", 32) for n in nomes]

    nw = sum(v.size for v in pesos)
    nb = sum(v.size for v in bias)

    p = ["/* Gerado por ferramentas/gera_pesos_h.py - nao editar a mao. */",
         "#ifndef PESOS_H", "#define PESOS_H", "", "#include <stdint.h>", "",
         f"#define ACEL_BASE    {a.base}u",
         f"#define DMA_BASE     {a.base_dma}u",
         f"#define BUF_DMA_FIS  {a.buf_dma}u",
         f"#define BUF_DMA_TAM  {a.buf_tam}u",
         f"#define PL_CLK_MHZ   {a.clk}u",
         f"#define ARM_CLK_MHZ  {a.arm}u",
         f"#define IN_LEN       {inlen}u",
         "// em fluxo continuo o motor nao para: a aplicacao empurra amostra e",
         "// colhe decisao. AMOSTRAS_POR_DECISAO vem do dobramento da ultima",
         "// camada, e e' o que liga a decisao k a' janela que ela cobre",
         f"#define MODO_FLUXO   {1 if modo_fluxo else 0}",
         f"#define AMOSTRAS_POR_DECISAO {am_por_dec}u",
         f"#define N_CLASSES    {ncls}",
         f"#define N_VETORES    {nvec}",
         "// escala da entrada, fixada no treino. A janela chega ao ARM ja'",
         "// normalizada e em int8; a constante fica aqui so' para o programa",
         "// recusar um pacote que nao seja o da rede embarcada",
         f"#define INPUT_SCALE  {escala!r}",
         f"#define N_W_TOTAL    {nw}u",
         f"#define N_B_TOTAL    {nb}u",
         "// Com a porta de escrita ligada a memoria de peso vira RAM de varias",
         "// portas e a sintese perde a inicializacao embarcada. Uma arquitetura",
         "// que atualiza peso em operacao nao pode depender do bitstream:",
         "// o peso vem pelo barramento, e o embarcado e' que e' o caso especial.",
         f"#define PESO_EMBARCADO {0 if escrita else 1}",
         "// carimbo da rodada: a placa imprime, e o fluxo recusa a medida se",
         "// a imagem que respondeu nao for a que acabou de ser gerada",
         f'#define RUN_ID       "{a.run or "?"}"',
         f'#define PESOS_SHA    "{digesto(d)}"', ""]

    convs = [L for L in manifesto["layers"] if L["type"] == "conv"]
    ativ = max(L["out_ch"] * L["out_len"] for L in convs)
    p.append(f"#define N_CAMADAS    {len(nomes)}")
    p.append(f"#define N_CONV       {len(conv)}")
    p.append(f"#define MAX_ATIV     {max(ativ, inlen)}")
    p.append(f"#define POOL_AVG     {1 if convs[0]['pool']['type'] == 'avg' else 0}")
    p.append(f"#define GAP          "
             f"{1 if manifesto['network'].get('head') == 'gap' else 0}")
    p.append("")
    p.append("static const int NIF_TAB[N_CONV]  = {"
             + ", ".join(str(L["in_ch"]) for L in convs) + "};")
    p.append("static const int NOF_TAB[N_CONV]  = {"
             + ", ".join(str(L["out_ch"]) for L in convs) + "};")
    p.append("static const int K_TAB[N_CONV]    = {"
             + ", ".join(str(L["kernel"]) for L in convs) + "};")
    p.append("static const int LEN_TAB[N_CONV]  = {"
             + ", ".join(str(L["in_len"]) for L in convs) + "};")
    p.append("static const int POOL_TAB[N_CONV] = {"
             + ", ".join(str(L["pool"]["size"]) for L in convs) + "};")
    p.append("")

    for i, n in enumerate(nomes):
        p.append(vetor_c(f"W_L{i}", "int8_t", pesos[i], 24))
        p.append("")
        p.append(vetor_c(f"B_L{i}", "int32_t", bias[i], 8))
        p.append("")

    p.append("static const int8_t *const W_TAB[N_CAMADAS] = {"
             + ", ".join(f"W_L{i}" for i in range(len(nomes))) + "};")
    p.append("static const uint32_t N_W_TAB[N_CAMADAS] = {"
             + ", ".join(str(v.size) for v in pesos) + "};")
    p.append("static const int32_t *const B_TAB[N_CAMADAS] = {"
             + ", ".join(f"B_L{i}" for i in range(len(nomes))) + "};")
    p.append("static const uint32_t N_B_TAB[N_CAMADAS] = {"
             + ", ".join(str(v.size) for v in bias) + "};")
    por_canal = bool(a.plano
                     and json.loads(a.plano.read_text()).get("rq_por_canal"))
    for i, n in enumerate(conv):
        if por_canal:
            if mult[n].size != convs[i]["out_ch"]:
                raise SystemExit(
                    f"{n}: {mult[n].size} multiplicadores para "
                    f"{convs[i]['out_ch']} canais de saida")
        else:
            unicos = set(mult[n].tolist())
            if len(unicos) > 1:
                raise SystemExit(
                    f"{n}: o plano pede uma constante de requantizacao por "
                    f"camada e o modelo trouxe {len(unicos)}. Quantize com "
                    f"--escala-por-camada ou gere com --rq-por-canal")
            mult[n] = mult[n][:1]
        p.append(vetor_c(f"MULT_L{i}", "int32_t", mult[n], 16))
        p.append("")
    p.append("static const int32_t *const MULT_TAB[N_CONV] = {"
             + ", ".join(f"MULT_L{i}" for i in range(len(conv))) + "};")
    p.append("static const uint32_t N_MULT_TAB[N_CONV] = {"
             + ", ".join(str(mult[n].size) for n in conv) + "};")
    p.append(f"#define N_MULT_TOTAL {sum(mult[n].size for n in conv)}u")
    p.append("")

    p.append("")
    p.append("#endif")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(p) + "\n")
    tam = a.out.stat().st_size
    print(f"{a.out}: {tam/1e6:.1f} MB, {nw} pesos, {nb} bias, "
          f"{nvec} vetores de {inlen} amostras")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
