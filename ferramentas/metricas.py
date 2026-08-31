"""junta area, tempo, potencia e correcao num relatorio unico."""

import csv
import json
import math
import re
from pathlib import Path

CHIP = {"lut": 53200, "ff": 106400, "bram36": 140, "dsp": 220}
ACC_DSP = 48
BRAM_BITS = 36864

def _tabela(txt: str, nome: str) -> list | None:
    for ln in txt.splitlines():
        if ln.startswith("|") and ln.split("|")[1].strip() == nome:
            return [c.strip() for c in ln.split("|")[1:-1]]
    return None

def area(dir_rpt: Path) -> dict:
    f = dir_rpt / "util.rpt"
    if not f.exists():
        return {}
    t = f.read_text()
    campos = {"lut": "Slice LUTs", "ff": "Slice Registers",
              "bram36": "Block RAM Tile", "dsp": "DSPs"}
    r = {}
    for k, nome in campos.items():
        ln = _tabela(t, nome)
        if ln:
            r[k] = float(ln[1]) if k == "bram36" else int(ln[1])
            r[k + "_pct"] = round(100.0 * r[k] / CHIP[k], 2)
    return r

def tempo(dir_rpt: Path) -> dict:
    f = dir_rpt / "timing.rpt"
    if not f.exists():
        return {}
    t = f.read_text()
    r = {}
    m = re.search(r"^(\S+)\s+\{[\d.\s]+\}\s+([\d.]+)\s+([\d.]+)", t, re.M)
    if m:
        r["clock"] = m.group(1)
        r["periodo_ns"] = float(m.group(2))
        r["freq_mhz"] = float(m.group(3))
    m = re.search(r"Setup\s*:\s*(\d+)\s+Failing Endpoints,\s*Worst Slack\s*"
                  r"(-?[\d.]+)ns", t)
    if m:
        r["setup_falhas"] = int(m.group(1))
        r["wns_ns"] = float(m.group(2))
        r["fecha"] = r["wns_ns"] >= 0 and r["setup_falhas"] == 0
        if "periodo_ns" in r:

            r["fmax_mhz"] = round(1000.0 / (r["periodo_ns"] - r["wns_ns"]), 1)
    m = re.search(r"Hold\s*:\s*(\d+)\s+Failing Endpoints,\s*Worst Slack\s*"
                  r"(-?[\d.]+)ns", t)
    if m:
        r["hold_falhas"] = int(m.group(1))
        r["whs_ns"] = float(m.group(2))
    return r

def potencia(dir_rpt: Path) -> dict:
    f = dir_rpt / "potencia.rpt"
    if not f.exists():
        return {}
    t = f.read_text()
    r = {}
    for chave, nome in (("total_w", "Total On-Chip Power (W)"),
                        ("dinamica_w", "Dynamic (W)"),
                        ("estatica_w", "Device Static (W)")):
        ln = _tabela(t, nome)
        if ln:
            r[chave] = float(ln[1])
    for chave, nome in (("ps7_w", "PS7"), ("dsp_w", "DSPs"),
                        ("bram_w", "Block RAM"), ("sinais_w", "Signals"),
                        ("logica_w", "Slice Logic"), ("relogio_w", "Clocks")):
        ln = _tabela(t, nome)
        if ln:
            r[chave] = float(ln[1].lstrip("<"))
    if "dinamica_w" in r and "ps7_w" in r:
        r["pl_dinamica_w"] = round(r["dinamica_w"] - r["ps7_w"], 3)
    if "total_w" in r:

        m = re.search(r"Confidence Level\s*\|\s*(\w+)", t)
        if m:
            r["confianca"] = m.group(1)
        r["origem"] = "estimativa do Vivado, sem vetores de atividade"
    return r

_PADROES = {
    "acuracia_pct":      r"acuracia\s*:\s*([\d.]+)%",
    "concorda_pct":      r"concorda com o modelo:\s*([\d.]+)%",
    "estouro_acumulador": r"estouro de acumulador:\s*(\d+)",
    "decisoes_perdidas": r"decisoes perdidas por fila cheia:\s*(\d+)",
    "decisoes_colhidas": r"decisoes:\s*(\d+) colhidas",
    "amostras_por_s":    r"vazao\s*:\s*(\d+) amostras/s",
    "ciclos_por_amostra": r"->\s*([\d.]+) ciclos por amostra",
    "envio_janela_us":   r"envio da janela\s*:\s*([\d.]+) us",
    "envio_registrador_us": r"envio por registrador:\s*([\d.]+) us",
    "espera_us":         r"espera pelo done\s*:\s*([\d.]+) us",
    "pl_p99_us":         r"PL\s*:\s*p50 [\d.]+\s+p99 ([\d.]+)",
    "pl_p999_us":        r"PL\s*:\s*p50 [\d.]+\s+p99 [\d.]+\s+p99\.9 ([\d.]+)",
    "sistema_p99_us":    r"sistema\s*:\s*p50 [\d.]+\s+p99 ([\d.]+)",
    "sistema_p999_us":   r"sistema\s*:\s*p50 [\d.]+\s+p99 [\d.]+\s+p99\.9 ([\d.]+)",
    "fora_do_prazo_pl":  r"fora do prazo PL\s*:\s*(\d+) de",
    "fora_do_prazo_sistema": r"fora do prazo sistema\s*:\s*(\d+) de",
    "tempo_real":        r"tempo real\s*:\s*(.+)",
    "leitura_us":        r"leitura do resultado:\s*([\d.]+) us",
    "fila_apos_envio":   r"fila apos o envio\s*:\s*(\d+) amostras",
    "fora_do_computo_pct": r"fora do computo\s*:\s*([\d.]+)%",
    "vazao_inf_s":       r"vazao\s*:\s*(\d+) inferencias",
    "arm_us":            r"^\s*ARM\s+:\s*([\d.]+) us",
    "ganho_computo":     r"ganho no computo\s*:\s*([\d.]+)x",
    "ganho_sistema":     r"ganho de sistema\s*:\s*([\d.]+)x",
    "config_ms":         r"so config:\s*([\d.]+) ms",
    "pl_mhz_aferido":    r"relogio da PL\s*:\s*([\d.]+) MHz aferido",
    "tique_ns":          r"->\s*([\d.]+) ns por tique",
}

def da_placa(texto: str) -> dict:
    r = {}
    for chave, pad in _PADROES.items():
        m = re.search(pad, texto, re.M)
        if m:
            v = m.group(1).strip()
            try:
                r[chave] = int(v) if v.isdigit() else float(v)
            except ValueError:
                r[chave] = v
    m = re.search(r"computo na PL\s*:\s*(\d+)\.\.(\d+) ciclos, jitter (\d+)"
                  r"\s*\(([\d.]+) us\)", texto)
    if m:
        r["ciclos_min"] = int(m.group(1))
        r["ciclos_max"] = int(m.group(2))
        r["jitter_ciclos"] = int(m.group(3))
        r["pl_us"] = float(m.group(4))
    m = re.search(r"acelerador\s*:\s*([\d.]+) us\s*\(([\d.]+)\.\.([\d.]+), "
                  r"jitter ([\d.]+)\)", texto)
    if m:
        r["e2e_us"] = float(m.group(1))
        r["e2e_min_us"] = float(m.group(2))
        r["e2e_max_us"] = float(m.group(3))
        r["e2e_jitter_us"] = float(m.group(4))
    m = re.search(r"sistema completo\s*:\s*([\d.]+) us", texto)
    if m:
        r["sistema_us"] = float(m.group(1))
    m = re.search(r"soma das fases\s*:\s*([\d.]+) us\s*\(([\d.]+)\.\.([\d.]+), "
                  r"jitter ([\d.]+)\)", texto)
    if m:
        r["fases_us"] = float(m.group(1))
        r["sistema_min_us"] = float(m.group(2))
        r["sistema_max_us"] = float(m.group(3))
        r["sistema_jitter_us"] = float(m.group(4))
    m = re.search(r"logits identicos\s*:\s*(\d+)/(\d+)", texto)
    if m:
        r["logits_iguais"] = int(m.group(1))
        r["logits_conferidos"] = int(m.group(2))
    m = re.search(r"envio por registrador:.*?(\d+)/(\d+) com logits identicos",
                  texto)
    if m:
        r["rajada_igual_registrador"] = int(m.group(1))
        r["rajada_conferidas"] = int(m.group(2))
    m = re.search(r"carga de (\d+) pesos \+ (\d+) bias", texto)
    if m:
        r["n_pesos"] = int(m.group(1))
        r["n_bias"] = int(m.group(2))

    m = re.search(r"ciclos da PL\s*:\s*(\d+) para (\d+) amostras\s*->\s*"
                  r"([\d.]+) ciclos por amostra", texto)
    if m:
        r["fluxo_ciclos_pl"] = int(m.group(1))
        r["fluxo_amostras"] = int(m.group(2))
        r["fluxo_ciclos_por_amostra"] = float(m.group(3))
    m = re.search(r"vazao da PL\s*:\s*(\d+) amostras/s", texto)
    if m:
        r["fluxo_amostras_s_pl"] = int(m.group(1))
    m = re.search(r"vazao\s*:\s*(\d+) amostras/s, (\d+) decisoes/s", texto)
    if m:
        r["fluxo_amostras_s"] = int(m.group(1))
        r["fluxo_decisoes_s"] = int(m.group(2))
    m = re.search(r"uma a cada (\d+) amostras", texto)
    if m:
        r["fluxo_dec_intervalo"] = int(m.group(1))
    m = re.search(r"decisoes: (\d+) colhidas de (\d+)", texto)
    if m:
        r["fluxo_decisoes"] = int(m.group(1))
        r["fluxo_decisoes_esperadas"] = int(m.group(2))
    m = re.search(r"decisoes perdidas por fila cheia: (\d+)", texto)
    if m:
        r["fluxo_perdidas_fila"] = int(m.group(1))
    m = re.search(r"sequencia: (\d+) saltos, (\d+) decisoes perdidas", texto)
    if m:
        r["fluxo_saltos"] = int(m.group(1))
        r["fluxo_perdidas_seq"] = int(m.group(2))
    m = re.search(r"ARM em fluxo\s*:\s*(\d+) amostras/s", texto)
    if m:
        r["fluxo_arm_amostras_s"] = int(m.group(1))
    m = re.search(r"intervalo entre decisoes: (\d+)\.\.(\d+) ciclos, "
                  r"jitter (\d+)\s*\((\d+) intervalos\)", texto)
    if m:
        r["fluxo_iv_min"] = int(m.group(1))
        r["fluxo_iv_max"] = int(m.group(2))
        r["fluxo_iv_jitter"] = int(m.group(3))
        r["fluxo_iv_n"] = int(m.group(4))
    m = re.search(r"ciclos de parada por entrada vazia: (\d+) de (\d+)", texto)
    if m:
        r["fluxo_ciclos_parada"] = int(m.group(1))
    return r

def folga(plano: dict, ar: dict, pl: dict) -> dict:
    r = {}

    bits = {c["nome"]: c["acc_w"] for c in plano["camadas"]}
    pior = max(bits.values())
    r["aritmetica"] = {
        "bits_por_camada": bits,
        "bits_disponiveis": ACC_DSP,
        "pior_caso_usado": pior,
        "bits_ociosos": ACC_DSP - pior,
        "margem_termos": 2 ** (ACC_DSP - pior),
    }

    blocos = plano.get("dsp_blocos", plano["dsp_total"])
    r["multiplicadores"] = {
        "planejados": plano["dsp_total"],
        "em_blocos_dsp": blocos,
        "em_logica": plano.get("mac_logica", 0),
        "livres_no_plano": CHIP["dsp"] - blocos,
        "ocupacao_temporal": round(plano["eficiencia_dsp"], 4),
    }
    if "dsp" in ar:
        r["multiplicadores"]["implementados"] = ar["dsp"]
        r["multiplicadores"]["livres_no_chip"] = CHIP["dsp"] - ar["dsp"]

    if "bram36" in ar:
        livre = CHIP["bram36"] - ar["bram36"]
        r["memoria"] = {
            "bram36_usadas": ar["bram36"],
            "bram36_livres": round(livre, 1),
            "bytes_livres": int(livre * BRAM_BITS / 8),
        }

    pesos = sum(c["nof"] * c["nif"] * c["k"] for c in plano["camadas"]) \
        + plano["fc_nflat"] * 4
    r["escrita_de_peso"] = {
        "pesos": pesos,
        "porta_existe": bool(plano.get("escrita_de_peso", False)),
        "usada_na_operacao": False,
        "ns_por_peso_medido": 176.3,
        "tempo_para_reescrever_ms": round(pesos * 176.3e-6, 3),
        "custo_de_manter_a_porta": {"ff": 6408, "bram18": 10, "folga_ns": 0.106},
    }
    if "e2e_us" in pl:
        r["escrita_de_peso"]["inferencias_equivalentes"] = round(
            pesos * 176.3e-3 / pl["e2e_us"], 1)
    return r

def modelo_vs_medido(plano: dict, ar: dict, pl: dict,
                     latencia_prevista=None) -> dict:
    r = {}

    prev_dsp = plano.get("dsp_blocos", plano["dsp_total"])
    prev_bram = plano.get("bram")
    itens = {}
    if "dsp" in ar:
        itens["dsp"] = (prev_dsp, ar["dsp"])
    if prev_bram is not None and "bram36" in ar:
        itens["bram36"] = (prev_bram, ar["bram36"])
    if itens:
        r["recursos"] = {k: {"previsto": p, "medido": m,
                             "desvio": round(m - p, 2),
                             "bate": abs(m - p) < 1e-9}
                         for k, (p, m) in itens.items()}
        r["recursos_batem"] = all(v["bate"] for v in r["recursos"].values())

    if "fluxo_ciclos_por_amostra" in pl:
        der = plano["ii"] / plano["entrada"]
        med = pl["fluxo_ciclos_por_amostra"]
        r["ciclos_por_amostra_derivado"] = round(der, 4)
        r["ciclos_por_amostra_medido"] = med
        r["excesso_pct"] = round(100.0 * (med / der - 1), 3) if der else None
        if "fluxo_ciclos_parada" in pl and "fluxo_amostras" in pl:
            parada = pl["fluxo_ciclos_parada"] / pl["fluxo_amostras"]
            r["ciclos_de_parada_por_amostra"] = round(parada, 4)
            r["excesso_atribuido_a_parada_de_entrada"] = bool(
                med - der <= parada + 1e-3)
        if "fluxo_iv_jitter" in pl:
            r["jitter_entre_decisoes_ciclos"] = pl["fluxo_iv_jitter"]
            r["intervalo_entre_decisoes_ciclos"] = [pl["fluxo_iv_min"],
                                                    pl["fluxo_iv_max"]]
            r["intervalos_medidos"] = pl.get("fluxo_iv_n")
            der_iv = plano["ii"] * plano.get("dec_intervalo", 0) / plano["entrada"]
            if der_iv:
                r["intervalo_derivado_ciclos"] = round(der_iv, 2)
        return r

    if "ciclos_max" not in pl:
        return r
    med = pl["ciclos_max"]
    r.update({"ii_derivado": plano["ii"],
              "latencia_medida_ciclos": med,
              "enchimento_do_pipe_ciclos": med - plano["ii"]})
    if latencia_prevista:
        r["latencia_prevista_ciclos"] = latencia_prevista
        r["desvio_ciclos"] = med - latencia_prevista
    return r

def energia(pot: dict, pl: dict) -> dict:
    if "total_w" not in pot:
        return {}
    r = {"potencia_w": pot["total_w"],
         "origem": "estimativa do Vivado, sem vetores de atividade"}
    if "fluxo_decisoes_s" in pl and pl["fluxo_decisoes_s"]:
        r["energia_por_decisao_uj"] = round(
            1e6 * pot["total_w"] / pl["fluxo_decisoes_s"], 2)
        if "pl_dinamica_w" in pot:
            r["energia_pl_por_decisao_uj"] = round(
                1e6 * pot["pl_dinamica_w"] / pl["fluxo_decisoes_s"], 3)
    if "fluxo_amostras_s_pl" in pl and pl["fluxo_amostras_s_pl"] \
            and "pl_dinamica_w" in pot:
        r["energia_pl_por_amostra_nj"] = round(
            1e9 * pot["pl_dinamica_w"] / pl["fluxo_amostras_s_pl"], 2)
    if "e2e_us" in pl:
        r["energia_por_inferencia_uj"] = round(
            pot["total_w"] * pl["e2e_us"], 1)
    if "pl_us" in pl:
        r["energia_do_computo_uj"] = round(pot["total_w"] * pl["pl_us"], 1)

    if "pl_dinamica_w" in pot:
        r["potencia_pl_w"] = pot["pl_dinamica_w"]
        r["potencia_ps7_w"] = pot["ps7_w"]
        r["fracao_ps7"] = round(pot["ps7_w"] / pot["dinamica_w"], 3)
        if "pl_us" in pl:
            r["energia_pl_por_inferencia_uj"] = round(
                pot["pl_dinamica_w"] * pl["pl_us"], 1)
    return r

def eficiencia(plano: dict, ar: dict, tm: dict, pl: dict) -> dict:
    r = {}
    macs = plano.get("macs")

    if macs and "fluxo_amostras_s_pl" in pl:
        por_amostra = macs / plano["entrada"]
        gops = 2.0 * por_amostra * pl["fluxo_amostras_s_pl"] / 1e9
        r["mac_por_amostra"] = round(por_amostra, 1)
        r["gops"] = round(gops, 2)
        if ar.get("dsp"):
            r["gops_por_dsp"] = round(gops / ar["dsp"], 4)
            cpa = pl.get("fluxo_ciclos_por_amostra")
            if cpa:
                r["mac_por_dsp_ciclo"] = round(
                    por_amostra / (ar["dsp"] * cpa), 4)
        if "fluxo_decisoes_s" in pl:
            r["decisoes_s"] = pl["fluxo_decisoes_s"]
            r["mac_por_decisao"] = round(
                por_amostra * plano.get("dec_intervalo", 1), 1)
        if "fluxo_arm_amostras_s" in pl and pl["fluxo_arm_amostras_s"]:
            r["ganho_sobre_arm_em_fluxo"] = round(
                pl["fluxo_amostras_s_pl"] / pl["fluxo_arm_amostras_s"], 1)
        return r

    if macs and "pl_us" in pl:
        gops = 2.0 * macs / (pl["pl_us"] * 1e3)
        r["gops"] = round(gops, 2)
        if "dsp" in ar and ar["dsp"]:
            r["gops_por_dsp"] = round(gops / ar["dsp"], 4)
    if "freq_mhz" in tm and "ciclos_max" in pl:
        r["latencia_teorica_us"] = round(pl["ciclos_max"] / tm["freq_mhz"], 2)
    return r

def paralelismo(plano: dict, pl: dict) -> dict:
    cams = plano["camadas"]
    lanes = sum(c["pof"] * c["pox"] * c["pk"] for c in cams)
    macs = sum(c["nof"] * c["nif"] * c["k"] * c["comp"] for c in cams)
    ii = plano["ii"]
    r = {
        "lanes": lanes,
        "ocupacao_das_lanes": round(macs / (lanes * ii), 4) if lanes and ii else 0,
        "dobra_de_filtro": {c["nome"]: c["nof"] // c["pof"] for c in cams},
        "dobra_de_posicao": {c["nome"]: c["comp"] // c["pox"] for c in cams},
        "dobra_de_tap": {c["nome"]: c["k"] // c["pk"] for c in cams},
    }
    ciclos = [c["ciclos"] for c in cams]
    r["desequilibrio_entre_camadas"] = round(1 - min(ciclos) / max(ciclos), 4)
    r["ciclos_por_camada"] = {c["nome"]: c["ciclos"] for c in cams}
    if "ciclos_max" in pl:
        r["enchimento_ciclos"] = pl["ciclos_max"] - ii
        r["custo_do_enchimento"] = round(pl["ciclos_max"] / ii - 1, 4)
    return r

def procedencia(dir_rpt: Path) -> dict:
    r = {}
    for arq in ("potencia.rpt", "timing.rpt", "util.rpt"):
        f = dir_rpt / arq
        if not f.exists():
            continue
        cab = f.read_text()[:2000]
        for chave, rot in (("ferramenta", "Tool Version"),
                           ("peca", "Device"), ("estado", "Design State"),
                           ("topo", "Design")):
            m = re.search(rf"\|\s*{rot}\s*:\s*(.+)", cab)
            if m and chave not in r:
                r[chave] = m.group(1).strip()
        break
    r["diretivas"] = {"synth_design": "default", "opt_design": "default",
                      "place_design": "default", "phys_opt_design": "default",
                      "route_design": "default"}
    r["determinismo"] = ("implementacao sem semente aleatoria (UG904): mesma "
                         "versao e mesmas diretivas dao o mesmo resultado")
    return r

def passo_de_treino(plano: dict, ar: dict, pl: dict, tm: dict,
                    taxa: float | None) -> dict:
    cams = plano["camadas"]
    r = {}

    ida = plano["macs"]
    volta = 2 * ida - (cams[0]["nof"] * cams[0]["nif"]
                       * cams[0]["k"] * cams[0]["comp"])
    r["aritmetica"] = {
        "macs_ida": ida,
        "macs_volta": volta,
        "razao_volta_ida": round(volta / ida, 2),
        "ciclos_ida": plano["ii"],
        "ciclos_volta_cota": int(plano["ii"] * volta / ida),
        "ciclos_passo_cota": int(plano["ii"] * (1 + volta / ida)),
    }

    if taxa and "freq_mhz" in tm:
        prazo = tm["freq_mhz"] * 1e6 / taxa
        ocioso = prazo - plano["ii"]
        preciso = r["aritmetica"]["ciclos_volta_cota"]
        r["tempo"] = {
            "prazo_ciclos": int(prazo),
            "ociosos_por_janela": int(ocioso),
            "ciclos_para_a_volta": preciso,
            "cabe_em_uma_janela": bool(ocioso >= preciso),
            "janelas_para_pagar_um_passo": round(max(preciso, 0)
                                                 / ocioso, 2) if ocioso > 0
                                            else None,
        }

    guardar = sum(c["nof"] * (c["comp"] // c["pool"]) for c in cams)
    r["memoria"] = {"bytes_de_ativacao_por_janela": guardar}
    if "bram36" in ar:
        livres = int((CHIP["bram36"] - ar["bram36"]) * BRAM_BITS / 8)
        r["memoria"]["bytes_livres_em_bram"] = livres
        r["memoria"]["janelas_de_lote_que_cabem"] = livres // guardar
        r["memoria"]["cabe_pelo_menos_uma"] = bool(livres >= guardar)

    prec = max(math.ceil(math.log2(c["comp"] * 127 * 127)) + 1 for c in cams)
    r["faixa"] = {
        "bits_do_acumulador": ACC_DSP,
        "bits_usados_na_ida": max(c["acc_w"] for c in cams),
        "bits_para_o_gradiente_de_peso": prec,
        "cabe_no_acumulador": bool(prec <= ACC_DSP),
        "bits_de_sobra": ACC_DSP - prec,
    }

    pesos = plano["pesos_total"]
    r["banda"] = {
        "pesos_a_reescrever": pesos,
        "ms_para_reescrever": round(pesos * 176.3e-6, 3),
    }
    if taxa and "freq_mhz" in tm:
        ocioso_us = (r["tempo"]["ociosos_por_janela"] / tm["freq_mhz"])
        if ocioso_us > 0:
            r["banda"]["janelas_ociosas_para_uma_atualizacao"] = round(
                pesos * 176.3e-3 / ocioso_us, 1)
    return r

def consolida(plano: dict, ctx: dict, dir_rpt: Path) -> dict:
    ar = area(dir_rpt)
    tm = tempo(dir_rpt)
    pot = potencia(dir_rpt)
    pl = da_placa(ctx.get("placa", ""))

    m = {
        "identificador": ctx.get("nome"),
        "topologia": ctx.get("model"),
        "microarquitetura": {
            "multiplicadores": plano["dsp_total"],
            "blocos_dsp": plano.get("dsp_blocos", plano["dsp_total"]),
            "multiplicadores_em_logica": plano.get("mac_logica", 0),
            "ciclos_por_inferencia": plano["ii"],
            "ocupacao": round(plano["eficiencia_dsp"], 4),
            "pesos": plano["pesos_total"],
            "camadas": [{k: c[k] for k in
                         ("nome", "nif", "nof", "k", "comp", "pox", "pk",
                          "nrqu", "acc_w", "dsp", "ciclos") if k in c}
                        for c in plano["camadas"]],
        },
        "correcao": {k: pl[k] for k in
                     ("acuracia_pct", "concorda_pct",
                      "estouro_acumulador", "logits_iguais",
                      "logits_conferidos", "fluxo_decisoes",
                      "fluxo_decisoes_esperadas", "fluxo_perdidas_fila",
                      "fluxo_saltos", "fluxo_perdidas_seq") if k in pl},
        "tempo": {k: pl[k] for k in
                  ("ciclos_min", "ciclos_max", "jitter_ciclos", "pl_us",
                   "envio_janela_us", "espera_us", "leitura_us",
                   "e2e_us", "e2e_min_us", "e2e_max_us", "e2e_jitter_us",
                   "fila_apos_envio", "sistema_us", "sistema_min_us",
                   "sistema_max_us", "sistema_jitter_us",
                   "fora_do_computo_pct", "vazao_inf_s",
                   "config_ms", "pl_p99_us", "pl_p999_us",
                   "sistema_p99_us", "sistema_p999_us",
                   "fora_do_prazo_pl", "fora_do_prazo_sistema",
                   "fluxo_ciclos_por_amostra", "fluxo_ciclos_pl",
                   "fluxo_amostras", "fluxo_amostras_s", "fluxo_amostras_s_pl",
                   "fluxo_decisoes_s", "fluxo_dec_intervalo",
                   "fluxo_iv_min", "fluxo_iv_max", "fluxo_iv_jitter",
                   "fluxo_iv_n", "fluxo_ciclos_parada",
                   "tempo_real") if k in pl},
        "area": ar,
        "frequencia": tm,
        "potencia": pot,
        "energia": energia(pot, pl),
        "eficiencia": eficiencia(plano, ar, tm, pl),
        "modelo_vs_medido": modelo_vs_medido(
            plano, ar, pl, ctx.get("latencia_prevista")),
        "comparacao_arm": dict(
            {k: pl[k] for k in ("arm_us", "ganho_computo", "ganho_sistema")
             if k in pl},
            origem="inferencia inteira em C, -O3, sem NEON escrito a mao, "
                   "com a topologia lida em tabela"),
        "paralelismo": paralelismo(plano, pl),
        "procedencia": procedencia(dir_rpt),
        "folga": folga(plano, ar, pl),
        "passo_de_treino": passo_de_treino(plano, ar, pl, tm,
                                           ctx.get("taxa")),
    }
    return m

def _linhas(m: dict) -> list:
    saida = []

    def anda(pref, v):
        if isinstance(v, dict):
            for k, x in v.items():
                anda(f"{pref}.{k}" if pref else k, x)
        elif isinstance(v, list):
            saida.append((pref, json.dumps(v)))
        else:
            saida.append((pref, v))
    anda("", m)
    return saida

def escreve(m: dict, saida: Path) -> None:
    (saida / "metricas.json").write_text(json.dumps(m, indent=2))
    with (saida / "metricas.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerows(_linhas(m))

def imprime(m: dict) -> None:
    def num(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    def bloco(titulo, d, unid=None):
        if not d:
            return
        print(f"\n  {titulo}")
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                continue
            print(f"    {k:<28} {num(v)}")

    print(f"\n=== {m['identificador']} ===")
    bloco("correcao", m["correcao"])
    bloco("tempo", m["tempo"])
    bloco("area", m["area"])
    bloco("frequencia", m["frequencia"])
    bloco("potencia", m["potencia"])
    bloco("energia", m["energia"])
    bloco("eficiencia", m["eficiencia"])
    bloco("o que o gerador previu contra o que o silicio mediu",
          m["modelo_vs_medido"])
    bloco("contra o ARM", m["comparacao_arm"])
    fg = m["folga"]
    print("\n  folga (o que habilita o trabalho futuro)")
    for sec in ("aritmetica", "multiplicadores", "memoria", "banda_de_peso"):
        if sec in fg:
            for k, v in fg[sec].items():
                if not isinstance(v, dict):
                    print(f"    {sec}.{k:<20} {num(v)}")

    pa = m.get("paralelismo") or {}
    if pa:
        print("\n  paralelismo (espaco contra tempo)")
        for k, v in pa.items():
            print(f"    {k:32s} {v if not isinstance(v, dict) else v}")

    pt = m.get("passo_de_treino") or {}
    if pt:
        print("\n  o passo de treino cabe nesta folga? (cotas inferiores)")
        for sec in ("aritmetica", "tempo", "memoria", "faixa", "banda"):
            for k, v in (pt.get(sec) or {}).items():
                if not isinstance(v, dict):
                    print(f"    {sec}.{k:<32} {num(v)}")
