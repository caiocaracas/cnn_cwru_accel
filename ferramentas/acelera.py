"""orquestra as etapas que levam a rede treinada ate a placa."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent

ETAPAS = ["valida", "quantiza", "modela", "confere", "recursos", "sintetiza",
          "sistema", "placa", "relatorio"]

TEXTOS = {
    "quantiza":  "quantizando o modelo treinado",
    "modela":    "modelando a microarquitetura",
    "confere":   "conferindo exatidao do hardware",
    "recursos":  "medindo area do acelerador",
    "sintetiza": "sintetizando e gerando o bitstream",
    "sistema":   "montando o sistema da placa",
    "placa":     "rodando na placa",
    "relatorio": "consolidando resultado",
}

class Falha(Exception):
    pass

def passo(n: int, total: int, texto: str) -> float:
    print(f"\n[{n}/{total}] {texto}", flush=True)
    return time.time()

def fim(t0: float) -> None:
    print(f"      ({time.time()-t0:.0f}s)", flush=True)

def roda(cmd: list, registro: Path, env: dict | None = None) -> None:
    if cmd[0] == sys.executable and "-u" not in cmd:
        cmd = [cmd[0], "-u"] + cmd[1:]
    registro.parent.mkdir(parents=True, exist_ok=True)
    with registro.open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=RAIZ, env=env)
    if r.returncode:
        cauda = registro.read_text().splitlines()[-15:]
        raise Falha(f"{cmd[0]} falhou (codigo {r.returncode}); "
                    f"registro em {registro}\n  " + "\n  ".join(cauda))

def acompanha(cmd: list, registro: Path) -> str:
    registro.parent.mkdir(parents=True, exist_ok=True)
    linhas = []
    with registro.open("w") as fh:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=RAIZ, text=True)
        for linha in p.stdout:
            fh.write(linha)
            linhas.append(linha)
            print("      " + linha.rstrip(), flush=True)
        p.wait()
    if p.returncode:
        raise Falha(f"{cmd[0]} falhou (codigo {p.returncode}); "
                    f"registro em {registro}")
    return "".join(linhas)

def acha_config(a, modelo: Path) -> Path:
    if a.spec:
        return a.spec
    for cand in (modelo.parent.parent / "config.yaml",
                 modelo.parent / "config.yaml"):
        if cand.exists():
            return cand
    raise Falha(
        f"nao achei a config do treino perto de {modelo}.\n"
        f"  Ela e' escrita por 'python3 -m modelo.prepara'. Para um modelo "
        f"antigo, passe --spec com a config usada no treino dele.")

def nome_da_rodada(a, ctrl) -> str:
    if a.run:
        return a.run
    if a.modelo:
        m = Path(a.modelo).resolve()
        for pai in m.parents:
            if pai.parent.name == "runs":
                return pai.name
    return ctrl.nome

def valida(a) -> dict:
    from ferramentas.le_modelo import le, ModeloInvalido

    modelo = Path(a.modelo or (RAIZ / "runs" / a.run / "quant_int8" / "weights_int.npz"))
    try:
        ctrl, dados = le(modelo)
    except ModeloInvalido as e:
        raise Falha(str(e))

    print(f"      modelo:        {modelo}")
    print(f"      controlador:   {ctrl.num_layers} camadas, "
          f"{ctrl.num_filters_first} filtros iniciais, "
          f"kernel {ctrl.kernel_size}, pooling {ctrl.pool_type}")
    print(f"      caminho dados: {dados.n_pesos} pesos, {dados.n_bias} bias, "
          f"{len(dados.camadas)} memorias")
    run = nome_da_rodada(a, ctrl)
    print(f"      rodada:        {run}")
    print(f"      hardware:      {ctrl.nome}")

    config = acha_config(a, modelo)
    cfg = yaml.safe_load(config.read_text())
    if "training" not in cfg:
        raise Falha(f"{config} nao tem a secao 'training'; sem ela a particao "
                    f"de teste nao pode ser reproduzida")
    cfg["model"] = ctrl.como_dict()
    print(f"      config treino: {config}")
    from ferramentas.gerador import DSP_TOTAL
    sufixo = f"_D{a.orcamento}" if a.orcamento != DSP_TOTAL else ""
    nome_hw = ctrl.nome + ("_fluxo" if a.fluxo else "") + sufixo
    return {"cfg": cfg, "model": ctrl.como_dict(), "nome": nome_hw,
            "run": run, "sufixo": sufixo,
            "hw_data": RAIZ / "results/hw_data" / run,
            "modelo": modelo, "ctrl": ctrl, "dados": dados}

def quantiza(a, ctx, saida: Path) -> None:
    spec_efetiva = saida / "grafo.yaml"
    spec_efetiva.parent.mkdir(parents=True, exist_ok=True)
    spec_efetiva.write_text(yaml.safe_dump(ctx["cfg"], sort_keys=False))
    roda([sys.executable, "-m", "modelo.quantiza",
          "--npz", str(ctx["modelo"]), "--config", str(spec_efetiva),
          "--data-dir", str(a.dados), "--out", str(ctx["hw_data"]),
          "--n-vectors", str(a.n_teste if a.n_teste > 0 else 10 ** 9)],
         saida / "quantiza.log")
    man = json.loads((ctx["hw_data"] / "manifest.json").read_text())
    pesos = sum(L["weights"]["words"] for L in man["layers"])
    print(f"      {pesos} pesos INT8 em {len(man['layers'])} memorias, "
          f"escala de entrada {man['quant']['input_scale']:.4e}")
    print(f"      {man['test']['n_vectors']} janelas de teste para a placa")

def modela(a, ctx, saida: Path) -> dict:
    convs = [c for c in ctx["dados"].camadas if c[0].startswith("features.")]
    por_canal = any(np.size(c[2]) > 1 for c in convs)
    print(f"      requantizacao: "
          + ("uma constante por canal de saida (escalas por canal no modelo)"
             if por_canal else
             "uma constante por camada (o modelo traz escala unica)"))

    cmd = [sys.executable, "-m", "ferramentas.gerador",
           "--spec", str(saida / "grafo.yaml"),
           "--orcamento", str(a.orcamento),
           "--sufixo", ctx.get("sufixo", "")]
    if a.fluxo:
        cmd += ["--fluxo"]
    if a.taxa:
        cmd += ["--taxa", str(a.taxa), "--clock", str(a.clock)]
    if por_canal:
        cmd += ["--rq-por-canal"]
    roda(cmd, saida / "modela.log")
    plano = json.loads((RAIZ / "results/gen" / ctx["nome"] / "plano.json").read_text())
    if not plano["cabe"]:
        raise Falha(f"a topologia nao cabe: {plano['motivo']}")

    roda([sys.executable, "-m", "ferramentas.memorias",
          "--plano", str(RAIZ / "results/gen" / ctx["nome"] / "plano.json"),
          "--hw-data", str(ctx["hw_data"]), "--run", ctx["run"]],
         saida / "memorias.log")

    if a.fluxo:
        roda([sys.executable, "-u", "-m", "ferramentas.pacote_fluxo",
              "--npz", str(ctx["modelo"]), "--config", str(saida / "grafo.yaml"),
              "--plano", str(RAIZ / "results/gen" / ctx["nome"] / "plano.json"),
              "--data-dir", str(a.dados),
              "--out", str(ctx["hw_data"] / "entrada_ps.bin"),
              "--max-amostras", str(a.n_teste if a.n_teste > 0 else 0)],
             saida / "pacote_fluxo.log")
        reg = (saida / "pacote_fluxo.log").read_text().strip().splitlines()
        for ln in reg:
            if "decisoes," in ln or "fluxo:" in ln:
                print("      " + ln.strip())
    print("      " + (saida / "memorias.log").read_text().strip().splitlines()[-1])
    print(f"      {plano['dsp_total']} multiplicadores, "
          f"{plano['ii']} ciclos por inferencia, "
          f"{plano['eficiencia_dsp']:.1%} de ocupacao")
    for c in plano["camadas"]:
        print(f"        {c['nome']}: {c['nof']*c['pox']*c['pk']} lanes, "
              f"{c['nrqu']} via(s) de saida, acumulador de {c['acc_w']} bits")
    return plano

def confere(a, ctx, saida: Path) -> None:
    plano = RAIZ / "results/gen" / ctx["nome"] / "plano.json"
    proc = plano.parent / "mem" / "procedencia.json"
    if not proc.exists():
        raise Falha(f"sem procedencia em {proc}; rode a etapa 'modela'")
    dono = json.loads(proc.read_text())
    if dono.get("run") != ctx["run"]:
        raise Falha(f"as memorias em {proc.parent} sao da rodada "
                    f"{dono.get('run')!r} e esta e' {ctx['run']!r}; "
                    f"rode a etapa 'modela' antes de conferir")
    ferr_cam = ("ferramentas.verifica_fluxo" if a.fluxo
                else "ferramentas.verifica_config")
    roda([sys.executable, "-u", "-m", ferr_cam,
          "--plano", str(plano)]
         + ([] if a.fluxo else ["--nvec", "2"]),
         saida / "confere.log")
    print("      cada camada: exatidao bit-a-bit, com bolha e contrapressao")

    if a.fluxo:
        roda([sys.executable, "-u", "-m", "ferramentas.verifica_topo_fluxo",
              "--plano", str(plano), "--janelas", "3"],
             saida / "topo.log")
    else:
        roda([sys.executable, "-u", "-m", "ferramentas.verifica_topo",
              "--plano", str(plano), "--nvec", "2", "--barramento"],
             saida / "topo.log")
    texto = (saida / "topo.log").read_text()
    if a.fluxo:
        if "FLUXO BIT-EXATO" not in texto:
            raise Falha(f"a rede em fluxo nao reproduz a referencia; "
                        f"registro em {saida / 'topo.log'}")
        passo = [ln for ln in texto.splitlines() if "ciclos entre decisoes" in ln]
        print("      rede em fluxo: bit-exata." + (passo[0] if passo else ""))
        pl = json.loads(plano.read_text())
        ctx["latencia_prevista"] = int(pl["ii"] // pl["entrada"])
        (saida / "latencia_prevista.txt").write_text(str(ctx["latencia_prevista"]))
        return
    ciclos = [ln.split()[1].split("..")[0] for ln in texto.splitlines()
              if ln.strip().startswith("ciclos ")]
    if not ciclos:
        raise Falha(f"a simulacao do topo nao reportou ciclos; "
                    f"registro em {saida / 'topo.log'}")
    ctx["latencia_prevista"] = int(ciclos[-1])
    (saida / "latencia_prevista.txt").write_text(str(ctx["latencia_prevista"]))
    if "BARRAMENTO BIT-EXATO" not in texto:
        raise Falha(f"a rede passou no acesso direto mas nao pelo barramento; "
                    f"registro em {saida / 'topo.log'}")
    print(f"      rede montada: bit-exata pelo barramento, "
          f"{ctx['latencia_prevista']} ciclos de latencia previstos")

def digest_das_fontes(nome: str) -> str:
    import hashlib

    h = hashlib.sha256()
    fontes = sorted((RAIZ / "rtl").glob("*.v"))
    fontes += [RAIZ / "results/gen" / nome / "acelerador_gen.v",
               RAIZ / "scripts/bitstream.tcl", RAIZ / "scripts/leds.xdc"]
    for f in fontes:
        if f.name == "acelerador.v":
            continue
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.exists() else b"<ausente>")
    return h.hexdigest()[:16]

def reaproveita(a, ctx, saida: Path) -> bool:
    if a.refaz_sintese:
        return False
    guardado = RAIZ / "results/vivado" / f"soc_{ctx['nome']}" / "sintese.json"
    bit = RAIZ / "results/vivado" / f"soc_{ctx['nome']}" / f"{ctx['nome']}.bit"
    if not (guardado.exists() and bit.exists()):
        return False

    plano = json.loads(
        (RAIZ / "results/gen" / ctx["nome"] / "plano.json").read_text())
    antes = json.loads(guardado.read_text())
    if antes.get("plano") != plano:
        return False
    if antes.get("fontes") != ctx["fontes"]:
        print("      o Verilog mudou desde este bitstream: sintetizando de novo")
        return False
    if not plano.get("escrita_de_peso"):
        print("      o peso desta topologia nasce no bitstream: a receita nova "
              "exige sintetizar de novo")
        return False

    ctx["clock"] = antes["clock"]
    ctx["enderecos"] = antes["enderecos"]
    (saida / "clock.txt").write_text(str(antes["clock"]))
    (saida / "enderecos.json").write_text(json.dumps(antes["enderecos"], indent=2))
    print(f"      mesmo plano de {antes.get('run', '?')}: reaproveitando o "
          f"bitstream a {antes['clock']} MHz")
    print(f"      o peso entra pelo barramento, entao so' a receita mudou "
          f"(--refaz-sintese forca)")
    return True

def recursos(a, ctx, saida: Path) -> None:
    rel = saida / "recursos.log"
    rel.parent.mkdir(parents=True, exist_ok=True)
    with rel.open("w") as fh:
        r = subprocess.run(
            ["bash", "-lc",
             f"source {a.vivado}/settings64.sh && "
             f"vivado -mode batch -nojournal -nolog -notrace "
             f"-source scripts/recursos.tcl -tclargs {ctx['nome']}"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=RAIZ, env=dict(os.environ))
    texto = rel.read_text()
    med = {ln.split()[1]: ln.split()[2] for ln in texto.splitlines()
           if ln.startswith("RECURSOS") and len(ln.split()) > 2
           and ln.split()[1] != "erro"}
    if r.returncode != 0 or not med:
        cauda = texto.splitlines()[-15:]
        raise Falha(f"medida de area falhou; registro em {rel}\n  "
                    + "\n  ".join(cauda))

    def num(v):
        try:
            return int(v)
        except ValueError:
            return float(v)
    med = {k: num(v) for k, v in med.items()}
    (RAIZ / "results/gen" / ctx["nome"] / "recursos.json").write_text(
        json.dumps(med, indent=1))
    ctx["recursos"] = med

    if "fmax_mhz" in med:
        print(f"      clock     {med['fmax_mhz']:.1f} MHz "
              f"(wns {med.get('wns_ns', 0):+.3f} ns no periodo pedido)")
    teto = {"lut": 53200, "bram": 139, "dsp": 220}
    for k, lim in teto.items():
        v = med.get(k, 0)
        print(f"      {k:<9} {v} de {lim} ({100.0*v/lim:.0f}%)")
        if v > lim:
            raise Falha(f"o acelerador nao cabe na peca: {v} {k} contra {lim} "
                        f"disponiveis. Reduza a topologia ou o orcamento.")

def sintetiza(a, ctx, saida: Path) -> None:
    ctx["fontes"] = digest_das_fontes(ctx["nome"])
    if reaproveita(a, ctx, saida):
        return
    env = dict(os.environ)
    rel = saida / "sintetiza.log"
    clk = a.clock
    realizados = []

    for fator in (1.0, 0.97, 0.90, 0.80, 0.70):
        if fator != 1.0:
            clk = int(alcancado * fator)
        rel.parent.mkdir(parents=True, exist_ok=True)
        with rel.open("w") as fh:
            r = subprocess.run(
                ["bash", "-lc",
                 f"source {a.vivado}/settings64.sh && "
                 f"vivado -mode batch -nojournal -nolog -notrace "
                 f"-source scripts/bitstream.tcl -tclargs {clk} {ctx['nome']}"],
                stdout=fh, stderr=subprocess.STDOUT, cwd=RAIZ, env=env)
        texto = rel.read_text()
        resumo = {ln.split()[1]: " ".join(ln.split()[2:])
                  for ln in texto.splitlines() if ln.startswith("RESUMO")}

        if r.returncode == 0:
            end = {ln.split()[2]: ln.split()[3] for ln in texto.splitlines()
                   if ln.startswith("RESUMO endereco")}
            if end:
                (saida / "enderecos.json").write_text(json.dumps(end, indent=2))
                ctx["enderecos"] = end
            for k, v in resumo.items():
                if k != "endereco":
                    print(f"      {k:<9} {v}")
            for k, v in end.items():
                print(f"      {k:<22} {v}")
            real = round(float(resumo.get("clock", clk).split()[0]))
            ctx["clock"] = real
            (saida / "clock.txt").write_text(str(real))
            plano = json.loads(
                (RAIZ / "results/gen" / ctx["nome"] / "plano.json").read_text())
            (RAIZ / "results/vivado" / f"soc_{ctx['nome']}" / "sintese.json"
             ).write_text(json.dumps({"run": ctx["run"], "clock": real,
                                      "enderecos": end, "plano": plano,
                                      "fontes": ctx["fontes"]},
                                     indent=2))
            return

        if r.returncode != 2 or "fmax" not in resumo:
            cauda = texto.splitlines()[-15:]
            raise Falha(f"sintese falhou (codigo {r.returncode}); registro em "
                        f"{rel}\n  " + "\n  ".join(cauda))

        alvo = resumo.get("fmax_seguro", resumo["fmax"])
        alcancado = float(alvo.split()[0])
        real = float(resumo.get("clock", clk).split()[0])
        if realizados and real >= realizados[-1]:
            alcancado = realizados[-1] * 0.95
        realizados.append(real)
        print(f"      {resumo.get('fecha','nao')} em {real:.1f} MHz "
              f"(folga {resumo.get('wns')}, exigido "
              f"{resumo.get('margem_pedida','?')}); com margem alcanca "
              f"{alvo} - tentando mais baixo", flush=True)

    raise Falha(f"nao fechou tempo em nenhuma frequencia tentada: {realizados}")

def confere_prazo(a, ctx, plano, clock_real: float) -> None:
    if not a.taxa:
        return
    disp = clock_real * 1e6 / a.taxa
    ii = plano["ii"]
    print(f"      prazo: {a.taxa:.0f} janelas/s a {clock_real:.1f} MHz = "
          f"{disp:.0f} ciclos; o circuito leva {ii} "
          f"({100.0 * ii / disp:.1f}%)")
    if ii > disp:
        raise Falha(
            f"a sintese fechou em {clock_real:.1f} MHz e nesse clock o "
            f"circuito NAO cumpre a taxa de {a.taxa:.0f} janelas/s: "
            f"{ii} ciclos contra os {disp:.0f} do prazo.\n"
            f"  Peca uma taxa menor, ou mais orcamento, ou melhore o "
            f"caminho critico.")

def _clock(ctx, a, saida: Path) -> int:
    if "clock" in ctx:
        return ctx["clock"]
    f = saida / "clock.txt"
    if f.exists():
        return int(f.read_text().strip())
    return a.clock

def le_carimbo(pesos_h: Path) -> dict:
    texto = pesos_h.read_text()
    def val(chave):
        for ln in texto.splitlines():
            if ln.startswith(f"#define {chave}"):
                return ln.split('"')[1]
        return ""
    return {"run": val("RUN_ID"), "sha": val("PESOS_SHA")}

def _enderecos(ctx, saida: Path) -> dict:
    if "enderecos" in ctx:
        return ctx["enderecos"]
    f = saida / "enderecos.json"
    return json.loads(f.read_text()) if f.exists() else {}

def sistema(a, ctx, saida: Path) -> None:
    end = _enderecos(ctx, saida)
    base = end.get("SEG_acel_reg0", a.base)
    base_dma = end.get("SEG_dma_Reg", a.base_dma)

    roda([sys.executable, "-m", "ferramentas.gera_pesos_h",
          "--dir", str(ctx["hw_data"]), "--run", ctx["run"],
          "--base", base, "--base-dma", base_dma,
          "--buf-dma", a.buf_dma, "--clk", str(_clock(ctx, a, saida)),
          "--plano", str(RAIZ / "results/gen" / ctx["nome"] / "plano.json")],
         saida / "header.log")
    print(f"      acelerador em {base}, DMA em {base_dma}, "
          f"janela em {a.buf_dma}")
    carimbo = le_carimbo(RAIZ / "ps" / "pesos.h")
    (saida / "carimbo.json").write_text(json.dumps(carimbo, indent=2))
    print(f"      carimbo da imagem: rodada {carimbo['run']}, "
          f"pesos {carimbo['sha']}")
    app = RAIZ / "petalinux/cnn_soc/project-spec/meta-user/recipes-apps/acelerador/files"
    for f in ("acelerador.c", "mapa_linux.c", "inferencia_sw.c", "fluxo_sw.c",
              "pacote.c", "pacote.h", "pesos.h"):
        shutil.copy(RAIZ / "ps" / f, app / f)

    dt = RAIZ / "petalinux/cnn_soc/project-spec/meta-user/recipes-bsp/device-tree/files"
    shutil.copy(RAIZ / "ps" / "system-user.dtsi", dt / "system-user.dtsi")

    shutil.copy(ctx["hw_data"] / "entrada_ps.bin", app / "entrada_ps.bin")
    xsa = RAIZ / "results/vivado" / f"soc_{ctx['nome']}" / "sistema.xsa"
    alvo = RAIZ / "petalinux/cnn_soc/project-spec/hw-description/system.xsa"
    if not xsa.exists():
        raise Falha(f"a sintese nao exportou a plataforma em {xsa}")
    reimporta = (not alvo.exists()
                 or xsa.stat().st_mtime > alvo.stat().st_mtime)
    if reimporta:
        print("      hardware novo: reimportando a plataforma no PetaLinux")
        roda(["bash", "-lc",
              f"cd {RAIZ}/petalinux/cnn_soc && "
              f"source {a.petalinux}/settings.sh >/dev/null 2>&1 && "
              f"petalinux-config --silentconfig "
              f"--get-hw-description={xsa.parent}"],
             saida / "hw_import.log")

    roda(["bash", "-lc",
          f"cd {RAIZ}/petalinux/cnn_soc && "
          f"source {a.petalinux}/settings.sh >/dev/null 2>&1 && petalinux-build"],
         saida / "sistema.log")
    print("      imagem montada com a aplicacao embutida")

def placa(a, ctx, saida: Path) -> None:
    res = saida / "placa.txt"
    prazo_us = f"{1e6 / a.taxa:.3f}" if a.taxa else "0"
    roda(["bash", "scripts/na_placa.sh", str(res), a.vivado, a.petalinux,
      ctx["nome"], prazo_us],
         saida / "placa.log")
    texto = res.read_text()
    for linha in texto.splitlines():
        if any(k in linha for k in ("acuracia", "concorda", "computo na PL",
                                    "ponta a ponta", "fracao de barramento",
                                    "vazao", "ganho")):
            print("      " + linha.strip())
    ctx["placa"] = texto
    carimbo = saida / "carimbo.json"
    confere_silicio(texto, _clock(ctx, a, saida),
                    json.loads(carimbo.read_text()) if carimbo.exists() else None,
                    fluxo=a.fluxo)

def confere_silicio(texto: str, clock: int | None = None,
                    carimbo: dict | None = None, fluxo: bool = False) -> None:
    from ferramentas import metricas

    m = metricas.da_placa(texto)
    erros = []

    if carimbo:
        linha = next((l for l in texto.splitlines()
                      if l.startswith("rodada:")), "")
        if carimbo["sha"] not in linha or carimbo["run"] not in linha:
            erros.append(
                f"a placa respondeu '{linha.strip() or 'sem carimbo'}' e esta "
                f"rodada e' {carimbo['run']} com pesos {carimbo['sha']}: a "
                f"imagem que rodou nao e' a que o fluxo acabou de gerar")

    aferido = m.get("pl_mhz_aferido")
    if aferido is None:
        erros.append("a placa nao reportou o relogio da PL aferido")
    elif clock and aferido > clock * 1.02:
        erros.append(f"a PL esta' a {aferido:.1f} MHz e a sintese fechou em "
                     f"{clock} MHz ({100*(aferido/clock-1):+.1f}%): o fclk0 "
                     f"nao segue o projeto")
    if not m.get("tempo_real"):
        erros.append("a placa nao reportou o ajuste de tempo real; as medidas "
                     "de prazo do sistema nao valem como garantia")
    if fluxo:
        perd = m.get("decisoes_perdidas")
        if perd is None:
            erros.append("a placa nao reportou decisoes perdidas pela fila")
        elif perd:
            erros.append(f"{perd} decisoes perdidas por fila cheia: a medida "
                         f"nao vale")
        if m.get("fluxo_saltos"):
            erros.append(f"{m['fluxo_saltos']} saltos na sequencia das "
                         f"decisoes: {m.get('fluxo_perdidas_seq')} perdidas "
                         f"entre a fila e o processador")
        jit = m.get("fluxo_iv_jitter")
        if jit is None or "fluxo_ciclos_parada" not in m:
            erros.append("a placa nao reportou o intervalo entre decisoes e as "
                         "paradas de entrada contados na PL: sem os dois o "
                         "determinismo e' presumido, e um intervalo maior que o "
                         "derivado fica sem origem atribuida")
        elif jit and not m.get("fluxo_ciclos_parada"):
            erros.append(f"o intervalo entre decisoes variou {jit} ciclos sem "
                         f"nenhuma parada de entrada: a garantia de taxa nao "
                         f"vale")
    else:
        ig, conf = m.get("logits_iguais"), m.get("logits_conferidos")
        if conf is None:
            erros.append("a placa nao reportou comparacao de logits com o ARM")
        elif ig != conf:
            erros.append(f"logits diferentes do software em {conf - ig} de "
                         f"{conf} janelas conferidas")
    if m.get("concorda_pct", 0.0) < 100.0:
        erros.append(f"classificacao concorda com o modelo em apenas "
                     f"{m.get('concorda_pct')}% das janelas")
    if m.get("estouro_acumulador", 0):
        erros.append(f"{m['estouro_acumulador']} janelas com estouro de acumulador")
    if erros:
        raise Falha("o hardware nao reproduz o modelo:\n  - " + "\n  - ".join(erros)
                    + "\n  Nao trate os numeros desta rodada como resultado.")
    print("      confere com o modelo: classificacao e logits identicos")

def relatorio(a, ctx, plano, saida: Path) -> None:
    from ferramentas import metricas

    res = saida / "placa.txt"
    if "placa" not in ctx and res.exists():
        ctx["placa"] = res.read_text()
    prev = saida / "latencia_prevista.txt"
    if "latencia_prevista" not in ctx and prev.exists():
        ctx["latencia_prevista"] = int(prev.read_text().strip())
    ctx["taxa"] = a.taxa
    dir_rpt = RAIZ / "results" / "vivado" / f"soc_{ctx['nome']}"
    m = metricas.consolida(plano, ctx, dir_rpt)
    metricas.escreve(m, saida)
    metricas.imprime(m)

    mv = m.get("modelo_vs_medido", {})
    if mv.get("recursos") and not mv.get("recursos_batem"):
        fora = ", ".join(f"{k}: previsto {v['previsto']}, medido {v['medido']}"
                         for k, v in mv["recursos"].items() if not v["bate"])
        raise Falha(f"o recurso previsto nao bate com a sintese ({fora}).\n"
                    f"  Conserte a derivacao do planejador; nao calibre com "
                    f"fator.")
    if mv.get("excesso_atribuido_a_parada_de_entrada") is False:
        raise Falha(
            f"o circuito gastou {mv['ciclos_por_amostra_medido']} ciclos por "
            f"amostra contra {mv['ciclos_por_amostra_derivado']} derivados, e "
            f"as paradas de entrada medidas cobrem apenas "
            f"{mv.get('ciclos_de_parada_por_amostra')} deles.\n  O excesso "
            f"restante e' do circuito, e a derivacao esta' errada.")

    faltando = [k for k in ("area", "frequencia", "potencia") if not m[k]]
    if faltando:
        print(f"\n      sem {', '.join(faltando)}: relatorios do Vivado nao"
              f" encontrados em {dir_rpt}")
    if not m["correcao"]:
        print("      sem medidas da placa: a etapa 'placa' nao rodou")
    print(f"\n      consolidado em {saida}/metricas.json e metricas.csv")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="da rede treinada ate a placa, num comando")
    ap.add_argument("--spec", type=Path, default=None,
                    help="config de treino, so' para reproduzir a preparacao "
                         "dos dados; a topologia vem do modelo. O padrao e' a "
                         "config gravada junto do modelo pelo modelo.prepara")
    ap.add_argument("--modelo", type=Path,
                    help="modelo treinado: grafo da topologia mais parametros")
    ap.add_argument("--run", default=None,
                    help="pasta em runs/ com o modelo treinado")
    ap.add_argument("--dados", type=Path, default=Path("data/full"))

    ap.add_argument("--n-teste", type=int, default=0,
                    help="janelas levadas para a placa; 0 e' o conjunto de "
                         "teste inteiro, que e' o que produz acuracia")
    ap.add_argument("--orcamento", type=int, default=220,
                    help="multiplicadores disponiveis no chip")
    ap.add_argument("--taxa", type=float, default=None, metavar="JANELAS/S",
                    help="taxa que o circuito tem de sustentar. Com isto o "
                         "dimensionamento inverte: o prazo entra e o hardware "
                         "sai, e o fluxo recusa antes de sintetizar se nao "
                         "couber")
    ap.add_argument("--clock", type=int, default=133, help="clock da PL em MHz")
    ap.add_argument("--fluxo", action="store_true",
                    help="deriva o acelerador em fluxo continuo em vez de por "
                         "janela: o motor nunca para e nao ha' recomputo entre "
                         "decisoes. Exige cabeca gap")
    ap.add_argument("--refaz-sintese", action="store_true",
                    help="sintetiza de novo mesmo com bitstream valido para "
                         "este plano")
    ap.add_argument("--base", default="0x40000000")
    ap.add_argument("--base-dma", default="0x40400000")
    ap.add_argument("--buf-dma", default="0x1FF00000",
                    help="area reservada em ps/system-user.dtsi")
    ap.add_argument("--vivado", default="/home/caiocv/2025.2/Vivado")
    ap.add_argument("--petalinux", default="/home/caiocv/petalinux/2025.2")
    ap.add_argument("--ate", choices=ETAPAS, default="relatorio",
                    help="para depois desta etapa")
    ap.add_argument("--de", choices=ETAPAS, default=None,
                    help="comeca nesta etapa, reaproveitando o que ja existe")
    a = ap.parse_args()

    de = a.de or "valida"
    alvo = ETAPAS.index(a.ate)
    inicio = ETAPAS.index(de)
    if inicio > alvo:
        print(f"--de {de} vem depois de --ate {a.ate}", file=sys.stderr)
        return 2
    fila = [e for i, e in enumerate(ETAPAS)
            if inicio <= i <= alvo and e != "valida"]
    base = RAIZ / "results/fluxo"
    base.mkdir(parents=True, exist_ok=True)
    saida = base

    print("=" * 62)
    print(f"acelera - {a.modelo or a.run or 'topologia pedida na linha de comando'}")
    print("=" * 62)

    t_total = time.time()
    ctx, plano = None, None
    n = 0
    try:
        ctx = valida(a)
        saida = base / (ctx["run"] + ctx.get("sufixo", ""))
        saida.mkdir(parents=True, exist_ok=True)
        if (base / "treina.log").exists():
            shutil.move(base / "treina.log", saida / "treina.log")

        for etapa in fila:
            n += 1
            t = passo(n, len(fila), TEXTOS[etapa])
            if etapa == "quantiza":
                quantiza(a, ctx, saida)
            elif etapa == "modela":
                plano = modela(a, ctx, saida)
            elif etapa == "confere":
                confere(a, ctx, saida)
            elif etapa == "recursos":
                recursos(a, ctx, saida)
            elif etapa == "sintetiza":
                sintetiza(a, ctx, saida)
                confere_prazo(a, ctx, plano or json.loads(
                    (RAIZ / "results/gen" / ctx["nome"] / "plano.json").read_text()),
                    _clock(ctx, a, saida))
            elif etapa == "sistema":
                sistema(a, ctx, saida)
            elif etapa == "placa":
                placa(a, ctx, saida)
            elif etapa == "relatorio":
                if plano is None:
                    plano = json.loads(
                        (RAIZ / "results/gen" / ctx["nome"] / "plano.json").read_text())
                relatorio(a, ctx, plano, saida)
            fim(t)
    except Falha as e:
        print(f"\nPAROU: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrompido", file=sys.stderr)
        return 130

    print(f"\nconcluido em {time.time()-t_total:.0f}s - resultados em {saida}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
