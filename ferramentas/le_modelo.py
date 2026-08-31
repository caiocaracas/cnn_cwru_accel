"""le o modelo treinado e separa o grafo dos parametros."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EIXOS = ("num_layers", "num_filters_first", "kernel_size", "pool_type",
         "head", "input_len")
DOMINIO = {
    "num_layers":        (2, 3, 4),
    "num_filters_first": (8, 16, 32),
    "kernel_size":       (3, 5, 7),
    "pool_type":         ("max", "avg", "none"),
    "head":              ("flatten", "gap"),
    "input_len":         (128, 256, 512, 1024, 2048),
}
PADRAO = {"head": "flatten", "input_len": 1024}

class ModeloInvalido(Exception):
    pass

@dataclass
class Controlador:
    num_layers: int
    num_filters_first: int
    kernel_size: int
    pool_type: str
    head: str = "flatten"
    input_len: int = 1024

    @property
    def nome(self) -> str:
        return (f"L{self.num_layers}_F{self.num_filters_first:02d}"
                f"_K{self.kernel_size}_P{self.pool_type}_H{self.head}"
                + ("" if self.input_len == 1024 else f"_W{self.input_len}"))

    def como_dict(self) -> dict:
        return {k: getattr(self, k) for k in EIXOS}

@dataclass
class CaminhoDados:
    camadas: list = field(default_factory=list)

    @property
    def n_pesos(self) -> int:
        return int(sum(c[1].size for c in self.camadas))

    @property
    def n_bias(self) -> int:
        return int(sum(c[3].size for c in self.camadas))

def _ordena(prefixos: list[str]) -> list[str]:
    def chave(p):
        return (0, int(p.split(".")[1])) if p.startswith("features.") else (1, 0)
    return sorted(prefixos, key=chave)

def le(caminho: Path) -> tuple[Controlador, CaminhoDados]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise ModeloInvalido(f"modelo nao encontrado: {caminho}")
    z = np.load(caminho)

    if "grafo" not in z.files:
        raise ModeloInvalido(
            f"{caminho} nao carrega o grafo da topologia.\n"
            f"  Pooling nao tem parametro, entao nao da' para inferir dos "
            f"tensores.\n"
            f"  Reexporte com: python3 -m modelo.quantiza_pesos "
            f"--export-weights ...")
    grafo = json.loads(bytes(z["grafo"]).decode())

    grafo = {**PADRAO, **grafo}
    faltando = [k for k in EIXOS if k not in grafo]
    if faltando:
        raise ModeloInvalido(f"grafo incompleto, falta: {faltando}")
    fora = [f"{k}={grafo[k]} fora de {DOMINIO[k]}"
            for k in EIXOS if grafo[k] not in DOMINIO[k]]
    if fora:
        raise ModeloInvalido("topologia fora do dominio suportado:\n  - "
                             + "\n  - ".join(fora))

    ctrl = Controlador(**{k: grafo[k] for k in EIXOS})

    prefixos = _ordena([k[:-len(".weight_int")] for k in z.files
                        if k.endswith(".weight_int")])
    dados = CaminhoDados()
    for p in prefixos:
        dados.camadas.append(
            (p, z[f"{p}.weight_int"],
             np.atleast_1d(np.asarray(z[f"{p}.weight_scale"], dtype=np.float64)),
             z[f"{p}.bias_int"], float(z[f"{p}.bias_scale"])))

    confere_coerencia(ctrl, dados)
    return ctrl, dados

def confere_coerencia(ctrl: Controlador, dados: CaminhoDados) -> None:
    from ferramentas.gerador import geometria

    cams, nflat = geometria(ctrl.num_layers, ctrl.num_filters_first,
                            ctrl.kernel_size, ctrl.pool_type, ctrl.head)
    convs = [c for c in dados.camadas if c[0].startswith("features.")]
    fc = [c for c in dados.camadas if not c[0].startswith("features.")]

    erros = []
    if len(convs) != len(cams):
        erros.append(f"grafo pede {len(cams)} convolucoes, o modelo tem {len(convs)}")
    for (nome, w, _, b, _), c in zip(convs, cams):
        if tuple(w.shape) != (c["nof"], c["nif"], c["k"]):
            erros.append(f"{c['nome']}: tensor {tuple(w.shape)}, "
                         f"grafo {(c['nof'], c['nif'], c['k'])}")
        if b.size != c["nof"]:
            erros.append(f"{c['nome']}: {b.size} bias, grafo pede {c['nof']}")
    if fc and fc[0][1].shape[1] != nflat:
        erros.append(f"camada final: entra com {fc[0][1].shape[1]}, "
                     f"grafo pede {nflat}")
    if erros:
        raise ModeloInvalido("o grafo e os parametros discordam:\n  - "
                             + "\n  - ".join(erros))

def descreve(ctrl: Controlador, dados: CaminhoDados) -> str:
    linhas = [f"controlador   {ctrl.num_layers} camadas, "
              f"{ctrl.num_filters_first} filtros iniciais, "
              f"kernel {ctrl.kernel_size}, pooling {ctrl.pool_type}",
              f"caminho dados {dados.n_pesos} pesos, {dados.n_bias} bias, "
              f"{len(dados.camadas)} memorias"]
    for nome, w, ws, b, bs in dados.camadas:
        linhas.append(f"              {nome:14s} {str(tuple(w.shape)):16s} "
                      f"escala {ws:.4e}")
    return "\n".join(linhas)

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", type=Path)
    a = ap.parse_args()
    ctrl, dados = le(a.modelo)
    print(descreve(ctrl, dados))
    print(f"identificador {ctrl.nome}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
