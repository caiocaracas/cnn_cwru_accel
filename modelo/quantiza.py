"""quantiza o modelo e emite pesos escalas e vetores para o hardware."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from modelo.cwru import (ingest_directory, CLASS_NAMES,
                         janela_int8, para_int8, zscore_per_window)
from modelo.rede import ModelConfig, build_model, geometria_saida
from modelo.treina import TrainConfig, make_splits

def resolve_layer_order(npz) -> list[tuple[str, str]]:

    prefixes = []
    for k in npz.files:
        if k.endswith(".weight_int"):
            prefixes.append(k[: -len(".weight_int")])

    def sort_key(p):
        if p.startswith("features."):
            return (0, int(p.split(".")[1]))
        return (1, 0)

    prefixes.sort(key=sort_key)
    saida, n = [], 0
    for pref in prefixes:
        if pref.startswith("features."):
            n += 1
            saida.append((pref, f"conv{n}"))
        else:
            saida.append((pref, "fc"))
    return saida

def to_twos_complement(val: int, bits: int) -> int:
    if val < 0:
        val = (1 << bits) + val
    return val & ((1 << bits) - 1)

def write_coe(values: np.ndarray, bits: int, out_path: Path) -> None:
    flat = values.flatten()
    hex_digits = (bits + 3) // 4

    lines = ["memory_initialization_radix=16;", "memory_initialization_vector="]
    for i, v in enumerate(flat):
        tc = to_twos_complement(int(v), bits)
        sep = ";" if i == len(flat) - 1 else ","
        lines.append(f"{tc:0{hex_digits}X}{sep}")

    out_path.write_text("\n".join(lines) + "\n")

def write_mem(values: np.ndarray, bits: int, out_path: Path) -> None:
    flat = values.flatten()
    hex_digits = (bits + 3) // 4
    lines = [f"{to_twos_complement(int(v), bits):0{hex_digits}X}" for v in flat]
    out_path.write_text("\n".join(lines) + "\n")

def make_quantized_forward(model, npz, layer_order):

    raise NotImplementedError("usar capture_layer_activations")

def escala_peso(npz, pt: str) -> np.ndarray:
    return np.atleast_1d(np.asarray(npz[f"{pt}.weight_scale"], dtype=np.float64))

def _col(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return v.reshape(-1, 1) if v.size > 1 else v.reshape(())

def bias_in_acc(npz, pt: str, acc_scale) -> np.ndarray:
    b_int = npz[f"{pt}.bias_int"]
    b_scale = float(npz[f"{pt}.bias_scale"])
    return np.round(b_int.astype(np.float64) * b_scale / acc_scale).astype(np.int64)

def _conv1d_int_ref(x_int, w_int, b_int, stride=1, pad_same=True):
    out_ch, in_ch, k = w_int.shape
    length = x_int.shape[1]
    pad = (k - 1) // 2 if pad_same else 0
    x_pad = np.zeros((in_ch, length + 2 * pad), dtype=np.int64)
    x_pad[:, pad:pad + length] = x_int
    out = np.zeros((out_ch, length), dtype=np.int64)
    for oc in range(out_ch):
        for pos in range(length):
            acc = np.int64(0)
            for ic in range(in_ch):
                for kk in range(k):
                    acc += np.int64(x_pad[ic, pos + kk]) * np.int64(w_int[oc, ic, kk])
            out[oc, pos] = acc + np.int64(b_int[oc])
    return out

_DISP = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv1d_int(x_int, w_int, b_int, stride=1, pad_same=True):
    k = w_int.shape[2]
    pad = (k - 1) // 2 if pad_same else 0
    x = torch.as_tensor(np.ascontiguousarray(x_int), dtype=torch.float64,
                        device=_DISP).unsqueeze(0)
    w = torch.as_tensor(np.ascontiguousarray(w_int), dtype=torch.float64, device=_DISP)
    b = torch.as_tensor(np.ascontiguousarray(b_int), dtype=torch.float64, device=_DISP)
    out = torch.nn.functional.conv1d(x, w, b, stride=stride, padding=pad)
    return out.squeeze(0).round().to(torch.int64).cpu().numpy()

def relu_int(x):
    return np.maximum(x, 0)

def maxpool_int(x, pool=2):
    ch, length = x.shape
    length_out = length // pool
    x = x[:, :length_out * pool]
    return x.reshape(ch, length_out, pool).max(axis=2)

def avgpool_int(x, pool=2):
    ch, length = x.shape
    length_out = length // pool
    x = x[:, :length_out * pool]
    soma = x.reshape(ch, length_out, pool).sum(axis=2)
    return soma >> int(np.log2(pool))

def pool_int(x, pool=2, modo="max"):
    if pool <= 1:
        return x
    return maxpool_int(x, pool) if modo == "max" else avgpool_int(x, pool)

def para_json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"{type(o).__name__} nao vai para JSON")

def requantize(acc, mult, shift, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    acc = acc.astype(np.int64)
    m = np.asarray(mult, dtype=np.int64)
    if m.size > 1:
        if m.size != acc.shape[0]:
            raise ValueError(f"{m.size} multiplicadores para "
                             f"{acc.shape[0]} canais")
        m = m.reshape(-1, *([1] * (acc.ndim - 1)))
    round_offset = np.int64(1) << (shift - 1)
    q = (acc * m + round_offset) >> np.int64(shift)
    return np.clip(q, qmin, qmax).astype(np.int64)

MULT_W = 18

CALIB_METODOS = ("max", "p99.9", "p99.99", "p99.999", "entropia")
_HIST_BINS = 2048

def _limiar_percentil(hist, edges, pct):
    acum = np.cumsum(hist.astype(np.float64))
    total = acum[-1]
    if total <= 0:
        return float(edges[-1])
    alvo = total * pct / 100.0
    i = int(np.searchsorted(acum, alvo))
    return float(edges[min(i + 1, len(edges) - 1)])

def _limiar_entropia(hist, edges, n_niveis=128):
    hist = hist.astype(np.float64).copy()
    if len(hist) > 1:
        hist[0] = hist[1]
    nbins = len(hist)
    melhor_kl, melhor_i = np.inf, nbins
    for i in range(n_niveis, nbins + 1):
        p = hist[:i].copy()
        p[-1] += hist[i:].sum()
        soma_p = p.sum()
        if soma_p <= 0:
            continue
        p /= soma_p

        bordas = np.linspace(0, i, n_niveis + 1).astype(int)
        q = np.zeros(i)
        for j in range(n_niveis):
            a, b = bordas[j], bordas[j + 1]
            if a >= b:
                continue
            faixa = hist[a:b]
            nz = np.count_nonzero(faixa)
            if nz:
                q[a:b] = np.where(faixa != 0, faixa.sum() / nz, 0.0)
        soma_q = q.sum()
        if soma_q <= 0:
            continue
        q /= soma_q

        m = p > 0
        kl = float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-12))))
        if kl < melhor_kl:
            melhor_kl, melhor_i = kl, i
    return float(edges[melhor_i])

def _limiar(hist, edges, metodo):
    if metodo == "max":
        return float(edges[-1])
    if metodo.startswith("p"):
        return _limiar_percentil(hist, edges, float(metodo[1:]))
    if metodo == "entropia":
        return _limiar_entropia(hist, edges)
    raise ValueError(f"metodo de calibracao desconhecido: {metodo}")

def calibrate_out_scales(X_cal_int8, npz, layer_order, input_scale,
                         metodo="entropia", pool=2, modo="max"):
    if metodo not in CALIB_METODOS:
        raise ValueError(f"metodo {metodo} fora de {CALIB_METODOS}")
    conv_layers = [(pt, hw) for pt, hw in layer_order if hw.startswith("conv")]

    out_scales = {}
    prev_scale = input_scale
    for li, (pt, hw) in enumerate(conv_layers):
        w_int = npz[f"{pt}.weight_int"]
        w_scale = escala_peso(npz, pt)
        b_int = npz[f"{pt}.bias_int"]
        b_scale = float(npz[f"{pt}.bias_scale"])
        acc_scale = prev_scale * w_scale
        b_in_acc = np.round(b_int.astype(np.float64) * b_scale / acc_scale).astype(np.int64)

        saidas = []
        for x_int8 in X_cal_int8:
            x = _forward_until(x_int8, conv_layers[:li], npz, input_scale,
                               out_scales, pool, modo)
            acc = pool_int(relu_int(conv1d_int(x, w_int, b_in_acc)), pool, modo)
            saidas.append(np.abs(acc.astype(np.float64) * _col(acc_scale)).ravel())

        maxabs = max(float(s.max()) for s in saidas)
        if maxabs <= 0:
            out_scales[hw] = 1.0
            prev_scale = 1.0
            continue

        if metodo == "max":
            limiar = maxabs
        else:
            edges = np.linspace(0.0, maxabs, _HIST_BINS + 1)
            hist = np.zeros(_HIST_BINS, dtype=np.int64)
            for s in saidas:
                hist += np.histogram(s, bins=edges)[0]
            limiar = _limiar(hist, edges, metodo)

        out_scales[hw] = limiar / 127 if limiar > 0 else 1.0
        prev_scale = out_scales[hw]

    return out_scales

def _forward_until(x_int8, conv_layers_prefix, npz, input_scale, out_scales,
                   pool=2, modo="max"):
    x = x_int8.reshape(1, -1).astype(np.int64)
    cur_scale = input_scale
    for pt, hw in conv_layers_prefix:
        w_int = npz[f"{pt}.weight_int"]
        w_scale = escala_peso(npz, pt)
        b_int = npz[f"{pt}.bias_int"]
        b_scale = float(npz[f"{pt}.bias_scale"])
        acc_scale = cur_scale * w_scale
        b_in_acc = np.round(b_int.astype(np.float64) * b_scale / acc_scale).astype(np.int64)
        acc = pool_int(relu_int(conv1d_int(x, w_int, b_in_acc)), pool, modo)
        out_scale = out_scales[hw]
        mult = np.round((acc_scale / out_scale) * (1 << 16)).astype(np.int64)
        x = requantize(acc, mult, 16)
        cur_scale = out_scale
    return x

def compute_requant_params(input_scale, npz, layer_order, out_scales, frac_bits=16):
    conv_layers = [(pt, hw) for pt, hw in layer_order if hw.startswith("conv")]
    params = {}
    cur_scale = input_scale
    for pt, hw in conv_layers:
        w_scale = escala_peso(npz, pt)
        acc_scale = cur_scale * w_scale
        out_scale = out_scales[hw]
        M = acc_scale / out_scale
        shift = frac_bits
        mult = np.round(M * (1 << shift)).astype(np.int64)
        limite = (1 << (MULT_W - 1)) - 1
        if np.any(mult > limite):
            raise ValueError(
                f"{hw}: multiplicador de requantizacao {int(mult.max())} nao "
                f"cabe em {MULT_W} bits com sinal (limite {limite}). "
                f"Jacob et al. supoe M em (0,1); aqui M={float(M.max()):.4f}")
        params[hw] = {"mult": mult, "shift": shift,
                      "acc_scale": acc_scale, "out_scale": out_scale,
                      "M_real": M, "M_fixed": mult / (1 << shift)}
        cur_scale = out_scale
    return params

def bias_fc_em_acc(npz, pt: str, acc_scale: float, npos: int, head: str):
    b_int = npz[f"{pt}.bias_int"].astype(np.float64)
    b_scale = float(npz[f"{pt}.bias_scale"])
    fator = npos if head == "gap" else 1
    return np.round(b_int * b_scale * fator / acc_scale).astype(np.int64)

def capture_layer_activations(x_int8, npz, layer_order, input_scale,
                              out_scales, pool=2, modo="max", head="flatten"):
    activations = {}
    x = x_int8.reshape(1, -1).astype(np.int64)
    cur_scale = input_scale

    conv_layers = [(pt, hw) for pt, hw in layer_order if hw.startswith("conv")]
    fc_layers = [(pt, hw) for pt, hw in layer_order if hw == "fc"]

    for pt, hw in conv_layers:
        w_int = npz[f"{pt}.weight_int"]
        w_scale = escala_peso(npz, pt)
        b_int = npz[f"{pt}.bias_int"]
        b_scale = float(npz[f"{pt}.bias_scale"])

        acc_scale = cur_scale * w_scale
        b_in_acc = np.round(b_int.astype(np.float64) * b_scale / acc_scale).astype(np.int64)

        acc = conv1d_int(x, w_int, b_in_acc)
        acc = relu_int(acc)
        acc = pool_int(acc, pool, modo)

        out_scale = out_scales[hw]
        mult = np.round((acc_scale / out_scale) * (1 << 16)).astype(np.int64)
        x_int8_layer = requantize(acc, mult, 16, bits=8)
        activations[hw] = x_int8_layer.astype(np.int8)

        x = x_int8_layer
        cur_scale = out_scale

    npos = x.shape[1]
    x_flat = x.flatten().astype(np.int64)
    pt, hw = fc_layers[0]
    w_int = npz[f"{pt}.weight_int"].astype(np.int64)
    w_scale = escala_peso(npz, pt)

    acc_scale = cur_scale * w_scale
    b_in_acc = bias_fc_em_acc(npz, pt, acc_scale, npos, head)

    logits = np.zeros(w_int.shape[0], dtype=np.int64)
    if head == "gap":
        soma_ch = x.astype(np.int64).sum(axis=1)
        for o in range(w_int.shape[0]):
            logits[o] = np.int64(np.sum(soma_ch * w_int[o])) + b_in_acc[o]
    else:
        for o in range(w_int.shape[0]):
            logits[o] = np.int64(np.sum(x_flat * w_int[o])) + b_in_acc[o]

    activations["fc"] = x_flat.astype(np.int8)
    activations["logits"] = logits
    activations["classe"] = int(np.argmax(logits))

    return activations

def decisoes_em_fluxo(x_int8, npz, layer_order, input_scale, out_scales,
                      pool=2, modo="max", npg=1, npos=None):
    x = np.asarray(x_int8, dtype=np.int64).reshape(1, -1)
    cur_scale = input_scale
    conv_layers = [(pt, hw) for pt, hw in layer_order if hw.startswith("conv")]
    fc_layers = [(pt, hw) for pt, hw in layer_order if hw == "fc"]

    for pt, hw in conv_layers:
        w_int = npz[f"{pt}.weight_int"]
        w_scale = escala_peso(npz, pt)
        b_int = npz[f"{pt}.bias_int"]
        b_scale = float(npz[f"{pt}.bias_scale"])
        acc_scale = cur_scale * w_scale
        b_in_acc = np.round(b_int.astype(np.float64) * b_scale / acc_scale).astype(np.int64)
        acc = pool_int(relu_int(conv1d_int(x, w_int, b_in_acc)), pool, modo)
        out_scale = out_scales[hw]
        mult = np.round((acc_scale / out_scale) * (1 << 16)).astype(np.int64)
        x = requantize(acc, mult, 16, bits=8).astype(np.int64)
        cur_scale = out_scale

    nch, ntot = x.shape
    pt, hw = fc_layers[0]
    w_int = npz[f"{pt}.weight_int"].astype(np.int64)
    acc_scale = cur_scale * escala_peso(npz, pt)
    b_in_acc = bias_fc_em_acc(npz, pt, acc_scale, npos, "gap")

    csum = np.cumsum(np.pad(x, ((0, 0), (1, 0))), axis=1)
    pos = np.arange(npg - 1, ntot, npg)
    ini = np.maximum(0, pos - npos + 1)
    S = csum[:, pos + 1] - csum[:, ini]
    logits = w_int @ S + b_in_acc[:, None]
    return logits.T, np.argmax(logits, axis=0).astype(np.int32), pos

PACOTE_FLUXO_VERSAO = 6

def escreve_pacote_fluxo(path: Path, fluxo_int8, gold, verdade,
                         input_scale: float) -> None:
    import struct
    s = np.asarray(fluxo_int8, dtype=np.int8).reshape(-1)
    g = np.asarray(gold, dtype="<i4")
    v = np.asarray(verdade, dtype="<i4")
    assert g.size == v.size
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", 0x434E4E58, s.size, g.size,
                            PACOTE_FLUXO_VERSAO))
        f.write(struct.pack("<d", float(input_scale)))
        f.write(s.tobytes())
        f.write(g.tobytes())
        f.write(v.tobytes())

def generate_weight_files(npz, layer_order, out_dir: Path, weight_bits=8) -> dict:
    info = {}
    for pt, hw in layer_order:
        w_int = npz[f"{pt}.weight_int"]
        write_coe(w_int, weight_bits, out_dir / f"pesos_{hw}.coe")
        write_mem(w_int, weight_bits, out_dir / f"pesos_{hw}.mem")
        info[hw] = {"weight_shape": list(w_int.shape), "weight_count": int(w_int.size)}
    return info

def bias_acc_chain(npz, layer_order, input_scale, out_scales,
                   npos: int = 1, head: str = "flatten") -> dict:
    result = {}
    cur = input_scale
    for pt, hw in layer_order:
        w_scale = escala_peso(npz, pt)
        acc_scale = cur * w_scale
        if hw == "fc":
            b_in_acc = bias_fc_em_acc(npz, pt, acc_scale, npos, head)
        else:
            b_in_acc = bias_in_acc(npz, pt, acc_scale)
        result[hw] = (b_in_acc, acc_scale)
        if hw.startswith("conv"):
            cur = out_scales[hw]
    return result

def generate_bias_files(bias_acc: dict, out_dir: Path) -> None:
    for hw, (b, _) in bias_acc.items():
        write_mem(b, 32, out_dir / f"bias_{hw}.mem")

def generate_manifest(npz, layer_order, model_cfg, input_scale, out_scales,
                      requant, bias_acc, n_vectors, out_dir: Path) -> dict:
    import math
    layers = []
    w_base, b_base, cur_len, cur_bank = 0, 0, model_cfg.input_len, 0
    total_w = total_b = max_acc_bits = act_bank_words = max_out_ch = 0

    for pt, hw in layer_order:
        w_int = npz[f"{pt}.weight_int"]
        b_words = int(bias_acc[hw][0].size)
        acc_scale = np.asarray(bias_acc[hw][1], dtype=np.float64)

        if hw.startswith("conv"):
            oc, ic, k = (int(v) for v in w_int.shape)
            out_len_conv = cur_len
            out_len = cur_len // 2
            in_words, out_words = ic * cur_len, oc * out_len
            acc_bits = math.ceil(math.log2(ic * k * 127 * 127)) + 1
            in_bank, out_bank = cur_bank, 1 - cur_bank
            rq = requant[hw]
            layers.append({
                "index": len(layers), "name": hw, "type": "conv",
                "in_ch": ic, "out_ch": oc, "kernel": k, "stride": 1, "pad": (k - 1) // 2,
                "in_len": cur_len, "out_len_conv": out_len_conv, "out_len": out_len,
                "pool": {"type": model_cfg.pool_type, "size": 2}, "relu": True,
                "requant": {"mult": np.atleast_1d(rq["mult"]).astype(int).tolist(),
                            "shift": int(rq["shift"]),
                            "por_canal": bool(np.size(rq["mult"]) > 1),
                            "acc_scale": np.atleast_1d(acc_scale).tolist(),
                            "out_scale": float(rq["out_scale"])},
                "acc_bits": acc_bits,
                "weights": {"base": w_base, "words": int(w_int.size),
                            "order": "oc,ic,k row-major", "bits": 8},
                "bias": {"base": b_base, "words": b_words, "bits": 32, "scale": "acc_scale"},
                "act": {"in_bank": in_bank, "out_bank": out_bank,
                        "in_words": in_words, "out_words": out_words},
                "gold_file": f"gold_{hw}.mem",
            })
            act_bank_words = max(act_bank_words, in_words, out_words)
            cur_len, cur_bank = out_len, out_bank
        else:
            oc, flat = (int(v) for v in w_int.shape)
            acc_bits = math.ceil(math.log2(flat * 127 * 127)) + 1
            layers.append({
                "index": len(layers), "name": hw, "type": "fc",
                "in_ch": flat, "out_ch": oc, "kernel": 1, "stride": 1, "pad": 0,
                "in_len": 1, "out_len_conv": 1, "out_len": 1,
                "pool": {"type": "none", "size": 1}, "relu": False,
                "requant": None, "acc_bits": acc_bits,
                "weights": {"base": w_base, "words": int(w_int.size),
                            "order": "class,flat(ch,pos) row-major", "bits": 8},
                "bias": {"base": b_base, "words": b_words, "bits": 32, "scale": "acc_scale"},
                "act": {"in_bank": cur_bank, "out_bank": None, "in_words": flat, "out_words": oc},
                "output": "logits INT32 -> classe = argmax", "gold_file": "gold_logits.mem",
            })

        max_acc_bits = max(max_acc_bits, acc_bits)
        max_out_ch = max(max_out_ch, int(w_int.shape[0]))
        total_w += int(w_int.size)
        total_b += b_words
        w_base += int(w_int.size)
        b_base += b_words

    manifest = {
        "manifest_version": 1,
        "generated_by": "gerador_dados.py",
        "network": {"num_layers": model_cfg.num_layers,
                    "num_filters_first": model_cfg.num_filters_first,
                    "kernel": model_cfg.kernel_size, "pool": model_cfg.pool_type,
                    "head": model_cfg.head,
                    "num_classes": 4, "input_len": model_cfg.input_len,
                    "input_ch": 1},
        "quant": {"weight_bits": 8, "act_bits": 8, "input_scale": input_scale,
                  "requant": "(acc*MULT + 2^(SHIFT-1)) >> SHIFT",
                  "round": "half-up", "saturate": [-128, 127]},
        "engine_requirements": {"kernel": model_cfg.kernel_size, "max_acc_bits": max_acc_bits,
                                "act_bank_words": act_bank_words, "total_weight_words": total_w,
                                "total_bias_words": total_b, "max_out_ch": max_out_ch},
        "memory_map": {"wmem_words": total_w, "bmem_words": total_b,
                       "act_bank_words": act_bank_words, "n_act_banks": 2},
        "layers": layers,
        "test": {"n_vectors": n_vectors, "input_file": "test_vectors_input.mem",
                 "classes_file": "gold_classes.txt"},
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=para_json)
    return manifest

PACOTE_VERSAO = 5

def escreve_pacote_ps(path: Path, janelas_int8: np.ndarray,
                      gold: list[int], input_scale: float,
                      verdade: np.ndarray | None = None) -> None:
    import struct
    j = np.asarray(janelas_int8, dtype=np.int8)
    n, w = j.shape
    if verdade is None:
        verdade = np.full(n, -1, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", 0x434E4E58, n, w, PACOTE_VERSAO))
        f.write(struct.pack("<d", float(input_scale)))
        f.write(j.tobytes())
        f.write(np.asarray(gold, dtype="<i4").tobytes())
        f.write(np.asarray(verdade, dtype="<i4").tobytes())

def preparar_dados(train_cfg, data_dir, janela=None, passo=None):
    from modelo.cwru import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE
    data = ingest_directory(data_dir,
                            window_size=janela or DEFAULT_WINDOW_SIZE,
                            stride=passo or DEFAULT_STRIDE)
    Xd = np.concatenate([data.X, data.X_bruto[:, None, :]], axis=1)
    tr, va, te = make_splits(Xd, data.y, train_cfg.seed,
                             groups=data.grupo, loads=data.load_hp,
                             protocol=train_cfg.protocol,
                             held_out_load=train_cfg.held_out_load)

    return {
        "X_tr": Xd[tr][:, 0:1, :].astype(np.float32),
        "X_tr_bru": Xd[tr][:, 1, :],
        "y_tr": data.y[tr],
        "X_va": Xd[va][:, 0:1, :].astype(np.float32),
        "X_va_bru": Xd[va][:, 1, :],
        "y_va": data.y[va],
        "X_te": Xd[te][:, 0:1, :].astype(np.float32),
        "X_te_bru": Xd[te][:, 1, :],
        "y_te": data.y[te],
    }

def conjunto_calibracao(X_bru, input_scale, n_calib, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n_calib, len(X_bru))
    idx = rng.choice(len(X_bru), n, replace=False)
    return list(janela_int8(X_bru[idx], input_scale))

def pooling_da_spec(model_cfg) -> tuple[int, str]:
    if model_cfg.pool_type == "none":
        return 1, "max"
    return 2, model_cfg.pool_type

def classes_de_int8(X_q, npz, layer_order, input_scale, out_scales,
                    pool=2, modo="max", head="flatten"):
    saida = np.empty(len(X_q), dtype=np.int32)
    for i in range(len(X_q)):
        acts = capture_layer_activations(X_q[i], npz, layer_order, input_scale,
                                         out_scales, pool, modo, head)
        saida[i] = int(acts["classe"])
    return saida

def generate_test_vectors(model_cfg, train_cfg, data_dir, npz, layer_order,
                          n_vectors, out_dir: Path, calib_metodo="entropia",
                          n_calib=512, dados=None, n_placa=0,
                          input_scale=None) -> dict:
    d = dados if dados is not None else preparar_dados(train_cfg, data_dir)
    X_te, X_te_bru, y_te = d["X_te"], d["X_te_bru"], d["y_te"]

    if input_scale is None:
        raise ValueError("a escala da entrada vem do modelo treinado; sem ela "
                         "a janela do pacote nao seria a que a rede viu")

    n = min(n_vectors, len(X_te))
    idxs = np.linspace(0, len(X_te) - 1, n).astype(int)
    X_int8 = list(janela_int8(X_te_bru[idxs], input_scale))

    pool, modo = pooling_da_spec(model_cfg)
    head = model_cfg.head
    X_cal = conjunto_calibracao(d["X_tr_bru"], input_scale,
                                n_calib, seed=train_cfg.seed)
    out_scales = calibrate_out_scales(X_cal, npz, layer_order, input_scale,
                                      metodo=calib_metodo, pool=pool, modo=modo)
    requant = compute_requant_params(input_scale, npz, layer_order, out_scales)

    correct = 0
    ranges = {hw: [np.inf, -np.inf] for _, hw in layer_order}
    conv_names = [hw for _, hw in layer_order if hw.startswith("conv")]
    classes = []

    parciais = {}

    def abre(nome: str):
        alvo = out_dir / nome
        tmp = alvo.with_name(alvo.name + ".parcial")
        parciais[tmp] = alvo
        return tmp.open("w")

    fh_logits = abre("gold_logits.mem")
    fh_classes = abre("gold_classes.txt")

    for vi, idx in enumerate(idxs):
        x_int8 = X_int8[vi]

        acts = capture_layer_activations(x_int8, npz, layer_order, input_scale,
                                         out_scales, pool, modo, head)

        pred = acts["classe"]
        true = int(y_te[idx])
        if pred == true:
            correct += 1

        for _, hw in layer_order:
            a = acts[hw]
            ranges[hw][0] = min(ranges[hw][0], int(a.min()))
            ranges[hw][1] = max(ranges[hw][1], int(a.max()))

        fh_logits.write(f"// vetor {vi} (classe {pred})\n")
        fh_logits.writelines(
            f"{to_twos_complement(int(v),32):08X}\n" for v in acts["logits"]
        )
        fh_classes.write(f"{pred}\n")
        classes.append(pred)

    for fh in (fh_logits, fh_classes):
        fh.close()
    for tmp, alvo in parciais.items():
        tmp.replace(alvo)
    n_pl = len(X_te) if n_placa <= 0 else min(n_placa, len(X_te))
    if n_pl <= len(idxs):
        sel = idxs
        cls_pl = np.asarray(classes, dtype=np.int32)
        jan_pl = np.stack(X_int8)
    else:
        sel = np.arange(n_pl)
        print(f"  classificando {n_pl} janelas para o pacote da placa...")
        jan_pl = janela_int8(X_te_bru[sel], input_scale)
        cls_pl = classes_de_int8(jan_pl, npz, layer_order, input_scale,
                                 out_scales, pool, modo, head)
    escreve_pacote_ps(out_dir / "entrada_ps.bin", jan_pl,
                      cls_pl, input_scale, verdade=y_te[sel])
    print(f"  pacote da placa: {n_pl} janelas, "
          f"{(out_dir / 'entrada_ps.bin').stat().st_size / 1e6:.1f} MB, "
          f"acuracia do modelo quantizado {(cls_pl == y_te[sel]).mean():.2%}")

    req_lines = ["// requant por canal de saida: out = clip((acc*MULT)>>SHIFT)"]
    for hw in conv_names:
        p = requant[hw]
        m = np.atleast_1d(p["mult"]).astype(np.int64)
        write_mem(m, MULT_W, out_dir / f"mult_{hw}.mem")
        mr = np.atleast_1d(p["M_real"])
        req_lines.append(
            f"{hw}: N={m.size} SHIFT={p['shift']} "
            f"MULT_MIN={int(m.min())} MULT_MAX={int(m.max())} "
            f"M_real=[{float(mr.min()):.6e},{float(mr.max()):.6e}] "
            f"out_scale={p['out_scale']:.6e}")
    (out_dir / "requant_params.txt").write_text("\n".join(req_lines) + "\n")

    return {
        "n_vectors": n,
        "accuracy_on_vectors": correct / n,
        "input_scale": input_scale,
        "layer_ranges": {hw: ranges[hw] for _, hw in layer_order},
        "out_scales": out_scales,
        "requant": requant,
    }

def comparar_calibracoes(train_cfg, data_dir, npz, layer_order, n_calib=512,
                         n_aval=1000, dados=None, pool=2, modo="max",
                         head="flatten", input_scale=None) -> dict:
    d = dados if dados is not None else preparar_dados(train_cfg, data_dir)
    y_va = d["y_va"]

    X_cal = conjunto_calibracao(d["X_tr_bru"], input_scale,
                                n_calib, seed=train_cfg.seed)

    rng = np.random.default_rng(train_cfg.seed)
    sel = rng.choice(len(d["X_va"]), min(n_aval, len(d["X_va"])), replace=False)
    Xe = janela_int8(d["X_va_bru"][sel], input_scale)
    ye = y_va[sel]

    res = {}
    for metodo in CALIB_METODOS:
        t0 = time.time()
        oc = calibrate_out_scales(X_cal, npz, layer_order, input_scale,
                                  metodo=metodo, pool=pool, modo=modo)
        cls = classes_de_int8(Xe, npz, layer_order, input_scale, oc, pool, modo, head)
        acc = float((cls == ye).mean())
        res[metodo] = {"acc_int8_val": acc, "out_scales": oc,
                       "segundos": time.time() - t0}
        print(f"  {metodo:10s}  acc_int8(val)={acc:.4f}  "
              f"out_scales={[f'{v:.4g}' for v in oc.values()]}")
    melhor = max(res, key=lambda k: res[k]["acc_int8_val"])
    print(f"  -> melhor: {melhor} ({res[melhor]['acc_int8_val']:.4f} na validacao)")
    return {"metodos": res, "melhor": melhor, "n_calib": n_calib,
            "n_aval": len(sel), "selecionado_em": "validacao",
            "input_scale": input_scale}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/full"))
    parser.add_argument("--out", type=Path, default=Path("results/hw_data"))
    parser.add_argument("--n-vectors", type=int, default=20)
    parser.add_argument("--n-placa", type=int, default=0,
                        help="janelas no pacote da placa; 0 = todas")
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--calib", default="auto",
                        help=f"metodo de calibracao: auto ou {'/'.join(CALIB_METODOS)}")
    parser.add_argument("--n-calib", type=int, default=512)
    parser.add_argument("--n-aval", type=int, default=1000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz)
    layer_order = resolve_layer_order(npz)
    print("ordem das camadas detectada:")
    for pt, hw in layer_order:
        print(f"  {pt} -> {hw}  {npz[f'{pt}.weight_int'].shape}")

    atuais = {hw for _, hw in layer_order}
    orfaos = sorted(p for p in args.out.iterdir()
                    if (m := re.search(r"_(conv\d+)$", p.stem))
                    and m.group(1) not in atuais)
    for p in orfaos:
        p.unlink()
    if orfaos:
        print(f"  removidos {len(orfaos)} arquivos de outra topologia: "
              f"{', '.join(p.name for p in orfaos)}")

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    train_cfg = TrainConfig.from_dict(cfg["training"])

    print("\ngerando arquivos de pesos (.coe/.mem)...")
    weight_info = generate_weight_files(npz, layer_order, args.out, args.weight_bits)
    for hw, inf in weight_info.items():
        print(f"  pesos_{hw}: {inf['weight_shape']} = {inf['weight_count']} valores")

    dados = preparar_dados(train_cfg, args.data_dir,
                           janela=model_cfg.input_len,
                           passo=cfg.get("data", {}).get("stride"))
    if "entrada_escala" not in npz.files:
        raise SystemExit(
            f"{args.npz} nao carrega a escala da entrada.\n"
            f"  Ela e' fixada no treino e define o int8 que a placa recebe.\n"
            f"  Reexporte: python3 -m modelo.prepara --config <config> --refaz")
    input_scale = float(npz["entrada_escala"])
    print(f"\nescala da entrada (do modelo): {input_scale:.6e}")

    if args.calib == "auto":
        print(f"\ncomparando calibracoes de ativacao "
              f"(n_calib={args.n_calib}, n_aval={args.n_aval})...")
        pool_c, modo_c = pooling_da_spec(model_cfg)
        cmp_calib = comparar_calibracoes(train_cfg, args.data_dir, npz, layer_order,
                                         n_calib=args.n_calib, n_aval=args.n_aval,
                                         dados=dados, pool=pool_c, modo=modo_c,
                                         head=model_cfg.head,
                                         input_scale=input_scale)
        calib = cmp_calib["melhor"]
    else:
        calib = args.calib
        cmp_calib = None
    print(f"\ncalibracao adotada: {calib}")

    print("\ngerando vetores de teste (com gabaritos por camada)...")
    vec_stats = generate_test_vectors(model_cfg, train_cfg, args.data_dir,
                                      npz, layer_order, args.n_vectors, args.out,
                                      n_placa=args.n_placa,
                                      calib_metodo=calib, n_calib=args.n_calib,
                                      dados=dados, input_scale=input_scale)
    print(f"  {vec_stats['n_vectors']} vetores, "
          f"acuracia nos vetores: {vec_stats['accuracy_on_vectors']:.4f}")

    print("\ngerando bias_*.mem e manifest.json...")
    npos_fc = geometria_saida(model_cfg)[1]
    bias_acc = bias_acc_chain(npz, layer_order, vec_stats["input_scale"],
                              vec_stats["out_scales"], npos_fc, model_cfg.head)
    generate_bias_files(bias_acc, args.out)
    manifest = generate_manifest(npz, layer_order, model_cfg, vec_stats["input_scale"],
                                 vec_stats["out_scales"], vec_stats["requant"], bias_acc,
                                 vec_stats["n_vectors"], args.out)
    for L in manifest["layers"]:
        print(f"  {L['name']}: acc_bits={L['acc_bits']} w[{L['weights']['base']}:+{L['weights']['words']}] "
              f"bias[{L['bias']['base']}:+{L['bias']['words']}]")

    sat_info = {}
    for pt, hw in layer_order:
        b = npz[f"{pt}.bias_int"]
        sat_info[hw] = {
            "bias_saturated_hi": int((b == 32767).sum()),
            "bias_saturated_lo": int((b == -32768).sum()),
            "bias_count": int(b.size),
        }

    summary = {
        "npz": str(args.npz),
        "config": {"num_layers": model_cfg.num_layers,
                   "num_filters_first": model_cfg.num_filters_first,
                   "kernel_size": model_cfg.kernel_size,
                   "pool_type": model_cfg.pool_type,
                   "head": model_cfg.head},
        "layer_order": [hw for _, hw in layer_order],
        "weight_info": weight_info,
        "vector_stats": vec_stats,
        "bias_saturation": sat_info,
        "calibracao": {"metodo": calib, "n_calib": args.n_calib,
                       "comparacao": cmp_calib},
    }
    with (args.out / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=para_json)

    txt = ["=== resumo da geracao de dados de hardware ===", ""]
    txt.append(f"npz: {args.npz}")
    txt.append(f"config: L{model_cfg.num_layers}_F{model_cfg.num_filters_first:02d}"
               f"_K{model_cfg.kernel_size}_P{model_cfg.pool_type}"
               f"_H{model_cfg.head}")
    txt.append(f"kernel_size = {model_cfg.kernel_size}")
    txt.append("")
    txt.append(f"vetores de teste: {vec_stats['n_vectors']}")
    txt.append(f"acuracia nos vetores: {vec_stats['accuracy_on_vectors']:.4f}")
    txt.append(f"input_scale: {vec_stats['input_scale']:.6e}")
    txt.append("")
    txt.append("ranges das ativacoes por camada (INT8 esperado [-128,127]):")
    for hw, rng in vec_stats["layer_ranges"].items():
        txt.append(f"  {hw}: [{rng[0]}, {rng[1]}]")
    txt.append("")
    txt.append("saturacao de bias (artefato da quantizacao, documentar):")
    for hw, s in sat_info.items():
        txt.append(f"  {hw}: {s['bias_saturated_hi']}/{s['bias_count']} no teto, "
                   f"{s['bias_saturated_lo']}/{s['bias_count']} no piso")
    (args.out / "summary.txt").write_text("\n".join(txt) + "\n")

    print(f"\nok. arquivos em {args.out}/")
    print("  pesos_*.coe, pesos_*.mem  (init de bram)")
    print("  gold_logits.mem           (logits INT32 esperados)")
    print("  gold_classes.txt          (classe esperada por vetor)")
    print("  summary.txt / summary.json")

if __name__ == "__main__":
    main()
