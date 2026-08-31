"""quantiza os pesos para inteiro e mede a perda de acuracia."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, confusion_matrix

from modelo.cwru import ingest_directory, CLASS_NAMES
from modelo.rede import ModelConfig, build_model
from modelo.treina import TrainConfig, make_splits, JanelaDataset

RAIZ = Path(__file__).resolve().parent.parent

def quantize_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    maxabs = x.abs().max()
    if maxabs == 0:
        return x.clone()
    scale = maxabs / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale

def truncate_lsb(x: torch.Tensor, bits: int) -> torch.Tensor:
    return quantize_symmetric(x, bits)

def saturate(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    maxabs = x.abs().max()
    if maxabs == 0:
        return x.clone()
    scale = maxabs / qmax
    return torch.clamp(x / scale, -qmax - 1, qmax) * scale

def quantize_per_channel(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    eixos = tuple(range(1, x.dim()))
    maxabs = x.abs().amax(dim=eixos, keepdim=True)
    escala = torch.where(maxabs == 0, torch.ones_like(maxabs), maxabs / qmax)
    return torch.clamp(torch.round(x / escala), -qmax, qmax) * escala

def quantize_weights_inplace(model: nn.Module, bits: int) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            with torch.no_grad():
                q = (quantize_per_channel if isinstance(module, nn.Conv1d)
                     else quantize_symmetric)
                module.weight.data = q(module.weight.data, bits)
                if module.bias is not None:
                    bias_bits = min(bits * 2, 32)
                    module.bias.data = quantize_symmetric(module.bias.data, bias_bits)

def install_activation_quantization_hooks(model: nn.Module, bits: int, mode: str) -> list:
    handles = []
    if mode == "quantize":
        op = quantize_symmetric
    elif mode == "truncate":
        op = truncate_lsb
    elif mode == "saturate":
        op = saturate
    else:
        raise ValueError(f"modo nao reconhecido: {mode}")

    def hook(module, inputs, output):
        return op(output, bits)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.ReLU, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    return handles

def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()

@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "acc": correct / total,
        "f1_macro": f1_score(targets, preds, average="macro"),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "n_samples": total,
    }

def carrega_estado(model: nn.Module, state: dict, checkpoint: Path) -> None:
    faltam = sorted(set(model.state_dict()) - set(state))
    sobram = sorted(set(state) - set(model.state_dict()))
    if not (faltam or sobram):
        model.load_state_dict(state)
        return

    bn = any("running_mean" in k for k in faltam + sobram)
    causa = ("a normalizacao por lote" if bn else "a estrutura da rede")
    raise SystemExit(
        f"{checkpoint} foi treinado com outra receita: {causa} nao bate com "
        f"a config.\n"
        f"  faltam {len(faltam)} tensores, sobram {len(sobram)}"
        + (f" (ex.: {faltam[0] if faltam else sobram[0]})" if faltam or sobram else "")
        + f"\n  Retreine: python3 -m modelo.prepara --config <config> --refaz")

@torch.no_grad()
def fold_batchnorm(model: nn.Module) -> nn.Module:
    seq = model.features
    for i, mod in enumerate(seq):
        if not isinstance(mod, nn.BatchNorm1d):
            continue
        conv = seq[i - 1]
        if not isinstance(conv, nn.Conv1d):
            raise ValueError(f"BatchNorm em {i} nao segue uma Conv1d")

        gama = mod.weight.detach()
        beta = mod.bias.detach()
        media = mod.running_mean.detach()
        var = mod.running_var.detach()
        escala = gama / torch.sqrt(var + mod.eps)

        b = conv.bias.detach() if conv.bias is not None else torch.zeros_like(media)
        conv.weight.data = conv.weight.data * escala.view(-1, 1, 1)
        if conv.bias is None:
            conv.bias = nn.Parameter((b - media) * escala + beta)
        else:
            conv.bias.data = (b - media) * escala + beta

        seq[i] = nn.Identity()
    return model

def export_quantized_weights(model: nn.Module, bits: int, out_path: Path,
                             grafo: dict | None = None,
                             por_canal: bool = True,
                             entrada: dict | None = None) -> dict:
    arrays = {}
    summary = {"bits": bits, "layers": []}

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Linear)):
            continue
        w = module.weight.detach().cpu()
        qmax = 2 ** (bits - 1) - 1

        canal = por_canal and isinstance(module, nn.Conv1d)
        if canal:
            eixos = tuple(range(1, w.dim()))
            maxabs = w.abs().amax(dim=eixos, keepdim=True)
            escala = torch.where(maxabs == 0, torch.ones_like(maxabs),
                                 maxabs / qmax)
            w_int = torch.round(w / escala).clamp(-qmax, qmax)
            w_int = w_int.numpy().astype(np.int32)
            scale = escala.reshape(-1).numpy().astype(np.float32)
        else:
            maxabs = w.abs().max().item()
            if maxabs == 0:
                scale = np.float32(1.0)
                w_int = torch.zeros_like(w, dtype=torch.int32).numpy()
            else:
                s = maxabs / qmax
                w_int = torch.round(w / s).clamp(-qmax, qmax)
                w_int = w_int.numpy().astype(np.int32)
                scale = np.float32(s)

        arrays[f"{name}.weight_int"] = w_int
        arrays[f"{name}.weight_scale"] = np.asarray(scale, dtype=np.float32)
        layer_info = {"name": name, "kind": type(module).__name__,
                      "weight_shape": list(w.shape),
                      "weight_scale_por_canal": bool(canal),
                      "weight_scale": (scale.tolist() if canal
                                       else float(scale))}

        if module.bias is not None:
            b = module.bias.detach().cpu()
            bias_bits = min(bits * 2, 32)
            bqmax = 2 ** (bias_bits - 1) - 1
            bmax = b.abs().max().item()
            if bmax == 0:
                bscale = 1.0
                b_int = torch.zeros_like(b, dtype=torch.int32).numpy()
            else:
                bscale = bmax / bqmax
                b_int = torch.round(b / bscale).clamp(-bqmax - 1, bqmax).numpy().astype(np.int32)
            arrays[f"{name}.bias_int"] = b_int
            arrays[f"{name}.bias_scale"] = np.array(bscale, dtype=np.float32)
            layer_info["bias_shape"] = list(b.shape)
            layer_info["bias_scale"] = bscale
            layer_info["bias_bits"] = bias_bits

        summary["layers"].append(layer_info)

    if grafo is not None:
        arrays["grafo"] = np.frombuffer(
            json.dumps(grafo, sort_keys=True).encode(), dtype=np.uint8)
        summary["grafo"] = grafo

    if entrada is not None:
        arrays["entrada_escala"] = np.asarray(entrada["escala"], dtype=np.float64)
        arrays["entrada_bits"] = np.asarray(entrada["bits"], dtype=np.int32)
        summary["entrada"] = entrada

    np.savez(out_path, **arrays)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--mode", type=str, default="quantize",
                        choices=["fp32", "quantize", "truncate", "saturate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export-weights", action="store_true")
    parser.add_argument("--escala-por-camada", action="store_true",
                        help="uma escala de peso por camada em vez de por "
                             "canal; e' o caso simples do hardware, e so' "
                             "fecha se a rede tiver sido reajustada com "
                             "quantizacao simulada (modelo.qat)")
    parser.add_argument("--data-dir", default=None,
                        help="sobrepoe o data_dir do yaml")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    train_cfg = TrainConfig.from_dict(cfg["training"])

    data_dir = Path(args.data_dir) if args.data_dir else \
        (RAIZ / str(cfg["data"]["data_dir"]).lstrip("./").replace("../", "")).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, batchnorm=train_cfg.batchnorm).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    carrega_estado(model, state, Path(args.checkpoint))
    model.eval()
    if train_cfg.batchnorm:
        model = fold_batchnorm(model)
        print("batchnorm fundida na convolucao (custo zero em inferencia)")

    print(f"ingerindo {data_dir}...")
    data = ingest_directory(data_dir)
    _, _, te = make_splits(data.X_bruto, data.y, train_cfg.seed,
                           groups=data.grupo, loads=data.load_hp,
                           protocol=train_cfg.protocol,
                           held_out_load=train_cfg.held_out_load)

    ent = None
    for cand in (args.checkpoint.parent / "entrada.json",
                 args.config.parent / "entrada.json"):
        if cand.exists():
            ent = json.loads(cand.read_text())
            break
    if ent is None and train_cfg.entrada_bits:
        raise SystemExit(
            f"nao achei entrada.json perto de {args.checkpoint}.\n"
            f"  Ele e' escrito pelo treino e carrega a escala da entrada em "
            f"int{train_cfg.entrada_bits}. Sem ela o modelo exportado nao "
            f"define o formato que a placa recebe. Retreine.")
    q_ent = float(ent["escala"]) if ent else None
    if q_ent:
        print(f"entrada em int{ent['bits']}, escala {q_ent:.6e}")

    esc_g = None
    if train_cfg.normalizacao == "condicao":
        from modelo.cwru import escala_por_condicao
        mu, sd = escala_por_condicao(data.X_bruto, data.source_file)
        esc_g = (mu[te], sd[te])
        print(f"escala por condicao: {len(set(data.source_file))} gravacoes")
    elif train_cfg.normalizacao == "global":
        tr_g, _, _ = make_splits(data.X_bruto, data.y, train_cfg.seed,
                                 groups=data.grupo, loads=data.load_hp,
                                 protocol=train_cfg.protocol,
                                 held_out_load=train_cfg.held_out_load)
        xt = data.X_bruto[tr_g]
        esc_g = (float(xt.mean()), float(xt.std()))
        print(f"escala global do treino: media {esc_g[0]:.4f}, "
              f"desvio {esc_g[1]:.4f}")

    test_loader = torch.utils.data.DataLoader(
        JanelaDataset(data.X_bruto[te], data.y[te], train_cfg, treino=False,
                      escala_global=esc_g, escala_int8=q_ent),
        batch_size=512, shuffle=False)

    print("avaliando em fp32 (sanidade)...")
    fp32_metrics = evaluate(model, test_loader, device)
    print(f"  fp32 acc: {fp32_metrics['acc']:.4f}, f1: {fp32_metrics['f1_macro']:.4f}")

    if args.mode == "fp32":
        approx_metrics = fp32_metrics
        print("modo=fp32, sem aproximacao")
    else:
        model_q = deepcopy(model)
        print(f"aplicando modo={args.mode} com {args.bits} bits...")
        quantize_weights_inplace(model_q, args.bits)
        handles = install_activation_quantization_hooks(model_q, args.bits, args.mode)
        approx_metrics = evaluate(model_q, test_loader, device)
        remove_hooks(handles)
        print(f"  {args.mode} int{args.bits}: "
              f"acc: {approx_metrics['acc']:.4f}, f1: {approx_metrics['f1_macro']:.4f}")

    delta_acc = approx_metrics["acc"] - fp32_metrics["acc"]
    delta_f1 = approx_metrics["f1_macro"] - fp32_metrics["f1_macro"]
    print(f"  delta vs fp32: acc {delta_acc:+.4f}, f1 {delta_f1:+.4f}")

    result = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "bits": args.bits,
        "mode": args.mode,
        "fp32_acc": fp32_metrics["acc"],
        "fp32_f1_macro": fp32_metrics["f1_macro"],
        "approx_acc": approx_metrics["acc"],
        "approx_f1_macro": approx_metrics["f1_macro"],
        "delta_acc": delta_acc,
        "delta_f1": delta_f1,
        "approx_confusion_matrix": approx_metrics["confusion_matrix"],
        "n_test_samples": approx_metrics["n_samples"],
        "device": str(device),
    }
    with (args.out / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2)

    if args.export_weights and args.mode != "fp32":
        export_path = args.out / "weights_int.npz"
        summary = export_quantized_weights(
            deepcopy(model).to("cpu"), args.bits, export_path,
            por_canal=not args.escala_por_camada, entrada=ent,
            grafo=asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__")
            else dict(model_cfg))
        with (args.out / "weights_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  pesos exportados em {export_path}")

    print(f"saida em: {args.out}")

if __name__ == "__main__":
    main()
