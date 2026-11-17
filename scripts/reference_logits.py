"""Independent reference forward pass for llama GGUF models, to check Adamant's logits.

Weights are dequantized by the `gguf` Python package (from the llama.cpp project), not by
Adamant's code; the forward pass is plain numpy in float64 with float (unquantized)
activations. Compares the final-position logits with `adamant logits` output.

usage:
  python scripts/reference_logits.py MODEL.gguf ADAMANT_LOGITS.txt [--out report.txt]
The token ids are read from the first line of the logits file ("# tokens 1 2 3").
"""
import argparse
import sys
import time

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    if v.dtype == np.uint8 and len(f.types) and f.types[0].name == "STRING":
        return bytes(v).decode()
    return v[0].item() if v.size == 1 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("logits")
    ap.add_argument("--out")
    a = ap.parse_args()

    with open(a.logits, encoding="utf-8") as f:
        head = f.readline().split()
        assert head[:2] == ["#", "tokens"], "first line must be '# tokens ...'"
        tokens = [int(t) for t in head[2:]]
        got = np.array([float(x) for x in f if x.strip()], dtype=np.float64)

    t0 = time.time()
    r = GGUFReader(a.model)
    n_layer = field(r, "llama.block_count")
    dim = field(r, "llama.embedding_length")
    n_head = field(r, "llama.attention.head_count")
    n_kv = field(r, "llama.attention.head_count_kv", n_head)
    eps = field(r, "llama.attention.layer_norm_rms_epsilon", 1e-5)
    base = field(r, "llama.rope.freq_base", 10000.0)
    hd = dim // n_head

    W = {}
    for t in r.tensors:
        W[t.name] = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(
            [int(x) for x in reversed(t.shape.tolist())] if len(t.shape) > 1 else [-1])
    out_w = W.get("output.weight", W["token_embd.weight"])
    print(f"dequantized {len(W)} tensors with gguf {np.__name__} in {time.time() - t0:.1f} s", file=sys.stderr)

    def rms(x, w):
        return x / np.sqrt(np.mean(x * x) + eps) * w

    def rope(v, pos, nh):
        v = v.reshape(nh, hd).copy()
        i = np.arange(hd // 2)
        th = pos * base ** (-2.0 * i / hd)
        c, s = np.cos(th), np.sin(th)
        a0, b0 = v[:, 0::2].copy(), v[:, 1::2].copy()
        v[:, 0::2] = a0 * c - b0 * s
        v[:, 1::2] = a0 * s + b0 * c
        return v.reshape(-1)

    kc = [[] for _ in range(n_layer)]
    vc = [[] for _ in range(n_layer)]
    logits = None
    t0 = time.time()
    for pos, tok in enumerate(tokens):
        x = W["token_embd.weight"][tok].astype(np.float64)
        for l in range(n_layer):
            p = f"blk.{l}."
            h = rms(x, W[p + "attn_norm.weight"])
            q = W[p + "attn_q.weight"].astype(np.float64) @ h
            k = W[p + "attn_k.weight"].astype(np.float64) @ h
            v = W[p + "attn_v.weight"].astype(np.float64) @ h
            q, k = rope(q, pos, n_head), rope(k, pos, n_kv)
            kc[l].append(k.reshape(n_kv, hd))
            vc[l].append(v.reshape(n_kv, hd))
            K = np.stack(kc[l])  # [t, kv, hd]
            V = np.stack(vc[l])
            att = np.zeros(dim)
            for hh in range(n_head):
                g = hh // (n_head // n_kv)
                sc = K[:, g, :] @ q[hh * hd:(hh + 1) * hd] / np.sqrt(hd)
                sc = np.exp(sc - sc.max())
                sc /= sc.sum()
                att[hh * hd:(hh + 1) * hd] = sc @ V[:, g, :]
            x = x + W[p + "attn_output.weight"].astype(np.float64) @ att
            h = rms(x, W[p + "ffn_norm.weight"])
            gt = W[p + "ffn_gate.weight"].astype(np.float64) @ h
            up = W[p + "ffn_up.weight"].astype(np.float64) @ h
            x = x + W[p + "ffn_down.weight"].astype(np.float64) @ (gt / (1 + np.exp(-gt)) * up)
        logits = out_w.astype(np.float64) @ rms(x, W["output_norm.weight"])
    print(f"reference forward over {len(tokens)} tokens in {time.time() - t0:.1f} s", file=sys.stderr)

    assert got.shape == logits.shape, (got.shape, logits.shape)
    cos = float(got @ logits / np.linalg.norm(got) / np.linalg.norm(logits))
    diff = np.abs(got - logits)
    top_ref = np.argsort(-logits)[:10]
    top_got = np.argsort(-got)[:10]
    p_ref = np.exp(logits - logits.max()); p_ref /= p_ref.sum()
    p_got = np.exp(got - got.max()); p_got /= p_got.sum()
    kl = float(np.sum(p_ref * (np.log(p_ref + 1e-30) - np.log(p_got + 1e-30))))
    lines = [
        f"model            {a.model}",
        f"tokens           {len(tokens)}: {tokens}",
        f"cosine           {cos:.6f}",
        f"max |diff|       {diff.max():.4f}  (logit range {logits.min():.2f} .. {logits.max():.2f})",
        f"mean |diff|      {diff.mean():.4f}",
        f"top-1            reference {top_ref[0]}  adamant {top_got[0]}  {'MATCH' if top_ref[0] == top_got[0] else 'DIFFERENT'}",
        f"top-10 overlap   {len(set(top_ref) & set(top_got))}/10",
        f"KL(ref || adamant) over the softmax: {kl:.2e} nats",
    ]
    report = "\n".join(lines)
    print(report)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
