"""build_vec.py — 可配文本约定的重嵌（写任意输出文件，不碰生产索引）。

用法: python build_vec.py --chunks <jsonl> --out <vec.npy>
  KB_EMBED_HEADING=1  把标题链拼进嵌入文本（默认 0=纯 text，与生产 vec_m3.npy 约定一致；
                      H 臂实验证明拼标题过不了闸门，故不作为默认）
  KB_MODEL            嵌入模型（默认 bge-m3-onnx；写 intfloat/multilingual-e5-large 走
                      fastembed + "passage: " 前缀，即专利库那一代约定）
  --backend cpu|cuda  计算后端（默认 cpu；cuda 只在 onnxrt 探测到 CUDA provider 时启用）
  --prev / --prev-chunks  复用上一代向量：按「嵌入文本的 sha256」配对，只嵌新增/改动的块。
                      前提是模型/前缀/拼标题/后端全部一致（provenance 校验），否则拒绝，
                      避免同一份 vec 里混两种约定或两种后端。
  每次运行落 <out>.provenance.json：模型/后端/前缀/拼标题/块文件哈希，防止"看着能用其实不同代"。
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kb_atomic
USE_HEADING = os.environ.get("KB_EMBED_HEADING", "0") == "1"
MODEL = os.environ.get("KB_MODEL", "bge-m3-onnx")


def save_vec(vec, out, prov):
    """向量与它的 provenance 在同一瞬换档。

    旧写法是 np.save 直写 out、再单独 open 写溯源：中断（磁盘满、Ctrl-C、OOM）会留下
    半截 (N, d) 的生产 vec，同目录 chunks 的行号全体作废；而 kb_remember/kb_epilogue 的
    重建链正是拿这个脚本直写 mem/epi 生产档的。换成临时件 + 原子换档，旧档带时间戳留 3 份。"""
    with kb_atomic.stage(out) as tmp:
        np.save(tmp, vec)
        with open(tmp + kb_atomic.PROV_SUFFIX, "w", encoding="utf-8") as w:
            json.dump(prov, w, ensure_ascii=False, indent=1)


def get_embedder(model):
    """返回 (embedder, passage_prefix)。同一份 vec 文件里不得混用两种后端的向量，
    所以后端只在这里决定一次，不在循环里换。"""
    if model == "bge-m3-onnx":
        from m3embed import BgeM3ONNX
        return BgeM3ONNX(), ""
    from fastembed import TextEmbedding
    return TextEmbedding(model), ("passage: " if "e5" in model else "")


def keys_of(texts):
    return [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]


def load_prev(prev_vec, prev_chunks, want_model, want_pfx, want_backend):
    """读上一代 (chunks, vec)，返回 {文本哈希: 行向量}。任何一层对不上就拒绝复用。"""
    prov_path = prev_vec + ".provenance.json"
    if not os.path.exists(prov_path):
        sys.exit(f"拒绝复用：{prev_vec} 没有 provenance，无法确认是同一代约定")
    prov = json.load(open(prov_path, encoding="utf-8"))
    # 旧版 provenance 把 backend 写成运行时名 "onnx-m3"（当时只有 CPU），不是设备名
    old_backend = {"onnx-m3": "cpu"}.get(prov.get("backend"), prov.get("backend"))
    bad = [f"{k}={prov.get(k)!r} ≠ {v!r}" for k, v in
           (("model", want_model), ("passage_prefix", want_pfx), ("embed_heading", USE_HEADING))
           if prov.get(k) != v]
    if old_backend != want_backend:
        bad.append(f"backend={prov.get('backend')!r} ≠ {want_backend!r}")
    if bad:
        sys.exit("拒绝复用（约定不一致）: " + "；".join(bad))
    old_chunks = [json.loads(l) for l in open(prev_chunks, encoding="utf-8") if l.strip()]
    old_vec = np.load(prev_vec)
    if old_vec.shape[0] != len(old_chunks):
        sys.exit(f"拒绝复用：{prev_vec} 有 {old_vec.shape[0]} 行 ≠ {prev_chunks} 的 {len(old_chunks)} 块")
    texts = [c["heading"].replace("/", " ") + " " + c["text"] if USE_HEADING else c["text"]
             for c in old_chunks]
    if want_pfx:
        texts = [want_pfx + t for t in texts]
    return {k: old_vec[i] for i, k in enumerate(keys_of(texts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default=os.environ.get("KB_EMBED_BACKEND", "cpu"),
                    choices=["cpu", "cuda"],
                    help="计算后端（remote 未实现，别传）")
    ap.add_argument("--prev", help="上一代 vec.npy（配合 --prev-chunks 做增量复用）")
    ap.add_argument("--prev-chunks", help="与 --prev 行对齐的 chunks.jsonl")
    a = ap.parse_args()
    os.environ["KB_EMBED_BACKEND"] = a.backend   # m3embed 在 import 时读，必须先设

    chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8") if l.strip()]
    texts = [c["text"] for c in chunks]
    if USE_HEADING:
        texts = [c["heading"].replace("/", " ") + " " + t for c, t in zip(chunks, texts)]
    print(f"chunks={len(texts)} chars={sum(map(len, texts))} heading={USE_HEADING} "
          f"model={MODEL} backend={a.backend} out={a.out}", flush=True)

    t0 = time.time()
    emb, pfx = get_embedder(MODEL)
    if pfx:
        texts = [pfx + t for t in texts]
    backend = getattr(emb, "backend", "fastembed" if MODEL != "bge-m3-onnx" else a.backend)
    t1 = time.time()
    print(f"[0] load+warm: {t1 - t0:.1f}s backend={backend}", flush=True)

    reuse = {}
    if a.prev and a.prev_chunks:
        reuse = load_prev(a.prev, a.prev_chunks, MODEL, pfx, backend)
    keys = keys_of(texts)
    hit = [i for i, k in enumerate(keys) if k in reuse]
    miss = [i for i, k in enumerate(keys) if k not in reuse]
    print(f"[1] 复用 {len(hit)} 块，需嵌 {len(miss)} 块", flush=True)

    if hit:
        dim = int(np.asarray(reuse[keys[hit[0]]]).shape[0])
    else:
        # fastembed 的 embed() 返生成器（不能下标），m3embed 返 list；统一迭代一次取首条
        dim = len(np.asarray(next(iter(emb.embed(["probe"], batch_size=1)))))
    vec = np.empty((len(texts), dim), dtype=np.float32)
    for i in hit:
        vec[i] = reuse[keys[i]]

    B = 8
    nchars = sum(len(texts[i]) for i in miss)
    done_chars = 0
    for j in range(0, len(miss), B):
        idx = miss[j:j + B]
        vec[idx] = np.vstack([v for v in emb.embed([texts[i] for i in idx], batch_size=B)])
        done_chars += sum(len(texts[i]) for i in idx)
        if j % 64 == 0 or j + B >= len(miss):
            el = max(time.time() - t1, 1e-9)
            rate = done_chars / el
            print(f"[{j + len(idx)}/{len(miss)}] {rate:.0f} chars/s, "
                  f"ETA {(nchars - done_chars) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    prov = {"model": MODEL, "backend": backend,
            "passage_prefix": pfx, "embed_heading": USE_HEADING,
            "chunks": os.path.abspath(a.chunks), "n_chunks": len(texts), "vec_shape": list(vec.shape),
            "reused_prev": os.path.abspath(a.prev) if a.prev else None, "n_reused": len(hit),
            "chunks_sha256": hashlib.sha256(open(a.chunks, "rb").read()).hexdigest()[:16],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_vec(vec, a.out, prov)
    print(f"DONE: {os.path.basename(a.out)} {vec.shape} reused {len(hit)} total {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
