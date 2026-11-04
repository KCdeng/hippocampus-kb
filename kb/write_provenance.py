"""write_provenance.py — 给已有 vec.npy 补/刷新溯源 sidecar（带对齐抽样验证）。

为什么需要它：build_vec.py 每次嵌完都会写 `<out>.provenance.json`，但历史上 mem 区
把临时 vec 原子改名成生产档时没带上 sidecar（孤儿留在 vec.npy.new.<pid>.npy.provenance.json），
epi 区则从来没有落过 sidecar。手写一份 JSON 等于把"声明"当"事实"，所以这里先验证再写：
按宣称的模型/约定重新嵌若干块，与磁盘行算余弦——只有 mean≈1 才认为这份 vec 就是
当前 chunks 的那一代，然后把模型、约定、行数、块档哈希和这次验证结果一起写进去。

用法（只改 <vec>.provenance.json 这一个文件，不动 vec/chunks）:
  venv/Scripts/python.exe -X utf8 kb/write_provenance.py --zone mem
  ... --zone main --n 12 --note "V3 生产向量档，行号修正后文本未变"
"""
import argparse, datetime, hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ZONES = {
    "main": {"chunks": os.path.join(HERE, "chunks.jsonl"), "vec": os.path.join(HERE, "vec_m3.npy"),
             "model": "bge-m3-onnx"},
    "mem": {"chunks": os.path.join(HERE, "mem", "chunks.jsonl"), "vec": os.path.join(HERE, "mem", "vec.npy"),
            "model": "bge-m3-onnx"},
    "epi": {"chunks": os.path.join(HERE, "epi", "chunks.jsonl"), "vec": os.path.join(HERE, "epi", "vec.npy"),
            "model": "bge-m3-onnx"},
    "zh": {"chunks": os.path.join(BASE, "kb_zh", "chunks.jsonl"), "vec": os.path.join(BASE, "kb_zh", "vec.npy"),
           "model": "intfloat/multilingual-e5-large"},
    "comp": {"chunks": os.path.join(BASE, "kb_comp", "chunks.jsonl"), "vec": os.path.join(BASE, "kb_comp", "vec.npy"),
             "model": "bge-m3-onnx"},
}
MIN_COS = 0.999   # 低于这个说明行错位或换了模型/约定，不许落 provenance


def sample_rows(n_rows, n):
    """均匀取样，含首尾——行错位常从中段开始，只取前几条看不见。"""
    n = min(n, n_rows)
    return sorted({round(i * (n_rows - 1) / max(n - 1, 1)) for i in range(n)})


def verify(chunks, vec, idx, model, embedder=None):
    pfx = "passage: " if "e5" in model else ""
    texts = [pfx + chunks[i]["text"] for i in idx]
    if embedder is None:
        if model == "bge-m3-onnx":
            os.environ.setdefault("KB_EMBED_BACKEND", "cpu")
            from m3embed import BgeM3ONNX
            embedder, backend = BgeM3ONNX(), "cpu"
        else:
            from fastembed import TextEmbedding
            embedder, backend = TextEmbedding(model), "fastembed"
    else:
        backend = getattr(embedder, "backend", "cpu")
    got = np.vstack([np.asarray(v) for v in embedder.embed(texts, batch_size=max(len(texts), 1))])
    got /= (np.linalg.norm(got, axis=1, keepdims=True) + 1e-9)
    stored = vec[idx] / (np.linalg.norm(vec[idx], axis=1, keepdims=True) + 1e-9)
    cos = np.einsum("ij,ij->i", got, stored)
    return backend, float(cos.mean()), float(cos.min()), [int(i) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=sorted(ZONES))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true", help="抽样不过也照写（默认拒绝）")
    a = ap.parse_args()
    zc = ZONES[a.zone]
    chunks = [json.loads(l) for l in open(zc["chunks"], encoding="utf-8") if l.strip()]
    vec = np.load(zc["vec"])
    if vec.shape[0] != len(chunks):
        sys.exit(f"拒绝：{zc['vec']} 有 {vec.shape[0]} 行 ≠ {len(chunks)} 块")
    backend, mean, mn, idx = verify(chunks, vec, sample_rows(len(chunks), a.n), zc["model"])
    ok = mn >= MIN_COS
    print(f"[{a.zone}] 抽样 {len(idx)} 行重嵌比对：mean={mean:.5f} min={mn:.5f} "
          f"({'对齐' if ok else '不对齐'})")
    if not ok and not a.force:
        sys.exit(f"拒绝落 provenance：min 余弦 {mn:.5f} < {MIN_COS}，"
                 f"这份 vec 不是当前 chunks 的那一代（或模型/约定不是 {zc['model']} 纯 text）")
    prov = {"model": zc["model"], "backend": backend,
            "passage_prefix": "passage: " if "e5" in zc["model"] else "",
            "embed_heading": False,
            "chunks": os.path.abspath(zc["chunks"]), "n_chunks": len(chunks),
            "vec_shape": list(vec.shape),
            "chunks_sha256": hashlib.sha256(open(zc["chunks"], "rb").read()).hexdigest()[:16],
            "created": datetime.datetime.fromtimestamp(os.path.getmtime(zc["vec"]))
                        .strftime("%Y-%m-%d %H:%M:%S"),
            "verified": {"method": "sampled re-embed cosine (kb/write_provenance.py)",
                         "rows": idx, "mean_cos": round(mean, 6), "min_cos": round(mn, 6),
                         "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            "note": (a.note + (" " if a.note else "") +
                     "provenance 由抽样重嵌验证后补写/刷新；行对齐为抽样证据，非全量证明").strip()}
    out = zc["vec"] + ".provenance.json"
    json.dump(prov, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
