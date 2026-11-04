"""index.py — 构建 BM25 + 向量双索引（一次性，落盘复用）。
四件（bm25.npy / bm_vocab.json / vec.npy / meta.json）都经 kb_atomic 原子换档：
写完才见得到，中断只毁临时件，上一版带时间戳留 3 份可回滚。"""
import json, os, math, sys, time
import numpy as np
import jieba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_atomic

HERE = os.environ.get("KB_DIR", os.path.dirname(os.path.abspath(__file__)))
# 领域用户词典（主库 kb/ 目录下；mem/zh 区无此文件则自动跳过）
_USERDICT = os.path.join(HERE, "jieba_userdict.txt")
if os.path.exists(_USERDICT):
    jieba.load_userdict(_USERDICT)
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VEC = os.path.join(HERE, "vec.npy")
META = os.path.join(HERE, "meta.json")
BM25 = os.path.join(HERE, "bm25.npy")
VOCAB = os.path.join(HERE, "bm_vocab.json")
STOP = set("的了是在有和与及对于中为以将由通过等的或如果不则使能被把向从至于以及".split())

def tokenize(text):
    return [t for t in jieba.cut_for_search(text.lower())
            if t.strip() and (t.isdigit() or (len(t) > 1 and t not in STOP))]

def build():
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    t0 = time.time()

    # ---- BM25 ----
    doc_toks = [tokenize(c["heading"] + "。" + c["text"]) for c in chunks]
    df, N = {}, len(chunks)
    for toks in doc_toks:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    vocab = sorted(df)
    vid = {t: i for i, t in enumerate(vocab)}
    import os
    k1, b, avgdl = float(os.environ.get("KB_K1", "1.5")), float(os.environ.get("KB_B", "0.75")), sum(map(len, doc_toks)) / N
    bm = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, toks in enumerate(doc_toks):
        dl = len(toks)
        tfm = {}
        for t in toks:
            tfm[t] = tfm.get(t, 0) + 1
        idf_base = {}
        for t, tf in tfm.items():
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            bm[i, vid[t]] = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    kb_atomic.save_npy(BM25, bm)
    kb_atomic.save_json(VOCAB, {"vocab": vocab})

    # ---- 向量 ----
    # 不再"缺文件就自动重嵌"：历史上这条路径用默认 bge-small(512d)+heading 约定
    # 悄悄生成过与主引擎不匹配的 vec，形状合法但分数全错。要重嵌请显式跑
    # build_vec.py / build_m3_index.py，或设 KB_ALLOW_EMBED=1。
    if not os.path.exists(VEC):
        if os.environ.get("KB_ALLOW_EMBED") == "1":
            model = os.environ.get("KB_MODEL", "BAAI/bge-small-zh-v1.5")
            pfx = "passage: " if "e5" in model else ""
            if model == "bge-m3-onnx":
                from m3embed import BgeM3ONNX
                m = BgeM3ONNX()
            else:
                from fastembed import TextEmbedding
                m = TextEmbedding(model)
            # 生产主库向量是纯 text 约定（<对端引擎目录>/build_m3_index.py:12）。
            # 这里默认对齐，heading 拼进嵌入文本只作为实验开关。
            pre = (lambda c: c["heading"].replace("/", " ") + " ") if os.environ.get("KB_EMBED_HEADING") == "1" else (lambda c: "")
            texts = [pfx + pre(c) + c["text"] for c in chunks]
            vecs = np.vstack([v for v in m.embed(texts, batch_size=32)])
            kb_atomic.save_npy(VEC, vecs.astype(np.float16))
        else:
            print(f"WARN: {VEC} 缺失，向量档不可用（BM25 已建好）。", flush=True)
    meta = {"n_chunks": N, "vocab": len(vocab), "k1": k1, "b": b,
            "vec": os.path.basename(VEC) if os.path.exists(VEC) else None,
            "vec_shape": list(np.load(VEC, mmap_mode="r").shape) if os.path.exists(VEC) else None,
            "model": os.environ.get("KB_MODEL", "")}
    kb_atomic.save_json(META, meta, indent=1)
    print(f"built: N={N} vocab={len(vocab)} vec={os.path.exists(VEC)} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    jieba.setLogLevel(60)
    build()
