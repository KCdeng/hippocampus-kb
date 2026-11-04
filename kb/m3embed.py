r"""m3embed.py — bge-m3 ONNX 嵌入器（接口兼容 fastembed: .embed(texts, batch_size) → list[np.ndarray]）
模型: M3_ONNX_DIR 指向的 bge-m3 ONNX 目录（aapot/bge-m3-onnx fp32，下载步骤见 README）
契约: dense = last_hidden_state[:,0] (CLS) + L2 归一化; 文档/查询均无前缀
后端: KB_EMBED_BACKEND=cpu|cuda（默认 cpu）。cuda 只在 onnxruntime 真的探测到
CUDA provider 时启用，装不了就报错——静默退回 CPU 会让同一份 vec 混两种后端。
"""
import os
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort

M3_DIR = os.environ.get("M3_ONNX_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bge-m3-onnx"))
BACKEND = os.environ.get("KB_EMBED_BACKEND", "cpu").lower()


def providers():
    if BACKEND not in ("cpu", "cuda"):
        raise RuntimeError(f"未知 KB_EMBED_BACKEND={BACKEND}（可选 cpu/cuda）")
    avail = ort.get_available_providers()
    if BACKEND == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(f"请求 CUDA 但 onnxruntime 只有 {avail}；"
                               f"装好 onnxruntime-gpu 再开，或以 cpu 重跑")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class BgeM3ONNX:
    backend = BACKEND

    def __init__(self, max_seq=8192):
        self.tok = Tokenizer.from_file(os.path.join(M3_DIR, "tokenizer.json"))
        self.pad_id = self.tok.token_to_id("<pad>")
        self.tok.enable_truncation(max_length=max_seq)
        self.sess = ort.InferenceSession(
            os.path.join(M3_DIR, "model.onnx"),
            providers=providers(),
        )
        in_names = [i.name for i in self.sess.get_inputs()]
        self.in_ids = next(n for n in in_names if "input_ids" in n)
        self.in_mask = next(n for n in in_names if "attention" in n)
        self.extra = [n for n in in_names if n not in (self.in_ids, self.in_mask)]
        self._out_names = [o.name for o in self.sess.get_outputs()]
        self._warm()

    def _warm(self):
        self.embed(["warmup"])

    def _encode(self, texts):
        enc = self.tok.encode_batch(texts)
        L = max(len(e) for e in enc)
        ids = np.full((len(texts), L), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(texts), L), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        feed = {self.in_ids: ids, self.in_mask: mask}
        for n in self.extra:
            feed[n] = np.zeros_like(mask)
        return feed

    def embed(self, texts, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            b = texts[i : i + batch_size]
            res = self.sess.run(None, self._encode(b))
            o = {name: r for name, r in zip(self._out_names, res)}
            v = o["dense_vecs"].astype(np.float32)
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            out.extend(v.tolist())
        return out
