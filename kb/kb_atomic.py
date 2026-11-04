"""kb_atomic.py — 索引件的原子落盘（chunker / build_vec / index / kb_remember 共用）。

为什么要有这个件：`open(path, "w")` 和 `np.save(path, arr)` 都不是原子操作。磁盘满、进程被杀、
Ctrl-C 落在写中间，生产档就成半截文件——行对齐校验（KB.__init__ 的 chunks/bm25/vec 断言）能
拒挂它，但**上一版已经被覆写了**，等于把"坏了"升级成"坏了且没有可回滚的好档"。
本库把崩溃风险写实的路径有三条：chunker.py 直写 chunks.jsonl、build_vec.py 直写 vec.npy +
provenance、index.py 直写 bm25.npy/bm_vocab.json/meta.json；而 kb_epilogue 的复盘区入库链、
kb_remember 的 mem 重建链正是拿它们直写生产件的。

约定（沿用 mem 区已经跑熟的 promote_vec）：
  1. 先写到 `<dst>.new.<pid><同扩展名>`——同扩展名是必须的，np.save 会给无 .npy 的名字补后缀，
     临时件与目标不同名就会留下找不到主人的孤儿。
  2. 写完（含旁边的 `<dst>.provenance.json` 溯源）一次 `os.replace` 换档，读写者只会看见
     旧版或新版，不会看见半成品。
  3. 旧版带着它的溯源进 `<dst>.prev.<YYYYmmdd-HHMMSS>` 归档，只保最近 keep 份（成对剪，
     否则归档旁的溯源会让磁盘单向增长）。
  4. 换档时若新档没有溯源，就把留在 dst 旁边的旧溯源一起删掉——描述上一代的溯源比没有溯源更糟。
"""
import contextlib
import datetime
import glob
import json
import os

PROV_SUFFIX = ".provenance.json"


def _drop(path):
    try:
        os.remove(path)
    except OSError:
        pass


def tmp_name(dst):
    """临时件路径：带当前进程号，且与目标同扩展名。"""
    return f"{dst}.new.{os.getpid()}{os.path.splitext(dst)[1]}"


def _prune(dst, keep):
    """按时间戳把 .prev 归档压在 keep 份内，归档的 provenance 跟着一起剪。"""
    if keep <= 0:
        return
    d, base = os.path.dirname(dst) or ".", os.path.basename(dst)
    prefix = base + ".prev."
    try:
        names = sorted(f for f in os.listdir(d) if f.startswith(prefix) and not f.endswith(PROV_SUFFIX))
    except OSError:
        return
    for stale in names[:-keep]:
        p = os.path.join(d, stale)
        _drop(p)
        _drop(p + PROV_SUFFIX)


def promote(tmp, dst, keep=3):
    """把临时件原子换成生产件，旁边的 provenance sidecar 一起走，旧版带时间戳归档。"""
    src_prov, dst_prov = tmp + PROV_SUFFIX, dst + PROV_SUFFIX
    if os.path.exists(dst):
        stamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
        prev = f"{dst}.prev.{stamp}"
        seq = 0
        while os.path.exists(prev) or os.path.exists(prev + PROV_SUFFIX):
            seq += 1                     # 同一秒内连续换档：不避让会把上一份归档直接吃掉
            prev = f"{dst}.prev.{stamp}-{seq}"
        os.replace(dst, prev)
        if os.path.exists(dst_prov):
            os.replace(dst_prov, prev + PROV_SUFFIX)
    os.replace(tmp, dst)
    if os.path.exists(src_prov):
        os.replace(src_prov, dst_prov)
    elif os.path.exists(dst_prov):
        os.remove(dst_prov)
    for orphan in glob.glob(dst + ".new.*"):   # 上一次崩溃留下的半成品，这次顺手收干净
        _drop(orphan)
    _prune(dst, keep)


@contextlib.contextmanager
def stage(dst, keep=3):
    """在临时件里写，正常退出才原子换档；中途抛异常（含 Ctrl-C）只删临时件，生产件不动。"""
    tmp = tmp_name(dst)
    try:
        yield tmp
    except BaseException:
        _drop(tmp)
        _drop(tmp + PROV_SUFFIX)
        raise
    promote(tmp, dst, keep)


def save_json(path, obj, keep=3, indent=None):
    with stage(path, keep) as tmp:
        with open(tmp, "w", encoding="utf-8") as w:
            json.dump(obj, w, ensure_ascii=False, indent=indent)


def save_npy(path, arr, keep=3):
    import numpy as np
    with stage(path, keep) as tmp:
        np.save(tmp, arr)


def save_lines(path, lines, keep=3):
    with stage(path, keep) as tmp:
        with open(tmp, "w", encoding="utf-8") as w:
            for line in lines:
                w.write(line)
