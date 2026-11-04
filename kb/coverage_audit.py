"""coverage_audit.py — 只读的索引覆盖审计。

回答三个 /stats 答不了的问题：
  1. 语料目录里**现有**的合格文件，有哪些根本没进索引？（stale 只看已入库文件，看不见这个）
  2. 已入库的文件里，有哪些在索引之后又被改过？（内容漂移，行数对得上也照样是旧向量）
  3. 索引里混进了多少临时/中间件路径？（下次全量重建会把它们带进候选池）

不改任何文件；输出 JSON + 控制台摘要。
用法: python kb/scratch/coverage_audit.py [--out kb/scratch/coverage_audit.json]
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))            # …/hippocampus/kb
BASE = os.path.dirname(HERE)                                  # …/hippocampus
WORKSPACE = os.path.dirname(os.path.dirname(BASE))            # …/research -> 工作区根（main 语料根）
sys.path.insert(0, os.path.join(BASE, "kb"))
from chunker import iter_md_files  # noqa: E402  审计口径必须与入库口径同源

# 临时/中间件形态（工作区里人肉可辨的噪声，不是 chunker 的排除规则）
TEMP_HINTS = ("_tmp_frames/", "_pdf_snips/", "/scratch/", "preview/")
TEMP_NAME_HINTS = (".bak", ".old", ".prev.", ".new.")

ZONES = {
    "main": {"root": WORKSPACE, "index": os.path.join(BASE, "kb", "chunks.jsonl"),
             "vec": os.path.join(BASE, "kb", "vec_m3.npy"), "model": "bge-m3-onnx",
             "walker": True},
    "mem": {"root": os.path.join(BASE, "kb", "mem", "corpus"),
            "index": os.path.join(BASE, "kb", "mem", "chunks.jsonl"),
            "vec": os.path.join(BASE, "kb", "mem", "vec.npy"), "model": "bge-m3-onnx"},
    "epi": {"root": os.path.join(BASE, "kb", "epi", "corpus"),
            "index": os.path.join(BASE, "kb", "epi", "chunks.jsonl"),
            "vec": os.path.join(BASE, "kb", "epi", "vec.npy"), "model": "bge-m3-onnx"},
    "zh": {"root": os.path.join(BASE, "corpus_zh"),
           "index": os.path.join(BASE, "kb_zh", "chunks.jsonl"),
           "vec": os.path.join(BASE, "kb_zh", "vec.npy"), "model": "intfloat/multilingual-e5-large"},
    "comp": {"root": os.path.join(BASE, "corpus_comp", "md"),
             "index": os.path.join(BASE, "kb_comp", "chunks.jsonl"),
             "vec": os.path.join(BASE, "kb_comp", "vec.npy"), "model": "bge-m3-onnx"},
}


def ts(p):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")
    except OSError:
        return "-"


def read_index(p):
    """返回 (rows, {relfile: n_chunks})；relfile 统一成正斜杠，跟 chunks 里的写法一致。"""
    rows, files = 0, {}
    with open(p, encoding="utf-8") as r:
        for line in r:
            if not line.strip():
                continue
            rows += 1
            f = json.loads(line)["file"].replace("\\", "/")
            files[f] = files.get(f, 0) + 1
    return rows, files


def disk_files(zc):
    root = zc["root"]
    if not os.path.isdir(root):
        return set()
    if zc.get("walker"):
        return {os.path.relpath(p, root).replace("\\", "/") for p in iter_md_files(root)}
    out = set()
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".md"):
                out.add(os.path.relpath(os.path.join(dirpath, n), root).replace("\\", "/"))
    return out


def is_temp(f):
    return any(h in "/" + f for h in TEMP_HINTS) or any(f.lower().endswith(h) for h in (".bak", ".old"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "coverage_audit.json"))
    a = ap.parse_args()
    report = {}
    for zone, zc in ZONES.items():
        idx = zc["index"]
        if not os.path.exists(idx):
            report[zone] = {"error": f"索引缺失 {idx}"}
            continue
        rows, indexed = read_index(idx)
        on_disk = disk_files(zc)
        idx_mtime = os.path.getmtime(idx)
        unindexed = sorted(on_disk - set(indexed))
        gone = sorted(set(indexed) - on_disk)
        # 已入库但索引之后又被改过的（内容漂移）
        drifted = []
        for f in indexed:
            p = os.path.join(zc["root"], f.replace("/", os.sep))
            try:
                if os.path.getmtime(p) > idx_mtime:
                    drifted.append((f, os.path.getmtime(p) - idx_mtime))
            except OSError:
                drifted.append((f, None))
        drifted.sort(key=lambda x: -(x[1] or 0))
        temp_in = [f for f in indexed if is_temp(f)]
        temp_new = [f for f in unindexed if is_temp(f)]
        vec_rows = None
        prov = None
        if os.path.exists(zc["vec"]):
            import numpy as np
            vec_rows = int(np.load(zc["vec"], mmap_mode="r").shape[0])
        pp = zc["vec"] + ".provenance.json"
        if os.path.exists(pp):
            prov = json.load(open(pp, encoding="utf-8"))
        prov_state = "missing"
        prov_issues = []
        if prov:
            prov_state = "ok"
            if prov.get("model") != zc["model"]:
                prov_issues.append(f"model={prov.get('model')!r} ≠ 期望 {zc['model']!r}")
            if prov.get("n_chunks") != rows:
                prov_issues.append(f"n_chunks={prov.get('n_chunks')} ≠ 现索引 {rows}")
            want = hashlib.sha256(open(idx, "rb").read()).hexdigest()[:16]
            if prov.get("chunks_sha256") != want:
                prov_issues.append(f"chunks_sha256={prov.get('chunks_sha256')} ≠ 现档 {want}")
        # 孤儿临时 provenance（重嵌改名时漏带 sidecar 的痕迹）
        orphans = [fn for fn in os.listdir(os.path.dirname(zc["vec"]))
                   if ".new." in fn and fn.endswith(".provenance.json")]
        report[zone] = {"root": zc["root"], "index": idx, "index_mtime": ts(idx),
                        "rows": rows, "vec_rows": vec_rows,
                        "rows_mismatch": vec_rows is not None and vec_rows != rows,
                        "indexed_files": len(indexed), "disk_files": len(on_disk),
                        "unindexed_n": len(unindexed), "unindexed": unindexed,
                        "unindexed_temp": temp_new, "gone_n": len(gone), "gone": gone,
                        "drifted_n": len(drifted),
                        "drifted": [{"file": f, "age_s": d} for f, d in drifted[:20]],
                        "temp_in_index": sorted(temp_in),
                        "provenance": prov_state, "provenance_issues": prov_issues,
                        "orphan_tmp_provenance": sorted(orphans)}

    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", a.out)
    for z, r in report.items():
        if "error" in r:
            print(f"\n[{z}] {r['error']}")
            continue
        print(f"\n[{z}] 索引 {r['index_mtime']} rows={r['rows']} vec_rows={r['vec_rows']}"
              f"{' MISMATCH' if r['rows_mismatch'] else ''}")
        print(f"  磁盘合格文件 {r['disk_files']} / 已入库 {r['indexed_files']}"
              f" → 未入库 {r['unindexed_n']}（其中临时件 {len(r['unindexed_temp'])}）"
              f" / 索引里有但磁盘没了 {r['gone_n']}")
        print(f"  内容漂移（已入库但晚于索引被改）: {r['drifted_n']}")
        print(f"  索引里的临时件: {len(r['temp_in_index'])}")
        print(f"  provenance: {r['provenance']}"
              + ("" if not r['provenance_issues'] else " | " + "; ".join(r['provenance_issues'])))
        if r["orphan_tmp_provenance"]:
            print(f"  孤儿临时 provenance: {r['orphan_tmp_provenance']}")


if __name__ == "__main__":
    main()
