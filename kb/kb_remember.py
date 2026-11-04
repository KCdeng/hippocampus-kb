"""kb_remember.py — A1 闭环写入：把 agent 结论持久化进海马体 mem 区。

用法:
  python kb_remember.py add --title "..." --body "..." [--confidence 0.9] [--sources a.md,b.md] [--task "..."]
  python kb_remember.py add --title "..." --body-file note.md ...      # 长文用文件
  python kb_remember.py correct --id mem-0003 --body "新结论 ..."      # 旧记忆标记 superseded，写新版
  python kb_remember.py list
"""
import argparse, datetime, glob, json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_atomic

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research/hippocampus
MEM = os.environ.get("KB_MEM_DIR") or os.path.join(BASE, "kb", "mem")  # 供隔离测试指向副本
CORPUS = os.path.join(MEM, "corpus")
PY = sys.executable
MODEL = os.environ.get("KB_MODEL", "bge-m3-onnx")  # 与主引擎一致：mem/vec.npy 现为 m3 嵌入（2026-11-01 验证），改模型须两边同步

FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)

def parse_fm(path):
    head = open(path, encoding="utf-8").read(2000)
    m = FM_RE.match(head)
    if not m:
        return {}
    d = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d

def next_id():
    """glob+max+1 在并发会话下会撞号（后写静默覆盖前写）。用文件锁串行化。"""
    import msvcrt
    os.makedirs(CORPUS, exist_ok=True)
    lock = open(os.path.join(CORPUS, ".idlock"), "a+")
    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
    try:
        ids = [int(re.search(r"mem-(\d+)", os.path.basename(p)).group(1))
               for p in glob.glob(os.path.join(CORPUS, "mem-*.md"))]
        mid = f"mem-{max(ids, default=0) + 1:04d}"
        while os.path.exists(os.path.join(CORPUS, f"{mid}.md")):
            ids.append(int(mid.split("-")[1])); mid = f"mem-{max(ids) + 1:04d}"
        return mid, lock          # 调用方写完 md 后再 release（锁住到写入落地）
    except Exception:
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close(); raise

def _release(lock):
    import msvcrt
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    lock.close()

def promote_vec(tmp, vec):
    """把刚嵌好的临时向量原子换成生产档，并让它旁边的 provenance sidecar 一起落地。
    实现已抽到 kb_atomic.promote（chunker/build_vec/index 走同一份），这里只保留历史名：
    build_vec.py 把溯源写在 `<out>.provenance.json`，只改名 vec 的话溯源会永远留在
    `vec.npy.new.<pid>.npy.provenance.json` 这种孤儿名里，生产档反倒没有溯源；
    旧版溯源跟着旧向量进 .prev 归档，不留给新向量——描述错代的溯源比没有溯源更糟。"""
    kb_atomic.promote(tmp, vec, keep=3)

def rebuild_mem():
    """mem 区三件（chunks/bm25/vec）全量重建。
    旧实现把 vec.npy 挪成固定名 .old 并在成功后删除：中途崩溃或并发写入会毁掉唯一好副本。
    改为：嵌到临时文件 → 校验行数 → 原子替换 → 上一版按时间戳留档（只保最近 3 份）。
    切块同样先落临时文件：语料被清空/切块异常时 chunker 会写出 0 行，而 build_vec 跟着
    产出 0 行向量，"vec 行数 == chunks 行数"这条校验会 0==0 通过，整区记忆静默消失。"""
    env = dict(os.environ, KB_DIR=MEM, KB_MODEL=MODEL)
    chunks = os.path.join(MEM, "chunks.jsonl")
    tmp_chunks = chunks + f".new.{os.getpid()}"
    vec = os.path.join(MEM, "vec.npy")
    try:
        subprocess.run([PY, os.path.join(BASE, "kb", "chunker.py"), "--root", CORPUS,
                        "--out", tmp_chunks],
                       env=env, check=True, stdout=subprocess.DEVNULL)
        n = sum(1 for l in open(tmp_chunks, encoding="utf-8") if l.strip())
        if n == 0:
            raise SystemExit(f"mem 切块得到 0 行（语料 {CORPUS} 空了或切块异常），"
                             f"拒绝覆写现有索引——现有 chunks/vec 保持不动")
        kb_atomic.promote(tmp_chunks, chunks, keep=3)   # 旧 chunks 一并带时间戳留档，与 vec 同等待遇
    except BaseException:
        if os.path.exists(tmp_chunks):
            os.remove(tmp_chunks)
        raise
    tmp = vec + f".new.{os.getpid()}.npy"   # 必须以 .npy 结尾：np.save 会给无后缀名补 .npy
    # 生产主库向量用纯 text 约定（build_m3_index.py）；mem 区对齐，别用 index.py 的 heading 约定
    subprocess.run([PY, os.path.join(BASE, "kb", "build_vec.py"),
                    "--chunks", chunks, "--out", tmp],
                   env=dict(env, KB_EMBED_HEADING="0"), check=True, stdout=subprocess.DEVNULL)
    import numpy as np
    got = np.load(tmp)
    if got.shape[0] != n:
        os.remove(tmp)
        for junk in glob.glob(tmp + "*"):
            os.remove(junk)
        raise SystemExit(f"mem 重嵌行数不符：vec={got.shape[0]} chunks={n}（旧向量保留，未替换）")
    promote_vec(tmp, vec)      # .prev 归档（向量 + 它的溯源）成对保留，promote 里压在 3 份
    subprocess.run([PY, os.path.join(BASE, "kb", "index.py")], env=env, check=True,
                   stdout=subprocess.DEVNULL)

def hint_similar(mid):
    """写后自检：新条目是否与现有活跃记忆构成重复/修订候选。只提示，不拦写入——
    矛盾判定要人做；这里的作用是让"同一件事写第二遍/纠错没走 correct"当场可见。"""
    try:
        import mem_audit as MA
        active = [e for e in MA.load_entries() if not e["superseded_by"]]
        rows, vec = MA.vec_rows()
        dups, conf, _ = MA.candidates(active, rows, vec, 0.90, 0.70)
    except Exception as e:
        print(f"（相似度自检跳过：{type(e).__name__}: {e}）")
        return
    hits = [p for p in dups + conf if mid in (p["a"], p["b"])]
    if not hits:
        return
    print("⚠ 与现有记忆构成候选（同题/疑似冲突）：")
    for p in hits:
        other = p["b"] if p["a"] == mid else p["a"]
        print(f"   {other}  cos={p['cos']} 标题J={p['title_j']} 正文包含={p['cont']}  {p['reason']}")
    print("   裁决跑 python kb/mem_audit.py；合并跑 python kb/mem_audit.py merge <ids> … --apply")


def write_md(mid, title, body, confidence, sources, task):
    os.makedirs(CORPUS, exist_ok=True)
    path = os.path.join(CORPUS, f"{mid}.md")
    fm = (f"---\nid: {mid}\ncreated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
          f"confidence: {confidence}\nsuperseded_by:\n"
          f"sources: {json.dumps(sources, ensure_ascii=False)}\ntask: {json.dumps(task or '', ensure_ascii=False)}\n---\n")
    open(path, "w", encoding="utf-8").write(fm + f"# {title}\n\n{body.strip()}\n")
    return path

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--title", required=True); a.add_argument("--body"); a.add_argument("--body-file")
    a.add_argument("--confidence", type=float, default=0.8); a.add_argument("--sources", default="")
    a.add_argument("--task", default="")
    c = sub.add_parser("correct")
    c.add_argument("--id", required=True); c.add_argument("--body"); c.add_argument("--body-file")
    c.add_argument("--confidence", type=float, default=0.8); c.add_argument("--sources", default="")
    c.add_argument("--task", default="")
    sub.add_parser("list")
    args = ap.parse_args()

    if args.cmd == "list":
        for p in sorted(glob.glob(os.path.join(CORPUS, "mem-*.md"))):
            fm = parse_fm(p)
            first = open(p, encoding="utf-8").read().split("# ", 1)[-1].split("\n", 1)[0]
            print(f"{fm.get('id','?')} conf={fm.get('confidence','?')} "
                  f"{'[SUPERSEDED->'+fm['superseded_by']+']' if fm.get('superseded_by') else ''} {first}")
        return

    if args.confidence < 0.5:
        ap.error(f"confidence={args.confidence} < 0.5 不入库（规程：宁可让智能体说'不确定'）")
    body = args.body or (open(args.body_file, encoding="utf-8").read() if args.body_file else None)
    if not body:
        ap.error("需要 --body 或 --body-file")
    sources = [s for s in args.sources.split(",") if s]

    mid, lock = next_id()          # 锁持有到 md 落地，防并发会话撞号后互相覆盖
    try:
        if args.cmd == "add":
            title = args.title
        else:  # correct: 旧文件打 superseded 指针
            old_path = os.path.join(CORPUS, f"{args.id}.md")
            if not os.path.exists(old_path):
                ap.error(f"找不到 {old_path}")
            old = open(old_path, encoding="utf-8").read()
            if "superseded_by:" not in old:
                raise SystemExit(f"{args.id} 的 frontmatter 缺 superseded_by:，拒绝改写（否则指针静默丢失）")
            old = old.replace("superseded_by:", f"superseded_by: {mid}", 1)
            open(old_path, "w", encoding="utf-8").write(old)
            fm = parse_fm(old_path)
            title = old.split("# ", 1)[1].split("\n", 1)[0] + "（修订版）"
            sources = sources or json.loads(fm.get("sources") or "[]")
            print(f"superseded {args.id} -> {mid}")
        path = write_md(mid, title, body, args.confidence, sources, args.task)
    finally:
        _release(lock)
    rebuild_mem()
    print("written:", os.path.relpath(path, BASE))
    hint_similar(mid)
    print('提示：8310 已支持热重载——POST /reload {"zone":"mem"} 立即可见，或等自动监视（≤10s）追上，无需重启。')

if __name__ == "__main__":
    main()
