"""chunker.py — 把知识库 Markdown 按标题层级切成带元数据的块。
用法: python chunker.py [--root <本地工作区>] [--out kb/chunks.jsonl]
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_atomic

EXCLUDE_ANY = {"node_modules", ".git", "__pycache__", "venv", "_pdf_snips", "preview"}
EXCLUDE_TOP = {"research"}      # 只排工作区顶层的 research/（源码仓），嵌套的同名业务目录要入库
MAX_CHARS = 800          # 单块上限，超出按空行二次切
MIN_CHARS = 40           # 太短的块并入邻块
ROW_ATOMIC = os.environ.get("KB_ROW_ATOMIC", "0") == "1"   # 1=超长表格一行一块（V4 已证伪，hit@5 −3.6pp）
ROW_NL = os.environ.get("KB_ROW_NL", "0") == "1"           # 1=每行追加「列名=值」标签（V5b，合成标签有失真风险）
CTX_CHARS = int(os.environ.get("KB_CTX_CHARS", "0"))        # 表格/代码块前缀的上方散文上限，0=关闭（默认）

def iter_md_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        top = os.path.abspath(dirpath) == os.path.abspath(root)
        block = EXCLUDE_ANY | (EXCLUDE_TOP if top else set())
        dirnames[:] = [d for d in dirnames if d not in block]
        for fn in filenames:
            if fn.lower().endswith((".md", ".txt")) or (fn.lower().endswith(".py") and "cabinet_layout" in dirpath):
                yield os.path.join(dirpath, fn)

def split_headings(lines):
    """返回 [(line_no, level, heading_path_str)] 对应的分节起点。"""
    sections = []  # (start_line, heading_path, body_lines)
    stack = []     # [(level, text)]
    cur_start, cur_path, cur_body = 1, [], []
    for i, ln in enumerate(lines, 1):
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            if cur_body or stack:
                sections.append((cur_start, "/".join(cur_path), cur_body))
            level = len(m.group(1)); text = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            cur_path = [t for _, t in stack]
            cur_start, cur_body = i, [ln]
        else:
            cur_body.append(ln)
    sections.append((cur_start, "/".join(cur_path), cur_body))
    return sections

def _ctx_of(prev, cap):
    """表格/代码块上方紧邻散文的最后几句（句子对齐，不截半句）。标题或表格本身不作前缀。"""
    if cap <= 0 or not prev:
        return ""
    s = prev.strip()
    if s.startswith("#") or s.startswith("|"):
        return ""
    out, n = [], 0
    for sent in reversed([x for x in re.split(r"(?<=[。；;！？!?])", s) if x.strip()]):
        if n + len(sent) > cap:
            break
        out.insert(0, sent)
        n += len(sent)
    return "".join(out).strip()

def _cols_of(header_line):
    """表头行拆列名，用于给数据行生成「列名=值」标签。
    空列名必须保留占位：过滤后列表会左移，zip 到数据行就整体错列（armB 的列错配就是这么来的）。"""
    return [c.strip() for c in header_line.strip().strip("|").split("|")]

SEP_RE = re.compile(r"^\|?[\s:|-]+$")

def _is_sep(line):
    """markdown 表格分隔行：| --- | :---: | 这类。"""
    s = line.strip()
    return s.startswith("|") and "-" in s and bool(SEP_RE.match(s))

def _nl_row(cols, ln):
    """保留原始竖线行（金标准锚点按原文匹配，不能改），其后追加一行列名标签。"""
    if not cols:
        return ln
    vals = [c.strip() for c in ln.strip().strip("|").split("|")]
    pairs = [f"{c}={v}" for c, v in zip(cols, vals)
             if c and v and v not in ("—", "-", "－") and not set(v) <= set("-: ")]
    if len(pairs) < 2:
        return ln
    return ln + "\n" + " ".join(pairs)

def _line_groups(p, maxc, ctx="", base=0):
    """句子切不动的超长段按行切。返回 [(块文本, 段内字符偏移)]。
    偏移指向首条表格/代码行（不含 ctx 散文前缀），由调用方换算成原文行号。
    表格（有表头）：ROW_ATOMIC=1 时一行一块；=0 时按行组装袋（V4 口径），表头每块重复。
    ROW_NL=1 时每个数据行追加「列名=值」标签行（保留原始竖线行，金标准锚点按原文匹配）。
    代码/其它：无表头，仍按行组装到预算内。
    """
    lines, pos = [], 0
    for l in p.split("\n"):
        if l.strip():
            lines.append((pos, l))
        pos += len(l) + 1
    header = []
    # 表头必须"首行 | + 第二行是分隔行"才成立。旧实现只认前两行以 | 开头，
    # 没分隔行的两行小表会把首条数据行当表头吞掉（此后不再出现在任何块里）。
    if (len(lines) >= 3 and lines[0][1].startswith("|") and _is_sep(lines[1][1])
            and sum(1 for _, l in lines if l.startswith("|")) >= 3):
        header = lines[:2]
    pre = (ctx + "\n") if ctx else ""
    rows = lines[len(header):]
    if header and ROW_NL:
        cols = _cols_of(header[0][1])
        rows = [(i, _nl_row(cols, r)) for i, r in rows]
    head_text = [l for _, l in header]

    hsize = sum(map(len, head_text))
    if header and ROW_ATOMIC:            # 一行一块，表头重复且不计预算
        head = "\n".join(head_text)
        return [(pre + head + "\n" + r, base + i) for i, r in rows]

    out, cur, size = [], list(head_text), hsize
    cur_start = header[0][0] if header else (rows[0][0] if rows else 0)
    for i, r in rows:
        if size + len(r) + 1 > maxc and len(cur) > len(header):
            out.append((pre + "\n".join(cur), base + cur_start))
            cur, size = list(head_text), hsize
            cur_start = i
        cur.append(r)
        size += len(r) + 1
    if len(cur) > len(header):
        out.append((pre + "\n".join(cur), base + cur_start))
    return out

def chunk_text(text, maxc=MAX_CHARS, minc=MIN_CHARS):
    """按空行分组装袋，超 maxc 再按句号切。返回 [(块文本, 块内首字符偏移)]。
    偏移直接由切分过程携带：旧实现事后按首行文字回搜原文，同一首行在节内重复出现时
    （表头、空节标题）会命中前一处的行号，comp 域实测只有 83% 的块落在正确行。
    """
    paras, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        paras.append((text[pos:m.start()], pos)); pos = m.end()
    paras.append((text[pos:], pos))

    out, buf, buf_off, prev = [], "", 0, ""
    for p, off in paras:
        if not p.strip():
            continue
        if len(buf) + len(p) + 2 <= maxc:
            if not buf:
                buf_off = off
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                # 必须清空：超长段的句切分支里还会再判一次 buf，不清会把同一块写两遍
                out.append((buf, buf_off)); buf = ""
            if len(p) <= maxc:
                buf, buf_off = p, off
            else:  # 超长段按句子切；整句仍超长（表格/代码块）则按行切
                sents, sp = [], 0
                for m in re.finditer(r"(?<=[。；;])", p):
                    sents.append((p[sp:m.end()], sp)); sp = m.end()
                sents.append((p[sp:], sp))
                sb, sb_off = "", 0
                for s, so in sents:
                    if len(s) > maxc:
                        if sb:
                            out.append((sb, sb_off)); sb = ""
                        # CTX 前缀对两条路径都给：RAGFlow 口径是"行组 + 上方散文窗"，
                        # 而 V4 的行组袋死因正是丢掉了同块说明散文（CTX_CHARS=0 时行为不变）
                        pre = _ctx_of(prev, CTX_CHARS)
                        out.extend(_line_groups(s, maxc, pre, base=off + so))
                        continue
                    if len(sb) + len(s) <= maxc:
                        if not sb:
                            sb_off = off + so
                        sb += s
                    else:
                        if sb:
                            out.append((sb, sb_off))
                        sb, sb_off = s, off + so
                buf, buf_off = sb, sb_off
        prev = p
    if buf:
        out.append((buf, buf_off))
    # 合并过短的块（尾块并前，首块并后）
    merged = []
    for c, c_off in out:
        if merged and len(c) < minc:
            merged[-1] = (merged[-1][0] + "\n" + c, merged[-1][1])
        else:
            merged.append((c, c_off))
    if len(merged) > 1 and len(merged[0][0]) < minc:
        merged[1] = (merged[0][0] + "\n" + merged[1][0], merged[0][1])
        merged.pop(0)
    return merged

def emit(rows, out):
    """落盘走原子换档：写到一半炸（磁盘满、序列化异常、Ctrl-C）只毁临时件，
    生产 chunks.jsonl 保持上一版全文——同目录 vec 的行号还能对上。"""
    with kb_atomic.stage(out) as tmp:
        with open(tmp, "w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("KB_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "chunks.jsonl"))
    ap.add_argument("--force", action="store_true",
                    help="块数与 baseline.json 记录的生产基线不符时仍写入（实验目录用）")
    args = ap.parse_args()

    n_files, n_chunks = 0, 0
    rows = []
    for path in sorted(iter_md_files(args.root)):
        n_files += 1
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        except OSError as e:
            print("SKIP", path, e, file=sys.stderr); continue
        # YAML frontmatter 是审计用的元数据，不是正文：入索引会让 id/时间戳这类噪声
        # 参与 BM25 与 dense 打分（epi/mem 区每条都带，污染是成比例的）。
        off = 0
        if lines and lines[0].strip() == "---":
            for i in range(1, min(len(lines), 60)):
                if lines[i].strip() == "---":
                    lines, off = lines[i + 1:], i + 1
                    break
        for start_line, hpath, body in split_headings(lines):
            joined = "\n".join(body)
            text = joined.strip()
            if len(text) < MIN_CHARS:
                continue
            header = text.splitlines()[0]
            lead = len(joined) - len(joined.lstrip())   # text 相对 joined 的首字符位移
            for j, (c, c_off) in enumerate(chunk_text(text) or []):
                # 句切产出的块常以换行开头（断句点在句末，换行落在下一句行首）：
                # 按首字符定位会指到上一行行尾，必须按首个非空白字符定位。
                skip = len(c) - len(c.lstrip())
                ln = start_line + joined.count("\n", 0, lead + c_off + skip) + off
                rows.append({
                    "id": f"{os.path.relpath(path, args.root)}::L{ln}::{j}",
                    "file": os.path.relpath(path, args.root).replace("\\", "/"),
                    "line": ln,
                    "heading": hpath or header[:60],
                    "text": c,
                    "chars": len(c),
                    "kind": "table" if sum(1 for l in c.splitlines()
                                           if l.strip().startswith("|")) >= 3 else "text",
                })
    n_chunks = len(rows)

    # 生产基线守卫：切块粒度一改，同目录 vec 的行序就不再对齐，服务会静默换档。
    # 守卫按"输出目录里有没有 baseline.json"判定，而不是只认主库 kb/——
    # 专利库 kb_zh/、竞品库 kb_comp/ 用的是同一套失效模式，没理由只有主库受保护。
    base_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "baseline.json")
    if not args.force and os.path.exists(base_path):
        want = json.load(open(base_path, encoding="utf-8"))["baseline"]["chunks"]
        if n_chunks != want:
            sys.exit(f"拒绝覆写 {args.out}：本次切出 {n_chunks} 块 ≠ 该库基线 {want} 块。"
                     f"这是另一代索引（同目录 vec 行序失效）。确认要换档请先过闸门，"
                     f"或写实验目录并用 --force/--out 显式指向它。")
    d = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(d, exist_ok=True)          # 实验臂常写到新建目录，不该为此手工 mkdir
    emit(rows, args.out)
    total_kb = os.path.getsize(args.out) / 1e6
    print(f"files={n_files} chunks={n_chunks} out={args.out} ({total_kb:.1f} MB)")

if __name__ == "__main__":
    main()
