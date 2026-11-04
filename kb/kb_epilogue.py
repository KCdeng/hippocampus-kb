"""kb_epilogue.py — 把上下文压缩摘要蒸馏成"周期性经验教训"，进海马体候选区（epi）。

  python kb_epilogue.py scan                 # 只算不写：列出每个周期的机器信号与分数
  python kb_epilogue.py ingest [--limit N]   # 蒸馏落盘到 kb/epi/corpus/ 并重建 epi 索引
  python kb_epilogue.py promote              # 复盘闸门过了才允许进生产查询链路

为什么不是"把摘要原文入库"：压缩摘要里带工具输出的原始片段（命令、日志、可能的密钥），
整段入索引既污染召回又泄露风险。**只蒸馏条目**、且蒸馏是确定性规则（不靠模型自评）。

打分口径（全部来自 transcript 的可机器读取字段，可复核）：
  x1 一次做对率 = 1 - tool_error/tool_call
  x2 返工密度   = 1 - min(1, human_turn/tool_call * 5)      # 用户反复插话≈方向没定住
  x3 收口信号   = 该周期摘要里是否出现闸门/终评/PASS|FAIL 结论词
  score = 100 * (0.45*x1 + 0.30*x2 + 0.25*x3)
confidence = 0.30 + 0.65 * score/100，低于 0.5 不入库（沿用闭环红线）。
"""
import argparse, glob, json, os, re, sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
EPI = os.path.join(HERE, "epi")
CORPUS = os.path.join(EPI, "corpus")
SRC_DIR = os.environ.get("EPI_SRC") or sys.exit("请设 EPI_SRC 指向智能体会话记忆目录（transcript 所在）")
MARKER = os.path.join(EPI, "PROMOTED")
GATE_TOKENS = ("闸门", "终评", "过闸", "PASS", "FAIL", "基线", "回归")
MAX_BODY = 1200          # 单条蒸馏正文上限，超出截到整句
PER_SECTION = 6          # 每节最多留几条要点：防止"事实复述节"吃光预算、把教训挤掉
# 只要"行为类结论"节。Key Technical Concepts / Current Work 是对文档与状态的复述，
# 已在主库与 mem 区，重复入库就是近重复块（已实测会稀释 dense 召回）。
KEEP_SECTIONS = ("errors and fixes", "problem solving", "findings", "verdict", "decision",
                 "conclusion", "lesson", "root cause", "教训", "结论", "决策", "归因",
                 "修复", "收口", "验收", "实测", "发现")
SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\b(\s*[:=]\s*)\S+"),
    re.compile(r"[A-Za-z0-9+/]{24,}={1,2}"),                    # base64 密串：必须带填充，
    re.compile(r"(?i)\b(sk-|ghp_|gho_|xox[baprs]-|AKIA)[A-Za-z0-9]{8,}"),   # 常见凭据前缀
    re.compile(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END|$)"),
]


def _text_of(msg):
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return ""


def _is_tool_turn(d):
    c = (d.get("message") or {}).get("content")
    if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
        return True
    return "sourceToolAssistantUUID" in d or d.get("toolUseResult") is not None


def _err_in_tool_turn(d):
    c = (d.get("message") or {}).get("content")
    if isinstance(c, list):
        return any(isinstance(x, dict) and x.get("type") == "tool_result" and x.get("is_error") for x in c)
    return bool(d.get("isError"))


def _tool_calls(d):
    c = (d.get("message") or {}).get("content")
    if isinstance(c, list):
        return sum(1 for x in c if isinstance(x, dict) and x.get("type") == "tool_use")
    return 0


def read_session(path):
    """一次会话 -> 周期列表，每周期带机器信号与摘要原文。"""
    cycles, cur = [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t == "system" and d.get("subtype") == "compact_boundary":
                if cycles:
                    cycles[-1]["meta"] = d.get("compactMetadata") or {}
                continue
            if d.get("isCompactSummary") and t == "system":
                continue
            if d.get("isCompactSummary") and t == "user":
                cur = {"summary": _text_of(d.get("message")), "ts": d.get("timestamp"),
                       "meta": {}, "tool_call": 0, "tool_error": 0, "human": 0,
                       "sessionId": d.get("sessionId"), "cwd": d.get("cwd")}
                cycles.append(cur)
                continue
            if t == "system" and d.get("subtype") == "compact_boundary":
                if cycles:
                    cycles[-1]["meta"] = d.get("compactMetadata") or {}
                continue
            if cur is None:
                continue
            cur["tool_call"] += _tool_calls(d)
            if t == "user":
                if _is_tool_turn(d):
                    cur["tool_error"] += 1 if _err_in_tool_turn(d) else 0
                elif not d.get("isMeta") and d.get("userType") == "external":
                    cur["human"] += 1
    return cycles


def score_cycle(c):
    tc = max(1, c["tool_call"])
    x1 = 1 - min(1.0, c["tool_error"] / tc)
    x2 = 1 - min(1.0, c["human"] / tc * 5)
    x3 = min(1.0, sum(1 for tok in GATE_TOKENS if tok in c["summary"]) / 2)
    s = 100 * (0.45 * x1 + 0.30 * x2 + 0.25 * x3)
    return round(s, 1), {"x1_once_right": round(x1, 3), "x2_rework": round(x2, 3),
                         "x3_closure": round(x3, 3), "tool_call": c["tool_call"],
                         "tool_error": c["tool_error"], "human_turns": c["human"]}


def redact(s):
    s = SECRET_PATTERNS[0].sub(lambda m: m.group(1) + m.group(2) + "<REDACTED>", s)
    for pat in SECRET_PATTERNS[1:]:
        s = pat.sub("<REDACTED>", s)
    return s


def distill(summary):
    """按节取结论：先砍"这是上一段对话的延续"这类样板前言，再逐节判定；
    未命中结论节的整节丢弃（请求复述、文件清单、逐条消息都属高风险且无结论价值）。"""
    body = re.sub(r"```.*?```", " ", summary, flags=re.S)
    m = re.search(r"(?im)^\s*\**\s*Summary\s*[:：]\s*\**\s*$", body)
    if m:
        body = body[m.end():]
    head_re = re.compile(r"^(?:#{1,6}\s+(.*)$|\**(\d{1,2})[.)]\s+\*{0,2}([^*\n]{3,70}?)\*{0,2}:?\*{0,2}$)")
    keep, on, taken = [], False, 0
    for ln in body.splitlines():
        h = head_re.match(ln.strip())
        if h:
            title = (h.group(1) or h.group(3) or "").strip()
            on = any(k in title.lower() for k in KEEP_SECTIONS)
            taken = 0
            if on:
                keep.append("## " + title)
            continue
        if not on or taken >= PER_SECTION:
            continue
        s = ln.strip()
        if not s or re.match(r"^[|`~-]{3,}$", s):
            continue
        taken += 1
        if len(s) > 220:                                  # 长行多是原文粘贴/清单，压缩成要点
            s = s[:200].rsplit("，", 1)[0].rstrip(",，。;；") + "…"
        keep.append(s)
    out = redact("\n\n".join(keep)).strip()
    if len(out) > MAX_BODY:
        out = out[:MAX_BODY]
        cut = max(out.rfind("。"), out.rfind("."), out.rfind("\n"))
        out = out[:cut + 1] if cut > 400 else out
    return out


def frontmatter(cid, r, sc, br):
    conf = round(0.30 + 0.65 * sc / 100, 2)
    return "\n".join([
        "---", f"id: {cid}", "kind: epilogue", f"session: {r['sessionId']}",
        f"cwd: {r['cwd']}", f"cycle: {r['cycle']}", f"cycle_ts: {r['ts']}",
        f"tokens: {r['tokens']}", f"messages_summarized: {r['msgs']}",
        f"score: {sc}", f"score_breakdown: {json.dumps(br, ensure_ascii=False)}",
        f"confidence: {conf}", "status: candidate", "source: compaction-summary",
        f"made: {datetime.now().isoformat(timespec='seconds')}", "---", ""])


def iter_sessions(src):
    for p in sorted(glob.glob(os.path.join(src, "*.jsonl")), key=os.path.getmtime):
        yield p


def collect(limit=None):
    rows = []
    for p in iter_sessions(SRC_DIR):
        for i, c in enumerate(read_session(p), 1):
            if len(c["summary"]) < 400:
                continue
            sc, br = score_cycle(c)
            cid = f"epi-{(c['sessionId'] or os.path.basename(p))[:8]}-{i:02d}"
            meta = c["meta"]
            rows.append({"cid": cid, "path": p, "cycle": i, "score": sc, "br": br,
                         "summary": c["summary"], "ts": c["ts"],
                         "sessionId": c["sessionId"] or os.path.basename(p), "cwd": c["cwd"] or "",
                         "tokens": f"{meta.get('preTokens', '?')}->{meta.get('postTokens', '?')}",
                         "msgs": meta.get("messagesSummarized", "?")})
            if limit and len(rows) >= limit:
                return rows
    return rows


def cmd_scan(args):
    rows = collect(args.limit)
    print(f"会话目录 {SRC_DIR}\n周期条目 {len(rows)} 条\n")
    print(f"{'id':22s} {'score':>6s} {'工具':>5s} {'出错':>5s} {'人工':>5s} {'tokens':>16s} {'msgs':>5s}")
    for r in rows:
        print(f"{r['cid']:22s} {r['score']:6.1f} {r['br']['tool_call']:5d} {r['br']['tool_error']:5d} "
              f"{r['br']['human_turns']:5d} {r['tokens']:>16s} {str(r['msgs']):>5s}")
    ok = [r for r in rows if 0.30 + 0.65 * r["score"] / 100 >= 0.5]
    print(f"\n可入库(confidence>=0.5): {len(ok)}/{len(rows)}")


def cmd_ingest(args):
    rows = collect(args.limit)
    os.makedirs(CORPUS, exist_ok=True)
    written, skipped = [], []
    for r in rows:
        conf = round(0.30 + 0.65 * r["score"] / 100, 2)
        if conf < 0.5:
            skipped.append((r["cid"], f"confidence {conf} < 0.5"))
            continue
        if r["br"]["x3_closure"] <= 0:
            skipped.append((r["cid"], "无收口信号（周期里没有闸门/终评/PASS|FAIL 结论）"))
            continue
        body = distill(r["summary"])
        if len(body) < 200:
            skipped.append((r["cid"], f"蒸馏后仅 {len(body)} 字，无结论价值"))
            continue
        out = os.path.join(CORPUS, r["cid"] + ".md")
        text = frontmatter(r["cid"], r, r["score"], r["br"])
        open(out, "w", encoding="utf-8").write(
            text + f"# 周期复盘 {r['cid']}\n\n" + body + "\n")
        written.append((r["cid"], round(0.30 + 0.65 * r["score"] / 100, 2), len(body)))
    print(f"写入 {len(written)} 条 -> {CORPUS}；跳过 {len(skipped)} 条")
    for cid, conf, n in written[:10]:
        print(f"  ok {cid} conf={conf} body={n}")
    for cid, why in skipped[:10]:
        print("  skip", cid, why)
    if written and not args.no_reindex:
        import subprocess
        py = sys.executable
        env = {**os.environ, "KB_MODEL": "bge-m3-onnx", "KB_DIR": EPI}   # 与 mem 同后端，禁止混档
        run = lambda cmd: subprocess.run(cmd, check=True, env=env)
        run([py, os.path.join(HERE, "chunker.py"), "--root", CORPUS,
             "--out", os.path.join(EPI, "chunks.jsonl"), "--force"])
        run([py, os.path.join(HERE, "build_vec.py"), "--chunks", os.path.join(EPI, "chunks.jsonl"),
             "--out", os.path.join(EPI, "vec.npy")])
        run([py, os.path.join(HERE, "index.py")])
    print("epi 是候选区：闸门过了再 `promote`，届时 fuse/8310 才把它接进 tiered 尾部。")


def cmd_promote(args):
    rows = collect(None)
    print(f"promote 前必须人工过闸门：{len(rows)} 条候选，epi 复盘题命中率要 >= 目标档。")
    if not args.confirm:
        sys.exit("未加 --confirm，拒绝写 PROMOTED 标记（这条闸门是给人走的，不是脚本自己走的）")
    open(MARKER, "w", encoding="utf-8").write(datetime.now().isoformat(timespec="seconds") + "\n")
    print(f"已写 {MARKER}；重启 8310 后 epi 进 tiered 尾部。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "ingest", "promote"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-reindex", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    {"scan": cmd_scan, "ingest": cmd_ingest, "promote": cmd_promote}[a.cmd](a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
