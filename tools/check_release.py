r"""check_release.py — 发布目录净化闸门（可独立跑，也被 test_release_gate.py 调用）。

跑法：
    python tools/check_release.py <dist目录>        # 有发现即退出码 1
在打包之后、在 dist 里跑构建（chunker/index/build_vec）之前跑——闸门查的是「我们要发什么」，
构建产物出现在工作副本里正常，出现在发布快照里就是事故。

两类判据分开：
  BUILTIN —— 任何人都该拦下来的：本机绝对路径、账号目录、语料正文与派生索引、日志、
             记忆文档、模型权重、依赖目录、超过 1MB 的单文件、非 UTF-8 二进制。
  EXTRA   —— 作者本地表 release_local.json 里的 banned（用户名、家目录名、第三方厂商名
             这类「只在这台机器/这个身份上才算泄漏」的字面量）。这份表不随仓发布，
             所以 clone 的人跑到的是 BUILTIN 全集。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_TABLE = os.path.join(HERE, "release_local.json")
MAX_FILE_BYTES = 1024 * 1024

# 通用形态：本机绝对路径与账号目录。由 _leak_re 宽容 1~2 个反斜杠/斜杠
# （源码里是单反斜杠，JSON 里是双反斜杠，两种都得抓）。
# 字面量必须由片段拼出：本文件自己也走在闸门底下，写成完整路径会被自己举报。
_BS, _SL = "\\", "/"
BUILTIN_LEAKS = [f"C{_BS}Users", f"D{_BS}QODER", f"D{_BS}下载",
                 f"E{_BS}Users", f"{_SL}Users{_SL}"]

# 永远不该出现在发布件里的路径形态：语料正文、派生索引、日志、记忆文档、模型、依赖、实验目录
FORBIDDEN_PATHS = [
    r"(^|/)chunks",                              # 切块正文=语料本身
    r"(^|/)bm_vocab\.json$",                     # 从语料学出来的词表（实测连本地用户名都会成词）
    r"(^|/)eval_per_query",                      # 逐题明细带 gold 文件名
    r"\.npy$", r"\.onnx$", r"\.zip$", r"\.gz$",
    r"\.prev\.", r"\.bak$", r"\.new\.",
    r"\.log$", r"\.err$", r"\.out$",
    r"(^|/)query_log\.jsonl$", r"(^|/)reload_history\.jsonl$",
    r"\.provenance\.json$",
    r"(^|/)corpus_(comp|zh)/", r"(^|/)kb_(zh|comp)/",
    r"(^|/)(venv|src|snapshots|scratch|models|ab_zh|ab_comp)(/|$)",
    r"(^|/)kb/(mem|epi)/",                        # 记忆区与复盘区整目录（corpus 与派生索引都在这）
    r"(^|/)meta\.json$",                          # index.py 生成，且记录本机路径
]


def _leak_re(literal):
    out = []
    for ch in literal:
        if ch == "\\":
            out.append(r"\\{1,2}")
        elif ch == "/":
            out.append(r"[/\\]{1,2}")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out))


BANNED_RE = [_leak_re(p) for p in BUILTIN_LEAKS]


def is_forbidden_path(rel):
    rel = rel.replace("\\", "/").lower()
    return any(re.search(p, rel) for p in FORBIDDEN_PATHS)


def load_local_banned(path=LOCAL_TABLE):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("banned", [])


def check(dist, extra_banned=None):
    """返回发现列表 [{path, kind, detail}]，空列表即闸门通过。"""
    pats = BANNED_RE + [re.compile(p) for p in
                        list(extra_banned or []) + load_local_banned()]
    findings = []
    for root, dirs, files in os.walk(dist):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, dist).replace("\\", "/")
            if is_forbidden_path(rel):
                findings.append({"path": rel, "kind": "forbidden_path",
                                 "detail": "禁止随仓发布的路径形态"})
                continue
            size = os.path.getsize(full)
            if size > MAX_FILE_BYTES:
                findings.append({"path": rel, "kind": "oversize",
                                 "detail": f"{size} 字节 > 单件上限 {MAX_FILE_BYTES}"})
            with open(full, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                findings.append({"path": rel, "kind": "binary", "detail": "非 UTF-8 文本"})
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                for pat in pats:
                    if pat.search(line):
                        findings.append({"path": rel, "kind": "leak",
                                         "detail": f"L{ln} 命中 {pat.pattern}: {line.strip()[:120]}"})
                        break
    return findings


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                     # noqa: BLE001
        pass
    if len(sys.argv) < 2:
        sys.exit("用法: python check_release.py <dist目录>")
    dist = sys.argv[1]
    res = check(dist)
    if res:
        print(f"发布闸门 FAIL：{len(res)} 项")
        for f in res:
            print(f"  [{f['kind']}] {f['path']}: {f['detail']}")
        sys.exit(1)
    n = len([1 for root, _d, files in os.walk(dist) for f in files])
    print(f"发布闸门 PASS：{dist} 共 {n} 个文件，无泄漏形态、无禁止路径、无超限件")


if __name__ == "__main__":
    main()
