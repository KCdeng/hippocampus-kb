r"""pack_release.py — 把 hippocampus 工作树打包成可公开发布的 dist 目录。

跑法：
    python tools/pack_release.py <输出目录>       # 然后 python tools/check_release.py <输出目录>

为什么要一份打包脚本而不是直接把工作树 push 上去：工作树里 **语料正文、派生索引、查询日志、
记忆文档、实验目录** 与引擎代码混在同一个目录里，人和机器都分不清哪些该发。这里用白名单
把它们物理隔开——**默认什么都不发**，要发必须点名。

两件事分开存放：
  - 名单（本文件内）＝可公开的编辑决定，随仓一起发布，别人能看到"这个仓库为什么是这些文件"；
  - `release_local.json` ＝替换表，含作者本机的用户名、家目录、第三方厂商名，**不随仓发布**
    （它自己就是那份最该被闸门拦住的泄漏）。仓库里放的是 `release_local.example.json`。
替换表按"必须命中"校验：源文件改名或字面量漂移时宁可打包报错，也不要静默少替换一处、
把本机路径发出去。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
LOCAL_TABLE = os.path.join(HERE, "release_local.json")

# 检索与索引链路
ENGINE = ["chunker.py", "index.py", "build_vec.py", "kb_search.py", "fuse.py",
          "m3embed.py", "kb_atomic.py", "kb_log_archive.py", "write_provenance.py"]
# 常驻服务与闭环写入
OPS = ["kb_serve.py", "kb_remember.py", "kb_epilogue.py", "mem_audit.py",
       "coverage_audit.py", "kb_health_probe.py", "kb_soak_test.py", "replay.py", "eval.py"]
# 生产件守门员（48 项，见 README 的验收口径）
TESTS = ["test_atomic_write.py", "test_log_rotation.py", "test_corpus_lag.py",
         "test_tail_provenance.py", "test_o0_query_reuse.py"]
SCRIPTS = ["_start.ps1", "_stop.ps1", "_k.ps1"]
# V3 配方里的两个词典件：词面档与别名扩写是 81.8% 的组成部分，且不是语料正文
DATA = ["alias_map.json", "jieba_userdict.txt"]
DOCS = ["CHUNKING-EVIDENCE.md", "VERSION-PROFILE.md", "E2E-AB-REPORT.md",
        "writeback_design.md"]
# 私有语料上的实测记录（只有分数、块数与 qid，无正文、无文件名）另档存放：
# 发布件里 kb/baseline.json 必须是**示例语料**的闸门，否则外人 clone 完跑 eval 一定 FAIL。
EXPLICIT = [("kb/baseline.json", "kb/baseline_privatedomain.json")]
TOP_DOCS = ["RESEARCH.md", "CORPUS-COMP-REPORT.md"]
TOOLS = ["pack_release.py", "check_release.py", "test_release_gate.py",
         "release_local.example.json"]


def manifest():
    """返回 [(源相对路径, 发布相对路径)]。白名单点名，不在名单里的一律不发。"""
    m = [("kb/" + f, "kb/" + f) for f in ENGINE + OPS + TESTS + SCRIPTS + DATA + DOCS]
    m += EXPLICIT
    m += [(f, f) for f in TOP_DOCS]
    m += [("tools/" + f, "tools/" + f) for f in TOOLS]
    m += [("release/" + s, d) for s, d in RELEASE_ONLY]
    m += [("release/corpus/" + f, "corpus/" + f)
          for f in sorted(os.listdir(os.path.join(SRC, "release", "corpus")))]
    return m


# release/ 下放的是"只为发布写的内容"，落到仓库里的目标路径
RELEASE_ONLY = [
    ("README.md", "README.md"),
    ("README_en.md", "README.en.md"),
    ("LICENSE", "LICENSE"),
    ("requirements.txt", "requirements.txt"),
    ("gitignore", ".gitignore"),
    ("eval_set_example.json", "kb/eval_set.json"),
    ("baseline_example.json", "kb/baseline.json"),
    ("corpus_README.md", "corpus/README.md"),
]


def load_table(path=LOCAL_TABLE):
    if not os.path.exists(path):
        raise SystemExit(f"缺少替换表 {path}（本地文件，不随仓发布；形态见 "
                         f"tools/release_local.example.json）")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_rewrites(text, pairs, dst, report):
    for old, new in pairs:
        n = text.count(old)
        if not n:
            report["unmatched"].append(f"{dst}: {old[:70]}")
            continue
        text = text.replace(old, new)
    return text


def pack(out_dir, table=None):
    table = table or load_table()
    glob_pairs = table.get("global", [])
    per_file = table.get("files", {})
    report = {"files": [], "bytes": 0, "missing": [], "unmatched": []}
    for src_rel, dst_rel in manifest():
        sp = os.path.join(SRC, src_rel.replace("/", os.sep))
        if not os.path.isfile(sp):
            report["missing"].append(src_rel)
            continue
        with open(sp, encoding="utf-8") as f:
            text = f.read()
        text = apply_rewrites(text, per_file.get(dst_rel, []), dst_rel, report)
        for old, new in glob_pairs:      # global 尽力而为：某文件里没这个字面量是正常的
            text = text.replace(old, new)
        dp = os.path.join(out_dir, dst_rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        with open(dp, "w", encoding="utf-8", newline="\n") as w:
            w.write(text)
        report["files"].append(dst_rel)
        report["bytes"] += len(text.encode("utf-8"))
    if report["missing"] or report["unmatched"]:
        raise SystemExit("打包中止：\n  缺源文件 " + str(report["missing"]) +
                         "\n  替换表失配 " + str(report["unmatched"]))
    return report


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                     # noqa: BLE001
        pass
    if len(sys.argv) < 2:
        sys.exit("用法: python pack_release.py <输出目录>")
    out = os.path.abspath(sys.argv[1])
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"输出目录必须为空或不存在（避免把上一次的构建产物打进发布件）：{out}")
    os.makedirs(out, exist_ok=True)
    r = pack(out)
    print(f"打包完成：{len(r['files'])} 个文件、{r['bytes'] / 1e6:.2f} MB → {out}")
    print("下一步：python tools/check_release.py", out)


if __name__ == "__main__":
    main()
