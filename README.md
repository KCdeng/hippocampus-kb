# 海马体 · 本地知识库检索引擎

面向**个人服务器自用**的 Markdown 知识库检索：BM25 词面 + ONNX dense 向量双路，
融合排序，常驻 HTTP 服务毫秒级返回，结果带文件名与行号可回查原文。
不依赖数据库、不依赖向量库、不需要 GPU，磁盘上的文件就是唯一真相。

English: a self-hosted, stdlib-served hybrid retrieval engine for Markdown knowledge bases.
Chunking is heading-aware; BM25 and dense (bge-m3 ONNX) scores are fused; every index swap is
atomic and every search-quality change must pass a machine gate recorded in `baseline.json`.
Written for a single operator on a personal server — it binds to 127.0.0.1 and has no auth layer.

```
POST /search {"query": "...", "k": 8}        → 命中的块 + 出处文件 + 行号
GET  /health                                 → ok / error / warn / lag / reload_fail
GET  /stats                                  → 各区块数、索引新鲜度、最近 10 次重载
POST /reload {"zone": "main"}                → 不重启换档（构建成功才替换，失败保留旧档）
```

## 为什么值得看

检索质量这件事最容易糊弄人的地方是：**改坏了没人知道**。所以这个仓库把闸门做成机器件——
`eval.py` 跑完自动与 `kb/baseline.json` 比对，不过闸直接 `exit 1`。
`kb/baseline_privatedomain.json` 里除了通过的基线，还完整记录了一批**被实测证伪的杠杆**
（表格行组袋手术、每行拼列名标签、表格上方散文窗、cross-encoder 精排、语义切块 / 固定 overlap），
每条带分数、耗时和死因。多数是负结果，价值就在这里：照着做能省掉几周。
详细证据在 `CHUNKING-EVIDENCE.md`、`CORPUS-COMP-REPORT.md`、`VERSION-PROFILE.md`。

## 快速上手

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. 准备模型（不随仓发布，见下）：把 bge-m3 的 ONNX 权重放到 models/bge-m3-onnx/
#    需要 model.onnx + tokenizer.json；也可用 M3_ONNX_DIR 指向任意目录
# 2. 语料放进 corpus/（仓库自带 7 篇讲自己的示例文档，切成 34 块，可以先原样跑通再换成自己的）
python kb/chunker.py            # corpus/ → kb/chunks.jsonl
python kb/index.py              # → kb/bm25.npy + kb/bm_vocab.json（确定性，b=0.75）
python kb/build_vec.py --chunks kb/chunks.jsonl --out kb/vec_m3.npy
python kb/eval.py grep,bm25,vec,hybrid,hybrid-v2      # 跑完自动过闸门，不过闸 exit 1

python kb/kb_serve.py            # 前台跑，Ctrl-C 停；或 kb/_start.ps1 后台拉起
curl -s -X POST http://127.0.0.1:8310/search -H "Content-Type: application/json" \
     --data-binary '{"query":"日志轮转应该在写入前还是写入后做","k":5}'
```

`index.py` 会打一行 `WARN: ...vec.npy 缺失`——正常：它只建 BM25，向量档由 `build_vec.py` 单独产出，
命名是 `vec_m3.npy`（`kb_search.py` 按这个优先级找档，历史上它默认指向 `vec.npy`，主库跑一次就把
一份陈旧的小模型向量当成向量档用了，分数看着正常其实全错）。

自带示例语料的块数与 `kb/baseline.json` 里记录的一致（34 块），所以上面四步会一路跑通并 PASS。
换自己的语料时 `chunker.py` 会因为块数不符而**拒绝覆写**——这是设计，不是 bug：块序一变，同目录
向量档的行号全体错位，服务会静默换到一份错位索引上。先建自己的基线（`corpus/README.md` 有步骤）。

## 目录

| 路径 | 内容 |
|---|---|
| `kb/chunker.py` `index.py` `build_vec.py` | 入库三件套：切块 → BM25 → dense |
| `kb/kb_search.py` `fuse.py` `m3embed.py` | 检索、多区融合、ONNX 嵌入器 |
| `kb/kb_serve.py` | 常驻服务、热重载、看板、查询日志与轮转 |
| `kb/kb_atomic.py` `kb_log_archive.py` `write_provenance.py` | 原子换档 / 日志归档 / 溯源落盘 |
| `kb/kb_remember.py` `kb_epilogue.py` `mem_audit.py` | 记忆区闭环：写回、复盘蒸馏、重复与冲突裁决 |
| `kb/coverage_audit.py` `kb_health_probe.py` `kb_soak_test.py` `replay.py` | 覆盖审计、健康探测、并发压测、真实查询回放 |
| `kb/eval.py` `baseline*.json` | 评测与机器闸门 |
| `kb/test_*.py` | 守门员单测：作者机器上 48 项全跑；全新 clone 里没有对端 MCP 引擎、也没有活服务，其中 3 项会显式跳过（46 跑 / 3 skip），全在临时目录里演练，不碰任何生产件 |
| `corpus/` | 语料入口（仓库自带自撰示例） |
| `tools/` | 发布打包与净化闸门 |

## 这个仓库里没有的东西

作者的真实语料**没有发布**，而且是打包脚本白名单挡住的，不是手工删的：

- `corpus/` 之外的文档正文，以及 `kb/chunks.jsonl`（正文的二次拷贝）、`bm_vocab.json`
  （从语料学出来的词表——实测连本机用户名都会成为词条）。
- **记忆文档**：`kb/mem/corpus/`（智能体写回的会话结论）与 `kb/epi/corpus/`（复盘条目）整目录。
  它们的代码在仓库里，数据不在。
- `kb/query_log.jsonl`、`kb/reload_history.jsonl`、`eval_per_query*.json`：含真实查询文本。
- 模型权重（`*.onnx`，1.1 GB）、`venv/`、对比实验目录与快照。
- `tools/release_local.json`：作者本机路径与第三方名称的替换表，本身就是最该被闸门拦住的东西。

想复现私有语料上的 81.8% / 96.4%？做不到，也不需要做到——那份金标准指向的是不该公开的文档。
仓库里的 `kb/baseline.json` 冻的是**示例语料**的闸门，clone 完就能跑通并看到 PASS。

## 五条设计规矩

1. **落盘只走原子换档。** 先写 `<dst>.new.<pid>`，构建成功才 `os.replace`，旧档带时间戳进 `.prev`
   留 3 份；溯源 sidecar 跟向量成对走；切块得到 0 行直接拒绝覆写。新增文件请用 `kb_atomic`，
   别写裸 `open(..., "w")` / `np.save`。
2. **服务只读、绝不自动重建。** 磁盘语料比索引新时，`/health` 报 `lag` 并在看板亮黄灯，
   但不会自己补建——重建闸门是人走的，把写权限交给 watcher 等于绕过第 1 条的基线守卫。
3. **质量改动必须过闸。** 融合权重、切块口径、词典任一变化，先跑 `eval.py`，
   再跑一次真实查询回放（`replay.py --diff`）确认命中没有大面积换位。
4. **占盘有硬上限。** 查询日志 10 MB × (3+1) = 40 MB，重载历史 2 MB × 3，
   服务日志在每次点火前带时间戳归档。轮转发生在写入**之前**。
5. **不做开机自启与看门狗。** 进程死了不会自己回来，监控只报警。这是权衡，不是遗漏：
   静默重启会把"什么时候换的档"变成不可追溯的问题。

## 运维脚本的两个 Windows 坑

- `_start.ps1` 必须是 **UTF-8 with BOM**。PowerShell 5.1 无 BOM 时按 GBK 解析，中文注释会让整个脚本
  语法报错、根本跑不起来（报错形似"字符串缺少终止符"）。
- 别用 `capture_output` 之类"等管道关闭"的方式调它——后台服务继承了那个管道，调用方会一直挂着。
  脚本本身没错，是调用姿势问题。

## 生产核验

`kb/kb_health_probe.py` 把判据一次跑完并给退出码（0 = 无 FAIL）：监听地址、内存块数 vs 磁盘、
入库文件新鲜度、语料比索引新、五区溯源、进程私有内存与线程、**代码是否比进程新**、
四档冒烟与延迟、日志轮转体检、覆盖审计。`--heavy-provenance` 会抽样重嵌比对余弦——
行数相等只证明没错位，不证明没拿错模型。

已跑过的实跑数据（私有部署上）：4469 请求 0 失败，稳态 P50/P95/P99 = 155/607/752 ms，
8 路突发 688/1221/1468 ms，21 分钟私有内存 +26.5 MB、线程 59→59。
这只证明"这段时间没涨"，不证明"没有泄漏"。

## License

MIT
