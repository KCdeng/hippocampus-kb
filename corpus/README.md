# corpus/ —— 放什么、不放什么

这个目录是引擎的**语料入口**，也是发布件里唯一带正文的东西。

仓库自带的是 7 篇讲这个引擎自己的短文档（加上本 README 共 8 个文件，切成 34 块），用来让评测链路
开箱能跑：`chunker → build_vec → index → eval` 四步不需要语料以外的任何数据，`kb/baseline.json`
里冻的闸门就是这套示例语料上的实测数字（2026-11-04 实跑：hybrid-v2 hit@5 23/24，文件级@5 24/24）。

**它们不是私有语料的节选，也不是任何第三方的文本。** 生产部署时请把这里的文件换成你自己的：
把项目笔记、手册、结论导出成 Markdown 放进这个目录即可，切块器只认 `.md` 与 `.txt`。

## 目录约定

| 路径 | 内容 | 是否进版本库 |
|---|---|---|
| `corpus/*.md` | 你的文档原文 | 由你决定（这是唯一含正文的位置） |
| `kb/chunks.jsonl` | 切块产物 | 不进（`.gitignore` 已排除，正文的二次拷贝） |
| `kb/bm25.npy`、`kb/bm_vocab.json`、`kb/vec*.npy` | 派生索引 | 不进（可确定性重建；词表从语料学出来，会带上语料里的字符串） |
| `kb/mem/corpus/`、`kb/epi/corpus/` | 智能体写回的记忆与复盘条目 | 不进（属于会话产物，不是知识文档） |
| `kb/query_log.jsonl`、`kb/reload_history.jsonl` | 运行日志 | 不进（含真实查询文本） |
| `models/bge-m3-onnx/` | ONNX 权重 | 不进（体积大，且应各自从上游拉取） |

## 换成语料后要做的三件事

1. 跑一次 `chunker.py`，它会因为块数与 `kb/baseline.json` 不符而拒绝覆写——这是设计。
   换自己的档：`chunker.py --force --out 实验目录/chunks.jsonl` 先看块数，再把块数写进你自己的
   `kb/baseline.json`（正式换代前先按下一、三步把闸门建起来，别用 `--force` 直写生产件）。
2. 建 BM25（`index.py`）与向量档（`build_vec.py`，**必须显式跑**：`index.py` 不再悄悄重嵌）。
3. 出一批金标准题（照 `kb/eval_set.json` 的形态写），跑 `eval.py` 定档，把实测数字冻进 `baseline.json`。

没有第 3 步，检索质量就只是"看起来能用"。闸门的作用不是证明你好，是让"改坏了"能被机器发现。
