# 智能体"海马体"（长期记忆检索）候选项目调研 — 2026-11-01

## 现状基线（已验证）
- 知识库：`<本地工作区>` 共 512 个 .md，约 3.4 MB（中文 ≈ 200 万~250 万字符 ≈ 150 万~200 万 token）。
- 现有检索方式：预读全部文档→压缩成 Markdown 索引→智能体 grep/读取。
- 痛点：语义召回靠关键词命中；跨文档关联题需多轮翻找；索引是"有损压缩"。

## 已下载源码（src/ 下，均为主分支 zip）
| 项目 | Stars | 定位 | 备注 |
|---|---|---|---|
| LightRAG-main (HKUDS) | 40,458 | 图+向量双层检索框架，EMNLP'25 | 索引期需 LLM 抽实体/关系；自带 WebUI、MCP、多后端 |
| txtai-master (neuml) | 13,054 | 一体化 embedding 数据库（向量+BM25 混合+图） | 轻、可完全本地、易包成 CLI/MCP |
| lancedb-main | 11,823 | 嵌入式（in-process）向量库，Rust 内核 | 无服务器、mmap 磁盘常驻、Python/JS/API |
| mcp-memory-service-main (doobidoo) | 2,056 | 现成 MCP 记忆服务器（语义+知识图谱） | 最省事的"给智能体接记忆"路径 |
| HippoRAG-main (OSU-NLP) | 4,087 | NeurIPS'24 海马体索引理论启发的记忆框架 | 检索期用 LLM 提问 + PPR 图传播；研究级 |
| MemoRAG-main (qhjqhj00) | 2,270 | 双系统：专用长记忆模型(7B 微调)先"回忆线索"再检索 | 需 Memorag-Qwen2-7B 权重；维护弱，原型性质 |
| fastembed-main (qdrant) | 3,277 | ONNX CPU embedding 推理 | 不占 2080Ti，bge/gte 中文模型可选 |

VectorSpaceLab/MemoRAG 只是镜像（仅 README），已用 qhjqhj00/MemoRAG 原件替换。

## 路线判断
1. **向量库常驻内存 = 可行、最实用**（成熟度高）。本库切块后约 1~2 万 chunks，1024 维 fp16 全量 ≈ 40~80 MB，可整表常驻内存，暴力余弦 <10 ms。embedding 一次算完落盘（LanceDB/txtai），按文件 hash 增量更新。
2. **KV cache 作为存储介质 = 不推荐**。KV 绑定具体模型/量化/位置编码，chunk 独立预计算的 KV 检索后会跨块失真；学界方向是 CacheBlend/LMCache（复用非前缀 KV 加速"取回后的长上下文读取"，EuroSys'25 最佳论文），是**速度优化**不是**索引**。vLLM 的 prefix caching 已自动覆盖稳定前缀。
3. **图检索（LightRAG/HippoRAG）= 解决关联题的第二层**。代价是索引期大量 LLM 调用（可用 8090/NAS 引擎当抽取模型），质量提升针对"多跳/汇总"类问题。
4. BM25/中文分词混合不可丢：术语号（地址表、状态机编号、PLC 指令）是精确匹配强项，向量反而弱。

## 建议分阶段（含验收门槛）
- **G0 评测集先行**：手写 50~100 条"问题→应命中段落"金标准；对比 grep 基线 vs 各方案 hit@5 / MRR。没有评测集就不改生产检索（受控 A/B 原则）。
- **G1 最小海马体**（≤1 天）：LanceDB（或 txtai）+ bge-m3 embedding（NAS vLLM `/v1/embeddings` 或 fastembed CPU）+ jieba BM25 混合 + bge-reranker-v2-m3 精排，包成 `kb_search` CLI/MCP 工具。全量索引预计几分钟（GPU）~20 分钟（CPU）。
- **G2 图增强**（视 G1 结果）：LightRAG 挂 8090 引擎做抽取，仅对高频关联域（PLC/HMI 手册）开图索引。
- **G3 研究对照**：读 HippoRAG 2 源码，其 PPR（个性化 PageRank）可只在检索期用，借鉴思路而非整体部署。
- **G4（可选）**：LMCache/CacheBlend 思路用于把"检索到的 3~5 个长段落"低成本塞进 262K 上下文，属推理侧加速。

## 风险 / 信息缺口
- bge-m3 在其 vLLM 引擎上的 embedding 服务化尚未实测（置信度：中高，vLLM pooling 路线成熟）。
- 中文 chunk 切分策略（按标题层级 vs 固定长度）对 hit@k 影响大，需在 G0 评测集上 A/B。
- LightRAG 图索引 token 成本需先在小目录（如 injection-molding-plc-hmi 的 36 篇）试算再放大。
