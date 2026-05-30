# ResearchMind

ResearchMind 是一个面向科研阅读的本地论文知识库工作台。它把论文搜索、结果筛选、PDF 入库、RAG 检索和带引用问答串成一条完整流程，目标是把论文从“查到”推进到“可检索、可问答、可追溯”。

## 演示

![ResearchMind Brief](show/image.png)

![ResearchMind Knowledge Base](show/show2.png)

## 产品定位

ResearchMind 由两个核心工作区组成：

- **Brief**：围绕一个主题发起论文搜索，保存候选论文和 rerun 结果，用于探索和筛选。
- **Knowledge Base**：将筛选后的论文沉淀为知识库，后台完成 PDF 下载、解析、分块、索引，并支持基于证据的问答。

典型流程：

1. 搜索论文并生成 Brief。
2. 从 Brief 中选择论文加入 Knowledge Base。
3. 系统下载 PDF、解析全文、提取图表并建立索引。
4. 用户在知识库内提问，系统返回答案和具体引用来源。

## 数据拉取与去重

论文搜索会拉取标题、作者、摘要、PDF 链接、DOI、发布时间等结构化元数据。当前主要面向 arXiv 场景，代码中保留 OpenAlex、PubMed 等来源的扩展能力。

数据侧重点：

- 支持分页拉取、时间窗口、回填和 rerun。
- 基于 source id、DOI、标题等信息去重。
- 搜索结果先进入 Brief，用户确认后再加入 Knowledge Base，减少无关论文进入长期知识库。

## RAG 能力

### PDF 解析与多模态分块

论文加入 Knowledge Base 后会进入 ingestion 流程：

- 使用 PyMuPDF 提取 PDF 逐页文本。
- 按章节和句子边界进行 section-aware chunking。
- 提取 PDF 图片，并结合页面 caption 生成 figure/table chunk。
- 可选使用 Qwen3-VL-Instruct 对图表做离线视觉理解，生成检索导向 caption。
- 为每篇论文生成 `Paper Summary` chunk，概括贡献、方法模块、实验设置和主要结论。

每个 chunk 会保留 source、source_id、页码、section、modality、image_path 等元信息，方便回答时追溯来源。

### 向量检索与索引

系统会为知识库建立多种索引：

- 文本 chunk：使用 embedding 模型生成向量表示。
- 图表 chunk：可使用 Qwen3-VL-Embedding 建立 image-vector 索引。
- 全文检索：使用 SQLite FTS5 支持关键词与 BM25 风格召回。
- 向量存储：优先使用 FAISS，不可用时退化为 numpy 索引。

### 多路融合与 Rerank

问答时会组合多条召回路径：

- dense vector retrieval
- SQLite FTS retrieval
- image-vector retrieval
- query-aware prior
- optional reranker

融合后的 top-k 证据进入回答生成。回答必须基于召回证据，并使用 `[1]`、`[2]` 形式引用具体来源。

### 引用溯源

每条回答引用会绑定：

- 论文标题
- 页码范围
- section title
- chunk 摘录
- 检索分数
- 图表图片链接（如果来自 figure/table chunk）

这使得用户可以判断回答依据是否充分，而不是只得到一段不可追溯的自然语言。

## 图像语义能力

当前系统已经接入 VL 模型，但方式是：

- 入库阶段：Qwen3-VL-Instruct 离线理解图表并生成 caption。
- 检索阶段：Qwen3-VL-Embedding 支持用文字问题召回相关图表。
- 回答阶段：基于召回到的图表 caption、图片引用和正文证据生成答案。

因此当前支持图像语义检索和图表证据引用；但还不是“提问时实时把原图送入 VLM 做视觉问答”的完整 VQA。后续 Agentic RAG 会加入关键图表的 VLM 二次验证。

## 检索模式

- **standard**：默认模式，执行向量检索、FTS、图像检索、多路融合和 rerank。
- **decompose**：轻量拆问模式，将复杂问题拆成多个子问题后检索并融合。
- **agent**：待做能力，计划基于 LangGraph 实现 Agentic RAG。

## Agentic RAG 计划

Agentic RAG 模式待做，计划 RAG Pipeline 编排和带循环反馈的 Agent 检索：

- 使用 LangGraph 构建状态机：`plan_query -> decide_action -> run_tool -> check_termination -> answer_or_abort`。
- 引入 AgentState，记录问题、目标、假设、开放子问题、证据和工具历史。
- 用 Query Planner 识别问题类型，并生成 normalized question / sub-queries。
- 用 Tool Registry 封装检索工具，如 text、figure、text+figure 的 `search_evidence`。
- 用 Evidence Store 聚合多轮证据，按 chunk id 去重并记录来源 query。
- 支持证据不足时自动改写 query 并再次检索。
- 支持 citation recovery，必要时扩展相邻 chunk 补足引用。
- 对关键图表触发 VLM visual verification。
- 最终输出 answer audit，标注证据覆盖度、不足点和低置信度提示。

## 技术栈

- Python / FastAPI
- SQLite / SQLite FTS5
- PyMuPDF
- FAISS 或 numpy fallback
- BAAI/bge-m3
- BAAI/bge-reranker-base
- Qwen3-VL-Instruct
- Qwen3-VL-Embedding
- OpenAI-compatible LLM API

## 启动

准备 `.env`：

```bash
LLM_API_KEY=your_api_key
```

启动服务：

```bash
cd research_mind
paper-tracker serve --config config/qwen.yml
```

打开：

```text
http://127.0.0.1:8000
```

## 当前边界

ResearchMind 仍是本地科研工作台原型。检索质量受 PDF 解析、embedding/reranker、图表 caption 和知识库证据覆盖度影响。`Paper Summary` chunk 能改善全文级问题，但回答仍需要原文 chunk 支撑引用与细节。

## License

MIT
