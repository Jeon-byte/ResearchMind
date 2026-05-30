# ResearchMind MVP 技术方案

## 1. 目标

将当前 `paper-tracker` 演进为一个面向科研场景的论文助手 `ResearchMind`，完成以下最小可用闭环：

1. 搜索论文（已有基础）
2. 勾选论文加入 `collection`
3. 下载 PDF
4. 解析全文
5. 对 `collection` 提问
6. 回答附 `citation`

MVP 的核心不是做成通用知识库，而是打通“发现论文 -> 入库 -> 基于全文问答”的科研工作流。

## 2. MVP 范围

### 2.1 必做能力

- 复用当前多源论文搜索能力，支持 `arXiv`、`OpenAlex`、`PubMed`
- 在网页中展示搜索结果，支持勾选和批量加入 `collection`
- 为已加入 `collection` 的论文下载 PDF
- 对 PDF 进行全文解析和分块
- 对指定 `collection` 建立全文检索索引
- 用户对 `collection` 提问，系统基于检索到的全文片段回答
- 回答必须附带引用来源，至少包含论文标题和证据片段

### 2.2 暂不纳入 MVP

- 团队协作和多租户
- 自动周报和订阅推送
- 上传任意本地文档
- 复杂 Agent 工作流
- 跨 collection 的全局智能分析
- 高级权限管理

## 3. 产品形态

MVP 需要有明确的前后端交互，不建议继续停留在 CLI + 静态 HTML 输出形态。

原因：

- 用户需要在搜索结果中勾选论文并加入 `collection`
- 用户需要查看下载、解析、索引状态
- 用户需要在指定 `collection` 上发起问答
- 用户需要看到回答对应的引用证据

因此，MVP 需要一个最小但完整的 Web 产品：

- 前端：提供搜索、收藏、入库状态、问答界面
- 后端：提供搜索 API、collection API、PDF 入库任务、RAG 问答 API

前端视觉建议沿用当前 HTML 模板风格，包括：

- 现有米色系浅色主题和深色主题切换
- `Crimson Pro` + `Atkinson Hyperlegible` 的排版风格
- 卡片式论文展示
- 左侧导航 / 右侧内容工作台布局

现有设计资源主要来自：

- `template/html/scholar/document.html`
- `template/html/scholar/assets/style.css`

## 4. 系统架构

建议拆成四层：

### 4.1 搜索内核层

直接复用当前 `paper-tracker` 已有能力：

- 多源搜索
- 聚合排序
- 跨源去重
- SQLite 持久化
- LLM 摘要增强

这一层继续作为 `ResearchMind` 的论文发现引擎。

### 4.2 API 服务层

新增 Web API，对外提供交互能力：

- 搜索论文
- 创建和管理 `collection`
- 将论文加入 `collection`
- 触发 PDF 下载 / 解析 / 建索引
- 提交问答请求

推荐使用 `FastAPI`。

### 4.3 知识库处理层

新增全文入库流水线：

1. 选择论文
2. 下载 PDF
3. 提取正文
4. 清洗文本
5. 切分 chunk
6. 生成 embedding
7. 建立检索索引

### 4.4 Web 前端层

提供最小工作台式交互：

- 搜索页
- Collection 页
- 问答页

## 5. 核心数据对象

### 5.1 collection

表示一个研究主题下的论文集合。

示例：

- `LLM for Science`
- `Medical VLM`
- `Agentic Coding`

### 5.2 paper

已有论文元数据对象，继续沿用当前 `Paper` 模型。

### 5.3 paper_asset

表示论文对应的 PDF 资产和处理状态。

关键字段建议：

- `paper_id`
- `pdf_url`
- `local_path`
- `download_status`
- `parse_status`
- `index_status`
- `error_message`

### 5.4 paper_chunk

表示 PDF 解析后的文本分块。

关键字段建议：

- `paper_id`
- `chunk_index`
- `section_title`
- `page_start`
- `page_end`
- `content`

### 5.5 conversation / message

表示围绕某个 `collection` 的问答会话。

## 6. 建议的数据表扩展

在现有 SQLite 基础上增加：

- `collections`
- `collection_papers`
- `paper_assets`
- `paper_chunks`
- `conversations`
- `messages`
- `answer_citations`

如需向量检索，MVP 可采用“两层存储”：

- 主数据：SQLite
- 向量索引：`FAISS` 或 `Chroma`

如果后续规模扩大，再考虑迁移到 `Postgres + pgvector`。

## 7. 入库流水线设计

### 7.1 用户动作

1. 用户搜索论文
2. 在结果页勾选论文
3. 加入某个 `collection`
4. 系统异步执行入库任务

### 7.2 后台任务状态

建议统一状态机：

- `queued`
- `downloading`
- `downloaded`
- `parsing`
- `parsed`
- `indexing`
- `indexed`
- `failed`

### 7.3 异常现实

需要接受以下现实情况：

- 不是所有论文都有可访问 PDF
- 不是所有 PDF 都能稳定解析
- 某些来源只有落地页，没有全文

因此，MVP 中必须把“失败状态”和“重试入口”设计出来。

## 8. RAG 问答设计

MVP 不做“只基于摘要的轻量问答”，而是直接面向全文知识库问答。

问答范围必须明确限定为：

`仅对已加入 collection 且完成全文索引的论文回答`

### 8.1 问答链路

1. 用户选择一个 `collection`
2. 输入问题
3. 系统在该 `collection` 的全文 chunk 中检索
4. 选出 top-k 证据片段
5. 组装上下文交给 LLM
6. 返回答案和引用

### 8.2 citation 要求

回答至少附带：

- 论文标题
- 片段内容
- 页码或 section
- 原论文链接或本地 PDF 标识

### 8.3 回答边界

系统提示中应明确：

- 只能依据已索引论文作答
- 若证据不足，应明确说不知道
- 不应伪造论文细节

## 9. 前后端交互设计

### 9.1 是否需要前后端交互

需要，而且是 MVP 的必要条件。

原因不是“为了做网页”，而是因为核心链路本身就是交互式的：

- 搜索后用户要筛选论文
- 用户要选择加入哪个 `collection`
- 用户要观察入库进度
- 用户要围绕某个 `collection` 提问
- 用户要查看回答对应的证据

### 9.2 前端页面建议

#### 页面一：搜索页

功能：

- 输入关键词
- 选择来源与时间范围
- 查看结果卡片
- 勾选论文
- 批量加入 `collection`

#### 页面二：Collection 页

功能：

- 查看某个 `collection` 的论文列表
- 查看每篇论文的 PDF 下载 / 解析 / 索引状态
- 手动重试失败任务

#### 页面三：问答页

功能：

- 选择 `collection`
- 输入问题
- 查看回答
- 查看 citation 和证据片段

## 10. API 草案

建议的最小接口：

- `POST /api/search`
- `GET /api/collections`
- `POST /api/collections`
- `POST /api/collections/{id}/papers`
- `GET /api/collections/{id}/papers`
- `POST /api/collections/{id}/ingest`
- `GET /api/collections/{id}/jobs`
- `POST /api/ask`

## 11. 技术选型建议

### 11.1 后端

- `FastAPI`
- 复用当前 Python 项目结构
- 后台任务初期可先用应用内任务队列

### 11.2 前端

- `Next.js` 或 `React + Vite`
- 延续现有页面风格，不推翻视觉语言

### 11.3 文本解析

PDF 解析建议优先评估：

- `PyMuPDF`
- `pdfplumber`
- 必要时加入 OCR 兜底，但不放入 MVP 必做项

### 11.4 向量检索

MVP 建议：

- embedding 模型保持 OpenAI-compatible
- 向量索引先采用 `FAISS`

## 12. 实施顺序

### Phase 1：Web 化搜索

- 将当前搜索能力封装为 API
- 实现搜索页
- 支持勾选并加入 `collection`

### Phase 2：全文入库

- 创建 `collection` 数据结构
- 下载 PDF
- 解析正文
- 建立 chunk 和索引

### Phase 3：问答闭环

- 实现 `collection` 范围内检索
- 接入 LLM 回答
- 返回 citation

## 13. MVP 完成标准

满足以下条件即可视为闭环打通：

1. 用户能在网页里搜索论文
2. 用户能勾选论文加入 `collection`
3. 系统能下载并解析 PDF
4. 系统能对该 `collection` 建立全文检索索引
5. 用户能对 `collection` 提问
6. 系统回答附引用证据

## 14. 当前结论

当前阶段应当明确：

- 需要前后端用户交互
- 前端界面继续沿用现有风格
- RAG 直接围绕 PDF 全文知识库建设
- 先完成 MVP 闭环，再扩展更复杂能力
