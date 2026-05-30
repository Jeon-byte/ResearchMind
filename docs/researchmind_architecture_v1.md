# ResearchMind V1 架构方案

## 1. 结论

当前目标下，`ResearchMind` 需要后端。

不是因为“有网页就一定要后端”，而是因为 MVP 的核心能力天然依赖服务端状态和任务编排：

- 搜索论文并聚合多源结果
- 将论文加入 `collection`
- 下载 PDF 并记录状态
- 解析 PDF 并切分全文
- 建立检索索引
- 对指定 `collection` 做问答
- 返回带 citation 的回答和证据

如果没有后端，这些能力会变成：

- 前端直接调用多个论文源 API，难以统一去重和缓存
- PDF 下载与解析无稳定执行位置
- 索引和问答状态难以持久化
- 无法可靠管理 `collection`

因此推荐采用：

- 前端：Web UI
- 后端：Python API 服务
- 主数据库：SQLite
- 向量索引：FAISS

## 2. 系统边界

`ResearchMind` 第一版是一个“论文发现 -> 论文入库 -> 全文问答”的单用户 / 小规模系统。

第一版不做：

- 多租户
- 复杂权限
- 分布式任务系统
- 云原生向量数据库依赖

第一版要做：

- 搜索工作台
- 文献集合管理
- PDF 入库流水线
- 基于全文的增强检索问答

## 3. 推荐总体架构

```text
researchmind/
  apps/
    api/           # FastAPI 后端
    web/           # React/Next.js 前端
  packages/
    core/          # 复用 paper-tracker 核心搜索能力
    ingestion/     # PDF 下载、解析、chunk、embedding
    retrieval/     # 混合检索、rerank、citation 组装
  data/
    app.db         # SQLite
    faiss/         # FAISS 索引文件
    papers/        # PDF 文件存储
```

如果第一阶段不想做 monorepo 重构，也可以更保守：

- 继续保留 `paper-tracker`
- 在仓库内新增 `src/ResearchMindApi`
- 新增 `web/`
- 逐步把可复用逻辑从 `PaperTracker` 抽到共享 service

## 4. 为什么需要后端

## 4.1 搜索需要统一编排

当前多源搜索、聚合、去重能力已经在服务层里，适合由后端统一暴露 API，而不是让前端直接拼多源请求。

## 4.2 PDF 入库是后台任务

下载 PDF、解析全文、生成 embedding 都不是前端该做的事。

这些任务需要：

- 文件系统访问
- 重试和错误状态
- 任务进度更新
- 长时间执行

## 4.3 问答必须依赖服务端检索

问答前需要在某个 `collection` 下做：

- chunk 过滤
- 文本召回
- 向量召回
- rerank
- citation 拼装

这些都应在服务端完成。

## 5. 前端方案

前端建议保留现有风格，但交互形态改为工作台。

推荐：

- `React + Vite`

原因：

- 比 Next.js 更轻
- MVP 上手快
- 当前并不需要 SSR
- 本项目的重点在交互和任务状态，不在 SEO

当然如果你更想做长期产品，也可以直接用 `Next.js`。但单纯为 MVP，我更倾向 `React + Vite`。

### 5.1 页面结构

#### 页面一：Search

用途：搜索论文并加入 `collection`

模块：

- 顶部搜索栏
- 来源筛选
- 时间范围筛选
- 结果列表
- 批量勾选
- “加入 collection” 面板

#### 页面二：Collection Detail

用途：查看某个知识库的入库状态

模块：

- collection 基本信息
- 论文列表
- 每篇论文的 PDF / parse / index 状态
- 手动重试按钮
- 打开 PDF

#### 页面三：Ask

用途：对某个 `collection` 提问

模块：

- 左侧 collection 列表
- 中间对话区
- 右侧 citation / evidence 面板

### 5.2 前端状态

前端只保存 UI 状态，不承担核心业务状态。

前端本地状态包括：

- 当前搜索条件
- 当前结果勾选项
- 当前选中的 collection
- 当前会话消息

核心业务状态必须来自后端：

- collection 内容
- 论文入库状态
- chunk 索引状态
- 问答结果和 citation

## 6. 后端方案

后端推荐：

- `FastAPI`

原因：

- 和现有 Python 代码天然契合
- 适合 API + 后台任务模式
- 非常容易复用 `paper-tracker` 逻辑

### 6.1 后端模块

建议拆成：

- `api/routes/search.py`
- `api/routes/collections.py`
- `api/routes/ingestion.py`
- `api/routes/qa.py`
- `services/search_service.py`
- `services/collection_service.py`
- `services/ingestion_service.py`
- `services/qa_service.py`
- `repositories/*.py`

### 6.2 后端职责

#### Search Service

- 调用现有多源搜索
- 标准化返回结构
- 处理搜索缓存

#### Collection Service

- 创建 collection
- 添加或移除论文
- 查询 collection 详情

#### Ingestion Service

- 下载 PDF
- 提取全文
- 切 chunk
- 调用 embedding
- 更新索引

#### QA Service

- 读取 collection 范围
- 检索相关 chunks
- rerank
- 拼装 prompt
- 生成回答
- 记录 citation

## 7. 数据库方案

第一版推荐：

- 主数据库：SQLite

原因：

- 当前项目已经有 SQLite 基础
- 开发和迁移成本低
- 单用户 / 小规模完全够用

### 7.1 为什么普通数据库是必须的

普通数据库负责系统真相来源：

- 哪些 collection 存在
- 哪些论文属于哪个 collection
- PDF 是否已下载
- 是否解析成功
- chunks 是什么
- 哪次问答引用了哪些 chunk

这些都不是向量库应该承担的事情。

### 7.2 建议表结构

#### collections

- `id`
- `name`
- `description`
- `created_at`
- `updated_at`

#### collection_papers

- `id`
- `collection_id`
- `source`
- `source_id`
- `added_at`
- `status`

说明：

- `status` 表示该论文在 collection 内的整体入库状态

#### paper_assets

- `id`
- `source`
- `source_id`
- `pdf_url`
- `local_path`
- `download_status`
- `download_error`
- `parse_status`
- `parse_error`
- `index_status`
- `index_error`
- `downloaded_at`
- `parsed_at`
- `indexed_at`

#### paper_chunks

- `id`
- `source`
- `source_id`
- `chunk_index`
- `section_title`
- `page_start`
- `page_end`
- `content`
- `token_count`
- `created_at`

#### conversations

- `id`
- `collection_id`
- `title`
- `created_at`
- `updated_at`

#### messages

- `id`
- `conversation_id`
- `role`
- `content`
- `created_at`

#### answer_citations

- `id`
- `message_id`
- `paper_chunk_id`
- `rank_order`
- `score`
- `quote_text`

### 7.3 是否需要单独 papers 表

短期不一定要新建。

因为当前系统已经通过 `seen_papers` 和 `paper_content` 保存论文主信息。第一阶段可以在此基础上扩展，避免重复建模。

## 8. 向量检索方案

需要向量检索能力，但第一版不需要独立向量数据库产品。

推荐：

- 向量索引：`FAISS`

### 8.1 为什么不是一开始就用向量数据库

因为第一版的数据规模和目标复杂度都不高：

- collection 数量有限
- 每个 collection 论文数有限
- chunk 总量有限
- 优先验证产品闭环

这时 `FAISS + SQLite` 足够。

### 8.2 向量索引如何组织

建议按 collection 组织索引，至少逻辑上这样分层：

- `collection_id`
- `chunk_id`
- `embedding vector`

可以有两种实现：

#### 实现 A：全局一个 FAISS 索引

优点：

- 实现简单

缺点：

- 查询时需要额外做 `collection_id` 过滤映射

#### 实现 B：每个 collection 一个 FAISS 索引

优点：

- collection 作用域天然隔离
- 查询简单

缺点：

- 管理多个索引文件

第一版我更建议 `实现 B`，因为更贴合产品语义。

## 9. 增强检索设计

增强检索不建议做成“纯向量检索”，而建议做混合召回。

### 9.1 推荐链路

1. 用户问题进入系统
2. 限定 `collection_id`
3. 做文本召回
4. 做向量召回
5. 合并候选 chunk
6. rerank
7. 取 top-k 作为上下文
8. 生成答案

### 9.2 文本召回

第一版可以很务实：

- 直接在 `paper_chunks.content` 上做 SQLite FTS5

好处：

- 简单
- 对术语检索效果好
- 适合论文里的专有名词

### 9.3 向量召回

对每个 chunk 生成 embedding，用 FAISS 做 ANN 检索。

### 9.4 rerank

第一版可以先做轻量 rerank：

- 规则融合分数

例如：

- `final_score = 0.4 * keyword_score + 0.6 * vector_score`

后续如果效果不够，再接 cross-encoder 或 LLM reranker。

## 10. PDF 处理方案

### 10.1 下载

来源优先级：

1. 直接 PDF 链接
2. abstract 页面中的 PDF 链接
3. 可解析落地页中的 PDF 按钮

下载后保存到：

- `data/papers/{source}/{source_id}.pdf`

### 10.2 解析

推荐优先尝试：

- `PyMuPDF`

原因：

- 速度快
- 工程实践成熟

### 10.3 chunk 策略

第一版不要太复杂。

建议：

- 先按页提取文本
- 再按段落或固定 token 窗口切分
- 每 chunk 保留页码和 section 信息

推荐参数起点：

- chunk size: `800-1200` tokens
- overlap: `100-150` tokens

### 10.4 section 信息

如果能从 PDF 文本中识别标题层级，就记录到 `section_title`。
识别不到也没关系，MVP 不必强求完美结构化。

## 11. 问答方案

### 11.1 问答输入

用户必须指定：

- `collection_id`
- `question`

### 11.2 检索上下文

构造上下文时每条证据保留：

- 论文标题
- source / source_id
- page range
- section title
- chunk text

### 11.3 模型提示要求

系统 prompt 必须约束：

- 只能根据提供证据回答
- 若证据不足，明确说不知道
- 引用时按编号标注
- 不要伪造方法细节

### 11.4 返回结构

后端返回：

- `answer`
- `citations[]`
- `evidence_chunks[]`

其中 `citations[]` 包含：

- `paper_title`
- `chunk_id`
- `page_start`
- `page_end`
- `quote_text`

## 12. API 设计

### 12.1 搜索

`POST /api/search`

请求：

- `query`
- `sources`
- `max_results`
- `date_range`

返回：

- `papers[]`

### 12.2 Collection

`GET /api/collections`
`POST /api/collections`
`GET /api/collections/{id}`

### 12.3 添加论文

`POST /api/collections/{id}/papers`

请求：

- `papers[]`

动作：

- 写入 `collection_papers`
- 为每篇论文创建或复用 `paper_assets`
- 创建入库任务

### 12.4 入库状态

`GET /api/collections/{id}/papers`

返回每篇论文的：

- 元数据
- PDF 状态
- parse 状态
- index 状态

### 12.5 问答

`POST /api/ask`

请求：

- `collection_id`
- `question`
- `conversation_id` 可选

返回：

- `answer`
- `citations`
- `conversation_id`
- `message_id`

## 13. 后台任务方案

第一版不必上 Celery。

推荐：

- FastAPI 应用内后台任务
- 或简单任务队列线程池

原因：

- MVP 简单
- 部署轻
- 易调试

但要注意：

- 任务状态必须落库
- 任务失败要可重试
- API 不能阻塞等待完整入库流程

## 14. 建议的实现顺序

### Step 1

先做后端骨架：

- FastAPI 启动
- search API
- collections API

### Step 2

做前端最小搜索页：

- 搜索
- 展示结果
- 新建 collection
- 加入 collection

### Step 3

做入库流水线：

- 下载 PDF
- 保存状态
- 提取文本
- 写 chunk

### Step 4

做向量索引：

- chunk embedding
- FAISS 索引
- collection 范围查询

### Step 5

做问答页：

- 发问
- 检索
- 生成回答
- 显示 citation

## 15. 当前最推荐的 MVP 技术组合

前端：

- `React + Vite`

后端：

- `FastAPI`

主数据库：

- `SQLite`

全文解析：

- `PyMuPDF`

向量索引：

- `FAISS`

检索策略：

- `SQLite FTS5 + FAISS 混合召回`

问答模型：

- 继续沿用当前 OpenAI-compatible LLM 接口

## 16. 最终判断

当前这个项目做成 `ResearchMind`，最合理的第一版不是“纯前端网页”，也不是“重型 RAG 平台”，而是：

- 以现有 `paper-tracker` 为搜索内核
- 增加 Python 后端 API
- 增加延续现有风格的前端工作台
- 用 SQLite 管业务数据
- 用 FAISS 做 collection 范围内全文增强检索

这套方案复杂度可控，也足以支撑一个完整可用的科研助手 MVP。
