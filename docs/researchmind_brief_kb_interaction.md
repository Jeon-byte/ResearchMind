# ResearchMind Brief / Knowledge Base 交互方案

## 1. 核心对象

当前版本的 `ResearchMind` 以两个核心对象组织用户工作流：

- `Brief`
- `Knowledge Base`

不再把“搜索页”“知识库页”“问答页”当成彼此并列的产品主对象，而是让它们服务于这两个核心对象。

## 2. Brief 定义

`Brief` 是一次搜索生成的论文候选工作区。

它的定位不是长期知识库，而是“研究扫描 + 筛选 + 分发”的地方。

### 2.1 Brief 的创建方式

- 用户发起一次新的搜索
- 系统自动创建一个新的 `Brief`
- `Brief` 标题默认取搜索词，后续可重命名

这类似 ChatGPT 中“发起一次对话就自动创建一个会话”。

### 2.2 Brief 保存的信息

每个 `Brief` 至少应保存：

- 搜索词
- 搜索时间
- 搜索来源
- `max_results`
- 当前 Brief 中的论文总数
- 上次 rerun 时间

### 2.3 Brief 支持的操作

- 查看本次搜索结果
- 重新执行同一搜索并将新增结果追加到当前 Brief
- 删除 Brief 中不需要的论文
- 勾选论文并加入指定 `Knowledge Base`

### 2.4 Brief 的边界

- 删除 Brief 中的论文，只影响当前 Brief
- 不影响已经加入过的 `Knowledge Base`
- Brief 中的论文默认不是“自动入库”
- 入库必须由用户显式选择目标 `Knowledge Base`

## 3. Knowledge Base 定义

`Knowledge Base` 是长期沉淀的研究知识库。

它只包含用户明确加入的论文，是全文下载、解析、索引和问答的作用域。

### 3.1 Knowledge Base 支持的操作

- 查看该知识库中的论文列表
- 查看每篇论文的全文入库状态
- 对该知识库发起问答
- 查看回答对应的 citation 和证据片段

### 3.2 Knowledge Base 页面布局

进入某个具体 `Knowledge Base` 后，主界面应聚焦于：

- 左侧：该知识库中的文献列表
- 右侧：问答面板

不再单独设置一个抢占主视觉的“中间入库状态大面板”。

状态仍然需要展示，但应弱化为文献列表项中的状态 badge，例如：

- `queued`
- `downloading`
- `parsed`
- `indexed`
- `failed`

## 4. 产品工作流

### 4.1 新搜索

1. 用户输入关键词并发起搜索
2. 系统自动创建新的 `Brief`
3. 搜索结果写入该 `Brief`

### 4.2 在 Brief 中整理内容

用户在 `Brief` 中可以：

- 删除无关论文
- 勾选若干论文
- 选择一个具体 `Knowledge Base` 执行入库
- 或新建 `Knowledge Base` 后再入库

### 4.3 Rerun Search

用户可以在某个 `Brief` 上执行 `rerun search`：

1. 复用该 Brief 的原始搜索配置
2. 再次执行搜索
3. 将新增论文追加到该 Brief
4. 已存在于该 Brief 的论文不重复追加

这样一个 `Brief` 不只是一次静态搜索快照，而是一个可以持续更新的研究简报。

## 5. 信息架构

推荐的全局结构如下：

- 顶部全局搜索框：`New Search`
- 左侧固定导航：
  - `Briefs`
  - `Knowledge Bases`

### 5.1 Briefs

左侧 `Briefs` 列表按时间倒序排列，类似会话列表。

点击某个 `Brief` 后：

- 中间区域显示该 Brief 的论文列表
- 顶部显示该 Brief 的搜索配置
- 支持 `rerun search`
- 支持删除论文
- 支持加入指定 `Knowledge Base`

### 5.2 Knowledge Bases

左侧 `Knowledge Bases` 列出所有知识库。

点击某个 `Knowledge Base` 后：

- 左边显示已入库论文列表
- 右边显示问答面板

## 6. 当前版本需要落地的能力

为了让现有实现朝这个模型靠拢，当前版本应补齐：

- `Brief` 数据模型
- `Brief` 数据表
- 新搜索自动创建 `Brief`
- `Brief` 详情接口
- `Brief` rerun 接口
- `Brief` 中单篇论文删除接口
- `Brief` 到 `Knowledge Base` 的入库接口
- 前端固定侧栏和 `Briefs / Knowledge Bases` 双导航

## 7. 当前版本暂不强求的能力

- Brief 重命名
- Brief 搜索历史过滤
- 多维排序和标签系统
- 多用户共享 Brief
- Knowledge Base 之间的论文批量迁移

## 8. 当前结论

当前版本的 `ResearchMind` 应当明确采用：

- `Brief` 作为搜索工作区
- `Knowledge Base` 作为长期全文知识库
- 入库动作由用户在 `Brief` 中显式选择目标知识库
- `Knowledge Base` 页面专注于“文献列表 + 问答”

这套交互模型既延续了 `paper-tracker` 的简报基因，也更符合科研助手的实际使用方式。
