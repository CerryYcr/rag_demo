```markdown
# RAG 多源文档问答系统
基于LangChain + FAISS + BM25 混合检索的RAG（检索增强生成）系统，支持6种文档格式，
具备多轮对话记忆、文件列表感知、人性化引导等功能，可离线运行。
---
## 功能特点
-多源文档支持：PDF、Word（.docx）、PPT（.pptx）、Excel（.xlsx）、TXT、Markdown（.md）
-混合检索：FAISS 向量检索 + BM25 关键词检索，兼顾语义理解与精确匹配
-文档缓存机制：基于文件哈希的增量加载，首次加载后秒级启动
-多轮对话记忆：自动记住最近 5 轮对话，支持上下文追问
-文件列表感知：支持“列出所有文档”等自然语言指令
-人性化引导：当无法回答时，主动提供提问示例，引导用户细化问题
-完全离线运行：模型本地缓存，无需联网（首次需下载模型，后续全离线），内置联网开关，网络畅通可自动更新
---
## 环境要求
- Python 3.11+（项目使用 3.11.9）
- Git 2.54.0.1
- Windows / macOS / Linux
```
## 更新
```bash
模块	   更新内容
-意图分析	 新增规则匹配 + LLM 双保险机制，优先通过关键词精准命中文档（如“双帝”→“双帝之战.docx”），大幅提升首次检索命中率
-兜底检索   新增全文档混合检索（full_retrieve），当首次检索未命中时自动触发，确保答案不遗漏
-离线评估   完善 evaluate_offline.py，模拟主程序完整流程（首次检索 + 兜底检索），输出 Hit Rate 和 MRR，真实反映系统能力
-在线监控	 每次问答自动记录结构化日志到 logs/rag_log.jsonl，包含问题、回答、耗时、检索片段数等
-质量评估	 支持 LLM-as-Judge 后台异步打分（忠实度 + 相关性），需通过 USE_QUALITY_EVAL 开关启用
-记忆系统	 intent_memory.py + intent_history.jsonl，实现意图分析结果持久化，越用越快
-检索参数优化	TOP_K_VECTOR 提升至 80，FINAL_K 提升至 20，大幅提高召回率
-代码质量	 完善调试日志，意图分析结果和配额分配实时可见，方便排查问题
```
## 安装指南
### 1. 克隆项目
```bash
git clone https://github.com/CerryYcr/rag_demo.git
cd rag_demo
```
### 2. 创建并激活虚拟环境
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```
### 3. 安装依赖
```bash
pip install -r requirements.txt
```
如果安装缓慢，可使用国内镜像源加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 升级 pip（可选）

```bash
python -m pip install --upgrade pip
```

### 5. 配置 API Key

在 `rag_demo_upgrade.py` 中，将 `SILICON_API_KEY` 替换为 LLM API Key。

### 6. 文档存储

将需要检索的文档放入 `docs/` 文件夹。

---

## 项目结构

```
rag_demo/
├── rag_demo_upgrade.py         # 主程序（核心）
├── intent_memory.py            # 意图记忆模块
├── evaluate_offline.py         # 离线评估
├── diagnose_offline.py         # 诊断工具
├── debug_pptx.py               # PPTX 诊断
├── data/test_qa.jsonl          # 测试集
├── logs/                       # 运行日志
├── reports/                    # 评估报告
├── docs/                       # 源文档目录
├── .gitignore                  # Git忽略
├── requirements.txt            # 依赖清单
└── README.md                   # 项目说明
```
---

## 项目流程

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 1: 意图分析（规则匹配 + LLM 双保险）      │
│  → 判断问题类型：通用问答 / 文件列表 / 特定文档   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 2: 混合检索（并行执行，权重融合）          │
│  ├── FAISS 向量检索（语义理解）                │
│  ├── BM25 关键词检索（精确匹配）               │
│  └── 动态配额分配（根据文档大小/格式分配预算）    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 3: 重排序（可选，Cross-Encoder）        │
│  → 精排前 N 个候选 chunk                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 4: 上下文构建 + LLM 生成               │
│  → 注入对话历史（最近 5 轮）+ 检索结果          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 5: 意图记忆持久化（intent_history.jsonl）│
│  → 记录问题类型 → 下次相似问题直接复用          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Step 6: 兜底机制                            │
│  → 如果首次检索无结果，触发全文档检索           │
│  → 如果仍无结果，返回人性化引导                │
└─────────────────────────────────────────────┘
```
---

## 技术栈

| 类别 | 工具/库 |
| :--- | :--- |
| 核心框架 | LangChain, LangChain-Community |
| 向量检索 | FAISS (CPU), Sentence-Transformers |
| 关键词检索 | BM25 (rank-bm25) |
| 中文分词 | Jieba |
| 文档加载 | PyPDFLoader, Docx2txtLoader, python-pptx, pandas/openpyxl |
| 大模型 API | 硅基流动（SiliconFlow）DeepSeek 系列 |
| 开发语言 | Python 3.11+ |

---

## 注意事项

- 首次运行会自动下载 Embedding 模型（`paraphrase-multilingual-MiniLM-L12-v2`，约 120MB），请确保网络畅通。
- 如果网络不通，可在代码中将 `USE_NETWORK = False`，强制使用本地缓存（需先成功下载过一次）。
- 重排序（Cross-Encoder）功能默认关闭（`USE_RERANK = False`），如需提高精度可开启，但会增加响应时间。

---

## 作者

**CerryYcr**  
GitHub: [@CerryYcr](https://github.com/CerryYcr)
```
