```markdown
# RAG 多源文档问答系统
基于LangChain + FAISS + BM25 混合检索的RAG（检索增强生成）系统，支持6种文档格式，
具备多轮对话记忆、文件列表感知、人性化引导等功能，可离线运行。
---
## 功能特点
- 多源文档支持：PDF、Word（.docx）、PPT（.pptx）、Excel（.xlsx）、TXT、Markdown（.md）
- 混合检索：FAISS 向量检索 + BM25 关键词检索，兼顾语义理解与精确匹配
- 文档缓存机制：基于文件哈希的增量加载，首次加载后秒级启动
- 多轮对话记忆：自动记住最近 5 轮对话，支持上下文追问
- 文件列表感知：支持“列出所有文档”等自然语言指令
- 人性化引导：当无法回答时，主动提供提问示例，引导用户细化问题
- 完全离线运行：模型本地缓存，无需联网（首次需下载模型，后续全离线），内置联网开关，网络畅通可自动更新
---
## 环境要求
- Python 3.11+（项目使用 3.11.9）
- Git 2.54.0.1
- VS Code
- Windows / macOS / Linux
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
├── docs/                          # 源文档目录
├── rag_demo_upgrade.py            # 主程序（核心文件）
├── rag_demo_old_first.py          # 早期版本参考（可忽略）
├── .gitignore                     # Git 忽略文件
├── requirements.txt               # 依赖清单
└── README.md                      # 项目说明
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
