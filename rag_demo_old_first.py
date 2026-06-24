# ============================================================
# 文件名: rag_demo.py 
# 功能: 基于 RAG 的本地知识库问答系统
# 特性: 支持单轮/多轮对话切换（通过 USE_MEMORY 开关）
# 技术栈: LangChain + Chroma + Sentence-Transformers + 硅基API
# ============================================================
#
###-------------------首先声明  个人 vs 企业
# 数据接入与清洗层-----支持PDF，缺少WORD、EXCEL、PPT、网页等----引入 unstructured 等专业解析库，处理复杂文档
# 索引与检索层-----轻量级Chroma，缺少分布式向量数据库，检索策略---“关键词+向量”的混合检索
# 评估与监控层-----无，需建立量化评估体系(提高准确率)---引入RAGAS等评估框架
# 安全与合规层-----无，权限控制(权限必须在后端代码层面实现，不能依赖大模型自觉)、审计追踪、内容安全


import os
import requests
import json
from typing import List, Optional

# 修复 LangChain 1.0 后的模块导入路径
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM

# ------------- 配置区（在这里控制是否开启记忆） -------------
# True = 开启多轮对话记忆（记住最近5轮），False = 仅单轮问答
USE_MEMORY = True   # <--- 记忆开关***YCR

# 其他配置
SILICON_API_KEY = "API"  # 请替换成你的真实 Key
VECTOR_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V3"
PDF_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
MAX_HISTORY_ROUNDS = 5  # 记忆最近几轮对话

# ------------- 封装硅基 API -------------
class SiliconFlowLLM(LLM):
    model_name: str = LLM_MODEL_NAME
    api_key: str = SILICON_API_KEY
    temperature: float = 0.3
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                return f"❌ API 请求失败 (状态码 {response.status_code})"
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ API 调用异常: {str(e)}"

# ------------- 主程序 -------------
def main():
    print("=" * 50)
    print("🚀 RAG 知识库问答系统启动 (终极合并版)")
    print(f"🧠 记忆模式: {'开启 (保留最近' + str(MAX_HISTORY_ROUNDS) + '轮)' if USE_MEMORY else '关闭 (单轮问答)'}")
    print("=" * 50)

    # 1. 检查并处理 PDF
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"⚠️ 已创建 {PDF_DIR}，请放入 PDF 后重新运行。")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️ 在 {PDF_DIR} 中未找到 PDF 文件。请放入文档后重试。")
        return

    # 2. 加载/创建向量库
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("📂 加载已有向量数据库...")
        embeddings = HuggingFaceEmbeddings(model_name=VECTOR_MODEL_NAME)
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        print("📄 首次运行，正在处理 PDF...")
        all_docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(PDF_DIR, pdf_file))
            all_docs.extend(loader.load())
        if not all_docs:
            print("❌ 无法读取 PDF 内容。")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name=VECTOR_MODEL_NAME)
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
        vector_store.persist()
        print(f"✅ 向量库构建完成，共 {len(chunks)} 个文本块")

    # 3. 初始化检索器和 LLM
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = SiliconFlowLLM()

    # 4. 对话记忆容器（只在 USE_MEMORY=True 时使用）
    conversation_history = []

    print("\n📝 准备就绪！输入问题开始测试（输入 exit 退出）\n" + "-" * 50)

    while True:
        user_input = input("❓ 请输入你的问题: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 再见！")
            break
        if not user_input.strip():
            print("⚠️ 问题不能为空。")
            continue

        print("🤖 正在处理...")

        # ----- 步骤 A：用当前问题检索文档（检索不依赖历史） -----
        try:
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            continue

        # ----- 步骤 B：构建提示词（根据 USE_MEMORY 决定是否包含历史） -----
        if USE_MEMORY and conversation_history:
            # 提取最近 N 轮对话
            recent = conversation_history[-MAX_HISTORY_ROUNDS:]
            history_text = ""
            for q, a in recent:
                history_text += f"用户曾问：{q}\n你曾回答：{a}\n"
            history_block = f"\n【对话历史】\n{history_text}\n"
        else:
            history_block = ""

        # 强制中文回答 + 严格基于文档
        final_prompt = f"""
你是一个严谨的文档问答助手。请遵守以下规则：
1. 必须基于【参考文档片段】回答，严禁编造。
2. 如果文档中没有相关信息，请直接回答“不知道，文档里没写”。
3. **所有回答必须使用中文**。
{history_block}
【当前用户问题】
{user_input}

【参考文档片段】
{context}

【你的回答】：
"""

        # ----- 步骤 C：调用模型并保存历史 -----
        try:
            response = llm.invoke(final_prompt)
            print(f"💬 回答: {response}\n")

            # 如果开启了记忆，将本轮对话存进去
            if USE_MEMORY:
                conversation_history.append((user_input, response))
                # 控制内存大小，防止无限增长
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-MAX_HISTORY_ROUNDS:]

        except Exception as e:
            print(f"❌ 生成回答时出错: {e}\n")

# ------------- 程序入口 -------------
if __name__ == "__main__":
    main()