# ============================================================
# 文件名: rag_demo_upgrade.py
# 功能: RAG 文档问答系统（人性化引导版）
# 特性: 智能意图识别 · 友好引导 · 多源文档感知
# ============================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import pickle
import hashlib
import time
import requests
import json
from typing import List, Optional
import numpy as np
import jieba

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# ------------- 配置区 -------------
USE_MEMORY = True       #记忆
USE_RERANK = False      #重排序
USE_NETWORK = False     #联网
SILICON_API_KEY = "LLM API "   # 大模型API
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
PDF_DIR = "./docs"
FAISS_INDEX_FILE = "./faiss_index.bin"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_VECTOR = 30
TOP_K_BM25 = 30
FINAL_K = 6
MAX_HISTORY_ROUNDS = 5

# ------------- 工具函数 -------------
def get_dir_hash(directory):
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    mtimes = [str(os.path.getmtime(os.path.join(directory, f))) for f in files]
    return hashlib.md5("".join(mtimes).encode()).hexdigest()

def clean_text(text: str) -> str:
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'收集版|历年真题收集|整理收集仅供参考|仅供学习参考', '', text)
    return text

# ------------- 加载文档（返回文档列表和文件名列表） -------------
def load_documents_from_dir(directory: str):
    cache_file = "loaded_docs_cache.pkl"
    hash_file = "loaded_docs_hash.txt"
    current_hash = get_dir_hash(directory)
    
    if os.path.exists(cache_file) and os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            saved_hash = f.read()
        if saved_hash == current_hash:
            with open(cache_file, 'rb') as f:
                print("📂 从缓存加载文档 (秒开)")
                data = pickle.load(f)
                if isinstance(data, tuple):
                    return data[0], data[1]
                else:
                    return data, []
    
    print("📖 首次加载或文件变更，解析中...")
    all_docs = []
    file_names = []
    supported_ext = ['.pdf', '.docx', '.pptx', '.txt', '.md', '.xlsx']
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext not in supported_ext:
            continue
        
        file_names.append(file)
        try:
            print(f"  📄 加载: {file}")
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif ext == '.pptx':
                from pptx import Presentation
                prs = Presentation(file_path)
                docs = []
                for slide_num, slide in enumerate(prs.slides, start=1):
                    text = "\n".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": file, "slide": slide_num}))
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            elif ext == '.md':
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            elif ext == '.xlsx':
                import pandas as pd
                df = pd.read_excel(file_path, engine='openpyxl', dtype=str)
                docs = []
                for idx, row in df.iterrows():
                    row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                    if row_text.strip():
                        docs.append(Document(page_content=row_text, metadata={"source": file, "row": idx+1}))
            else:
                continue
            
            for doc in docs:
                cleaned = clean_text(doc.page_content)
                doc.page_content = f"[来源文件：{file}] {cleaned}"
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ❌ 加载 {file} 失败: {e}")
            continue
    
    print(f"✅ 共加载 {len(all_docs)} 个文档片段，{len(file_names)} 个文件")
    with open(cache_file, 'wb') as f:
        pickle.dump((all_docs, file_names), f)
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    print("💾 缓存已保存")
    return all_docs, file_names

# ------------- 硅基 API 封装 -------------
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
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
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
            return f"❌ API 异常: {str(e)}"

# ------------- Embedding 模型（离线） -------------
class SimpleEmbeddings:
    def __init__(self, model_name, use_network=False):
        if use_network:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = SentenceTransformer(model_name, local_files_only=True, device='cpu')
    def embed_query(self, text):
        return self.model.encode(text).tolist()
    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)

# ------------- FAISS 索引 -------------
def build_or_load_faiss_index(chunks: List[Document], embedding_model):
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_INDEX_FILE + ".meta"):
        print("📂 加载已有 FAISS 索引...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_INDEX_FILE + ".meta", 'rb') as f:
            meta = pickle.load(f)
        return index, meta.get("chunks", [])
    
    print("🧮 构建 FAISS 索引...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_INDEX_FILE + ".meta", 'wb') as f:
        pickle.dump({"chunks": chunks}, f)
    print("✅ FAISS 索引已保存")
    return index, chunks

# ------------- 主程序 -------------
def main():
    print("=" * 60)
    print("📚 RAG 文档问答系统（人性化引导版）")
    print(f"🔗 联网模式: {'开启' if USE_NETWORK else '关闭 (离线)'}")
    print(f"🔍 重排序: {'开启' if USE_RERANK else '关闭'}")
    print(f"📦 向量模型: {EMBEDDING_MODEL}")
    print("=" * 60)

    raw_docs, file_names = load_documents_from_dir(PDF_DIR)
    if not raw_docs:
        print("⚠️ 无文档可加载")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️ 分块完成: {len(chunks)} 个片段")

    print("📦 加载 Embedding 模型...")
    embed_model = SimpleEmbeddings(EMBEDDING_MODEL, use_network=USE_NETWORK)

    faiss_index, stored_chunks = build_or_load_faiss_index(chunks, embed_model)
    if os.path.exists(FAISS_INDEX_FILE) and len(stored_chunks) > 0:
        chunks = stored_chunks

    print("📝 构建 BM25 索引 (jieba 分词)...")
    tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_retrieve(query: str) -> List[Document]:
        query_embedding = embed_model.embed_query(query)
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = faiss_index.search(query_vec, TOP_K_VECTOR)
        vector_docs = [chunks[i] for i in indices[0] if i < len(chunks)]

        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[-TOP_K_BM25:][::-1]
        bm25_docs = [chunks[i] for i in top_bm25_indices]

        seen = set()
        combined = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined[:FINAL_K]

    llm = SiliconFlowLLM()
    conversation_history = []

    print("\n📝 系统就绪！输入 exit 退出")
    print("💡 你可以这样问我：")
    print("   - '列出所有文档' 查看所有文件名")
    print("   - '介绍一下这些文档' 获取每个文档的摘要")
    print("   - 具体问题，例如 '双帝之战讲了什么故事？'")
    print("   - 如果我没有理解，我会友好地引导你提供更多细节\n" + "-" * 50)

    while True:
        user_input = input("❓ 请输入你的问题: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        if not user_input.strip():
            continue

        start_time = time.time()
        print("🤖 处理中...")

        # ======== 意图1：列出所有文档 ========
        if any(key in user_input for key in ["列出", "有哪些", "文件名", "几个文件", "全部文档", "文档列表", "显示所有"]):
            if not file_names:
                response = "📂 亲爱的用户，docs 文件夹中目前没有找到任何文件。请确认您已放入文档。"
            else:
                file_list = "\n".join([f"  - {name}" for name in file_names])
                response = f"📂 当前 docs 文件夹中共有 {len(file_names)} 个文件：\n{file_list}"
            print(f"💬 {response}\n")
            print(f"⏱️ 响应耗时: {time.time() - start_time:.2f} 秒\n")
            if USE_MEMORY:
                conversation_history.append((user_input, response))
            continue

        # ======== 意图2：介绍所有文档 ========
        if any(key in user_input for key in ["介绍所有", "分别介绍", "每个文档", "各个文档", "总结所有"]):
            file_list_str = "\n".join([f"- {name}" for name in file_names])
            try:
                retrieved_docs = hybrid_retrieve("文档内容摘要 主题 概括")
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            except:
                context = "未检索到具体内容。"
            
            final_prompt = f"""
你是一位友好的文档助手。用户想了解当前所有文档的概况。
已知文档列表如下：
{file_list_str}

系统从文档中检索到以下片段（可能不完整）：
{context}

请根据以上信息，为每个文档写一段简短介绍（约1-2句话），重点突出各文档的主题或用途。
如果某个文档没有足够信息，请友好地说明“暂时没有找到详细内容”，并建议用户针对该文档提出具体问题，例如：“如果您想了解《双帝之战》的具体情节，可以问我‘双帝之战讲的是什么故事？’。”
回答要简洁、清晰，用列表形式呈现，语气亲切。
"""
            try:
                response = llm.invoke(final_prompt)
                elapsed = time.time() - start_time
                print(f"💬 {response}\n")
                print(f"⏱️ 响应耗时: {elapsed:.2f} 秒\n")
                if USE_MEMORY:
                    conversation_history.append((user_input, response))
                continue
            except Exception as e:
                print(f"❌ 生成回答失败: {e}\n")
                continue

        # ======== 意图3：常规 RAG 问答（带人性化引导） ========
        try:
            retrieved_docs = hybrid_retrieve(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            continue

        if USE_MEMORY and conversation_history:
            recent = conversation_history[-MAX_HISTORY_ROUNDS:]
            history_text = "".join([f"用户曾问：{q}\n你曾回答：{a}\n" for q, a in recent])
            history_block = f"\n【对话历史】\n{history_text}\n"
        else:
            history_block = ""

        # ======== 核心：人性化引导的 Prompt ========
        final_prompt = f"""
你是一位耐心、友好的文档问答助手。你的任务是帮助用户从提供的文档片段中找到答案。

**规则：**
1. 必须基于【参考文档片段】回答，严禁编造。
2. 如果文档片段中**完全没有**相关信息，**不要直接说“不知道”**，而是用亲切的语气告诉用户：
   - “亲爱的用户，我目前找到的文档内容中没有直接回答您的问题。”
   - “您可以尝试把问题说得更具体一些，比如：‘概括一下《双帝之战》的主要内容’，或者‘请分别介绍每个文档的主题’。”
   - 根据用户的问题，提供 **1-2 个更具体的提问示例**，帮助用户调整问题。
3. 如果问题涉及多个文档，请尽量从不同来源的片段中提取信息，覆盖更广。
4. 所有回答必须使用中文，语气温和、清晰。

{history_block}
【当前用户问题】{user_input}
【参考文档片段】{context}
【你的回答】：
"""
        try:
            response = llm.invoke(final_prompt)
            elapsed = time.time() - start_time
            print(f"💬 {response}\n")
            print(f"⏱️ 响应耗时: {elapsed:.2f} 秒\n")
            if USE_MEMORY:
                conversation_history.append((user_input, response))
        except Exception as e:
            print(f"❌ 生成回答失败: {e}\n")

if __name__ == "__main__":
    main()