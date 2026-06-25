# diagnose_offline.py
# 诊断工具 - 显示每次检索的 Top-5 片段（含兜底检索）

import json
import os
import rag_demo_upgrade as rag

print("🔍 初始化检索组件...")
rag.initialize_retriever()
retriever = rag.get_retriever()
print("✅ 检索组件已就绪\n")

TEST_FILE = "data/test_qa.jsonl"
if not os.path.exists(TEST_FILE):
    print(f"❌ 测试集文件不存在: {TEST_FILE}")
    exit(1)

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    qas = [json.loads(line) for line in f if line.strip()]

print(f"📂 测试集共 {len(qas)} 条\n")

hit_count = 0
for idx, qa in enumerate(qas, 1):
    question = qa.get("question", "")
    answer = qa.get("answer", "").strip().lower()
    if not question or not answer:
        continue

    # 首次检索
    docs_first = retriever(question)
    top5_first = docs_first[:5]

    # 检查首次是否命中
    hit_first = False
    for doc in top5_first:
        if answer in doc.page_content.lower():
            hit_first = True
            break

    # 如果首次未命中，执行兜底检索，并将结果合并显示
    if not hit_first:
        docs_full = rag.full_retrieve(
            question,
            rag._global_embed_model,
            rag._global_faiss_index,
            rag._global_chunks,
            rag._global_bm25,
            top_k_vector=100,
            top_k_bm25=100
        )
        # 合并去重，取前5
        seen = set()
        combined = []
        for doc in docs_first:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        for doc in docs_full:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        top5 = combined[:5]
        hit_final = any(answer in doc.page_content.lower() for doc in top5)
        hit_status = "✅ 兜底命中" if hit_final else "❌ 完全未命中"
    else:
        top5 = top5_first
        hit_final = True
        hit_status = "✅ 首次命中"

    print(f"[{idx}/{len(qas)}] {hit_status}")
    print(f"   Q: {question}")
    print(f"   A: {qa['answer']}")
    print("   Top-5 片段:")
    for pos, doc in enumerate(top5, 1):
        snippet = doc.page_content.replace('\n', ' ')[:120]
        has_answer = "✅" if qa['answer'].strip().lower() in doc.page_content.lower() else " "
        print(f"      {pos}. {has_answer} {snippet}...")
    print("-" * 60)
    if hit_final:
        hit_count += 1

print(f"\n📊 最终命中率 (含兜底): {hit_count}/{len(qas)} = {hit_count/len(qas):.2%}")