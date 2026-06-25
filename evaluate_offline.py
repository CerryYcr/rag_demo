# evaluate_offline.py
# 离线评估 - 模拟主程序完整流程（首次检索 + 兜底检索）

import json
import os
import rag_demo_upgrade as rag

print("📊 初始化检索组件...")
rag.initialize_retriever()
retriever = rag.get_retriever()
print("✅ 检索组件已就绪\n")

TEST_FILE = "data/test_qa.jsonl"
TOP_K = 5

if not os.path.exists(TEST_FILE):
    print(f"❌ 测试集文件不存在: {TEST_FILE}")
    exit(1)

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    qas = [json.loads(line) for line in f if line.strip()]

print(f"📂 测试集规模: {len(qas)} 条\n")

hit_count_first = 0
hit_count_final = 0
reciprocal_ranks_first = []
reciprocal_ranks_final = []

for idx, qa in enumerate(qas, 1):
    question = qa.get("question", "")
    answer = qa.get("answer", "").strip().lower()
    if not question or not answer:
        continue

    # ----- 首次检索 -----
    docs_first = retriever(question)
    top_k_first = docs_first[:TOP_K]

    pos_first = -1
    for pos, doc in enumerate(top_k_first, start=1):
        if answer in doc.page_content.lower():
            pos_first = pos
            break

    # ----- 兜底检索（如果首次未命中）-----
    docs_final = docs_first[:]  # 拷贝
    if pos_first == -1:
        # 从 rag 模块获取最新的全局变量
        docs_full = rag.full_retrieve(
            question,
            rag._global_embed_model,
            rag._global_faiss_index,
            rag._global_chunks,
            rag._global_bm25,
            top_k_vector=100,
            top_k_bm25=100
        )
        # 合并去重
        seen = set()
        for doc in docs_first:
            seen.add(doc.page_content)
        for doc in docs_full:
            if doc.page_content not in seen:
                docs_final.append(doc)
        # 只取前 TOP_K 用于判断
        docs_final = docs_final[:TOP_K]
    else:
        docs_final = top_k_first

    pos_final = -1
    for pos, doc in enumerate(docs_final, start=1):
        if answer in doc.page_content.lower():
            pos_final = pos
            break

    # ----- 统计 -----
    if pos_first != -1:
        hit_count_first += 1
        reciprocal_ranks_first.append(1.0 / pos_first)
        status = f"✅ 首次命中 (位置: {pos_first})"
    else:
        reciprocal_ranks_first.append(0.0)
        if pos_final != -1:
            hit_count_final += 1
            reciprocal_ranks_final.append(1.0 / pos_final)
            status = f"⚠️ 首次未命中 → 兜底命中 (位置: {pos_final})"
        else:
            reciprocal_ranks_final.append(0.0)
            status = "❌ 完全未命中"

    print(f"[{idx}/{len(qas)}] {status} | {question[:30]}...")

total = len(qas)
hit_rate_first = hit_count_first / total if total > 0 else 0.0
hit_rate_final = (hit_count_first + hit_count_final) / total if total > 0 else 0.0
mrr_first = sum(reciprocal_ranks_first) / total if total > 0 else 0.0
mrr_final = sum(reciprocal_ranks_final) / total if total > 0 else 0.0

print("\n" + "=" * 60)
print("📊 离线评估报告")
print("=" * 60)
print(f"📌 测试集规模: {total} 条")
print(f"📌 首次 Hit Rate@{TOP_K}: {hit_rate_first:.4f} ({hit_count_first}/{total})")
print(f"📌 最终 Hit Rate@{TOP_K} (含兜底): {hit_rate_final:.4f} ({hit_count_first + hit_count_final}/{total})")
print(f"📌 首次 MRR: {mrr_first:.4f}")
print(f"📌 最终 MRR (含兜底): {mrr_final:.4f}")
print("=" * 60)