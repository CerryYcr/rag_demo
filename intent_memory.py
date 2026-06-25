# ============================================================
# 文件名: intent_memory.py
# 功能: 意图记忆管理器
# ============================================================

import os
import json
import hashlib
import time
from typing import Dict, List, Optional
import jieba

class IntentMemory:
    def __init__(self, history_file: str = "intent_history.jsonl"):
        self.history_file = history_file
        self.records = []
        self._load_records()

    def _load_records(self):
        self.records = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.records.append(json.loads(line))
                        except:
                            continue

    def _save_record(self, record: Dict):
        with open(self.history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.records.append(record)

    def save_all(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    @staticmethod
    def _compute_fingerprint(query: str) -> str:
        words = [w for w in jieba.cut(query) if len(w) > 1 and w not in ['什么', '怎么', '为什么', '哪个', '哪些', '如何']]
        words = sorted(words)[:5]
        text = ''.join(words)
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def find_match(self, query: str) -> Optional[Dict]:
        query_fingerprint = self._compute_fingerprint(query)
        for rec in self.records:
            if rec.get('fingerprint') == query_fingerprint:
                return rec
        query_words = set(jieba.cut(query))
        best_match = None
        best_overlap = 0
        for rec in self.records:
            rec_words = set(rec.get('keywords', []))
            overlap = len(query_words & rec_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = rec
        if best_match and best_overlap >= 2:
            return best_match
        return None

    def add_record(self, query: str, doc_scores: Dict[str, int], quotas: Dict[str, int], hit_rate: float = None, usage_count: int = 1):
        keywords = list(set([w for w in jieba.cut(query) if len(w) > 1]))
        record = {
            "fingerprint": self._compute_fingerprint(query),
            "query": query,
            "keywords": keywords,
            "doc_scores": doc_scores,
            "quotas": quotas,
            "hit_rate": hit_rate,
            "usage_count": usage_count,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_used": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_record(record)
        return record

    def get_scores(self, query: str) -> Optional[Dict[str, int]]:
        match = self.find_match(query)
        return match.get('doc_scores') if match else None

    def get_quotas(self, query: str) -> Optional[Dict[str, int]]:
        match = self.find_match(query)
        return match.get('quotas') if match else None