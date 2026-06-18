#!/usr/bin/env python
"""
14_RAG_Hybrid_Search.py
混合检索 RAG 示例：BM25 关键词检索 + 向量语义检索 + RRF 融合 + DeepSeek 生成

实现一个完整的 RAG（检索增强生成）流水线，演示如何结合：
  - BM25（关键词匹配，适合精确查询）
  - 向量检索（语义匹配，适合意图理解）
  - RRF 融合算法（Reciprocal Rank Fusion，融合两种检索的最佳排序）
  - DeepSeek 大模型（基于检索结果生成回答）

数据来源：/Users/ye/code/agent_learning/data/ 下的 DeepSeek 知识库文件
使用方式：python 14_rag_hybrid_search.py
"""

import os
import glob
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

from dotenv import load_dotenv
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

# ===================== 1. 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== 2. 加载环境变量 & 初始化 DeepSeek =====================
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


# ===================== 3. 文档加载与分块 =====================

def load_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """加载 data/ 目录下所有 .txt 文档"""
    docs = []
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    logger.info(f"发现 {len(txt_files)} 个文档文件")
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        filename = os.path.basename(filepath)
        docs.append(Document(page_content=text, metadata={"source": filename}))
        logger.info(f"  加载: {filename}（{len(text)} 字符）")
    return docs


def split_chunks(docs: List[Document], chunk_size: int = 300, overlap: int = 50) -> List[Document]:
    """将文档切分为固定大小的文本块，便于检索"""
    splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_documents(docs)
    # 为每个块分配 chunk_id，用于 RRF 融合时追踪文档身份
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    logger.info(f"切分为 {len(chunks)} 个文本块")
    return chunks


# ===================== 4. BM25 检索器（关键词） =====================

class BM25Retriever:
    """
    基于 BM25 算法的关键词检索器
    使用 jieba 进行中文分词，擅长捕获精确的关键词匹配
    """

    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        logger.info("构建 BM25 索引...")
        tokenized_corpus = [self._tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 索引构建完成")

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """检索 top_k 个最相关的文档块，返回 (文档, BM25分数)"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]


# ===================== 5. 向量检索器（稠密向量语义） =====================

class VectorRetriever:
    """
    基于 TF-IDF + SVD 降维的稠密向量检索器
    流程：字符 n-gram TF-IDF → SVD 降维至 128 维 → L2 归一化 → 余弦相似度检索
    通过向量空间模型捕捉字词共现模式，与 BM25 的词级关键词匹配形成互补
    """

    def __init__(self, chunks: List[Document], vector_dim: int = 128):
        self.chunks = chunks
        texts = [c.page_content for c in chunks]

        logger.info("构建字符 n-gram TF-IDF 矩阵...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer="char",
            ngram_range=(2, 4),
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        logger.info(f"SVD 降维至 {vector_dim} 维稠密向量...")
        self.svd = TruncatedSVD(n_components=vector_dim, random_state=42)
        dense_vectors = self.svd.fit_transform(tfidf_matrix)

        # L2 归一化 → 余弦相似度等价于内积
        self.normalizer = Normalizer()
        self.vectors = self.normalizer.fit_transform(dense_vectors)
        logger.info(f"向量索引构建完成（shape: {self.vectors.shape}）")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """将查询转为稠密向量，计算余弦相似度，返回 top_k 结果"""
        query_vec = self.vectorizer.transform([query])
        query_vec = self.svd.transform(query_vec)
        query_vec = self.normalizer.transform(query_vec)

        # 余弦相似度（基于 L2 归一化后的内积）
        similarities = np.dot(self.vectors, query_vec.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.chunks[i], float(similarities[i])) for i in top_indices]


# ===================== 6. 混合检索器（RRF 融合） =====================

@dataclass
class RetrievalResult:
    """单个文档的检索结果，记录来自 BM25 和向量检索的双重分数"""
    document: Document
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0


class HybridRetriever:
    """
    混合检索器：使用 RRF（Reciprocal Rank Fusion）融合 BM25 和向量检索结果

    RRF 公式：score(d) = 1/(k + r₁(d)) + 1/(k + r₂(d))
    其中 k 为常数（默认 60），r 为文档 d 在各自检索系统中的排名（从1开始）
    """

    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks)

    def search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[RetrievalResult]:
        """
        混合检索流程：
        1. 分别从 BM25 和向量检索获取候选结果（扩大候选池）
        2. 按文档 chunk_id 合并分数
        3. 计算 RRF 融合得分
        4. 返回 top_k 最佳结果
        """
        # 1. 执行两种检索（候选池扩大 3 倍给 RRF 更多融合空间）
        bm25_hits = self.bm25.retrieve(query, top_k=top_k * 3)
        vector_hits = self.vector.retrieve(query, top_k=top_k * 3)

        logger.info(f"BM25 检索到 {len(bm25_hits)} 个结果")
        logger.info(f"向量检索到 {len(vector_hits)} 个结果")

        # 2. 用 chunk_id 为 key 合并结果
        merged: Dict[int, RetrievalResult] = {}

        for doc, score in bm25_hits:
            cid = doc.metadata["chunk_id"]
            merged[cid] = RetrievalResult(document=doc, bm25_score=score)

        for doc, score in vector_hits:
            cid = doc.metadata["chunk_id"]
            if cid not in merged:
                merged[cid] = RetrievalResult(document=doc)
            merged[cid].vector_score = score

        # 3. 构建排名字典（排名从 1 开始）
        bm25_ranks = {doc.metadata["chunk_id"]: i + 1
                      for i, (doc, _) in enumerate(bm25_hits)}
        vector_ranks = {doc.metadata["chunk_id"]: i + 1
                        for i, (doc, _) in enumerate(vector_hits)}

        # 4. 计算 RRF 分数
        # 未命中特定检索系统的文档给予一个极大排名（贡献趋近于 0）
        for cid, result in merged.items():
            r1 = bm25_ranks.get(cid, 1000)
            r2 = vector_ranks.get(cid, 1000)
            result.rrf_score = 1.0 / (rrf_k + r1) + 1.0 / (rrf_k + r2)

        # 5. 按 RRF 分数降序排列，取 top_k
        results = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)
        return results[:top_k]


# ===================== 7. DeepSeek 生成回答 =====================

def generate_answer(query: str, context: List[RetrievalResult]) -> str:
    """基于检索到的上下文，使用 DeepSeek 生成回答"""
    context_text = "\n\n---\n\n".join(
        f"[来源：{r.document.metadata['source']}]\n{r.document.page_content}"
        for r in context
    )

    prompt = f"""你是一个基于检索增强生成（RAG）的智能问答助手。

请根据以下检索到的参考信息，准确回答用户的问题。
注意：如果参考信息不足以回答问题，请直接说明，不要编造。

【检索到的参考信息】
{context_text}

【用户问题】
{query}

请基于参考信息给出准确、简洁的回答。"""

    logger.info("调用 DeepSeek 生成回答...")
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.warning(f"DeepSeek API 连接失败: {e}")
        return f"[错误] DeepSeek API 连接失败: {e}。\nRAG 检索已完成，请在本地环境运行以获取 DeepSeek 生成结果。"


# ===================== 8. 主流程：三方检索对比 =====================

def print_divider(title: str):
    """打印格式化的分隔标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result_card(result: RetrievalResult, rank: int):
    """打印单个检索结果的详细信息"""
    src = result.document.metadata["source"]
    preview = result.document.page_content[:100].replace("\n", " ")
    print(f"\n  ┌─ [#{rank}] 来源: {src}")
    print(f"  ├─ BM25  分数: {result.bm25_score:.4f}")
    print(f"  ├─ 向量  分数: {result.vector_score:.4f}")
    print(f"  ├─ RRF   分数: {result.rrf_score:.4f}")
    print(f"  └─ 内容预览: {preview}...")


def main():
    """运行完整的 RAG 演示流程"""
    print_divider("14_RAG_Hybrid_Search — 混合检索 RAG 示例")
    print("  检索方法：BM25（关键词） + 向量（语义） + RRF 融合")
    print("  生成模型：DeepSeek-V3 (deepseek-chat)")
    print(f"  数据来源：{DATA_DIR}")

    # ---------- 步骤 1：加载数据 ----------
    print_divider("1. 加载文档与分块")
    docs = load_documents()
    chunks = split_chunks(docs)
    print(f"  共 {len(chunks)} 个文本块")

    # ---------- 步骤 2：构建检索器 ----------
    print_divider("2. 构建混合检索器")
    hybrid = HybridRetriever(chunks)

    # ---------- 步骤 3：测试查询 ----------
    test_queries = [
        "DeepSeek-R1的核心优势是什么？",
        "DeepSeek API的价格是多少？",
        "使用DeepSeek有哪些最佳实践？",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print_divider(f"3.{q_idx} 查询：{query}")

        # 3.1 BM25 单独检索（作为对比基线）
        print("\n  >> [Baseline] BM25 检索结果：")
        bm25_only = hybrid.bm25.retrieve(query, top_k=2)
        for rank, (doc, score) in enumerate(bm25_only, 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"     [{rank}] {doc.metadata['source']} (score={score:.2f})")
            print(f"            {preview}...")

        # 3.2 向量单独检索（作为对比基线）
        print("\n  >> [Baseline] 向量检索结果：")
        vec_only = hybrid.vector.retrieve(query, top_k=2)
        for rank, (doc, score) in enumerate(vec_only, 1):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"     [{rank}] {doc.metadata['source']} (score={score:.4f})")
            print(f"            {preview}...")

        # 3.3 混合检索（BM25 + 向量 + RRF 融合）
        print("\n  >> [Hybrid] 混合检索结果（BM25 + 向量 + RRF）：")
        results = hybrid.search(query, top_k=2)
        for rank, result in enumerate(results, 1):
            print_result_card(result, rank)

        # 3.4 DeepSeek 生成回答
        print("\n  >> 基于检索结果，DeepSeek 生成回答：")
        answer = generate_answer(query, results)
        print(f"  ┌─ {'─' * 50}")
        for line in answer.strip().split("\n"):
            print(f"  │ {line}")
        print(f"  └─ {'─' * 50}")

    print_divider("演示结束")


if __name__ == "__main__":
    main()
