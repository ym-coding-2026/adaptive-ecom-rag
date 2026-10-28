# retrieval.py —— 混合检索（BM25 稀疏 + 稠密向量 + RRF 融合）
#
# 为什么叫「混合」？因为只靠一种检索都不够好：
#   - BM25（关键词匹配）：能精准命中「七天无理由」「5000mAh」这种具体词，但不懂同义词。
#   - 稠密向量（语义匹配）：能懂「电池多大」≈「5000mAh」，但容易漏掉精确条款号。
#   - RRF（倒数排名融合）：把两种排名拼在一起，谁排前面谁得分高，取长补短。
#
# 稠密向量怎么来？两种模式，可插拔、自动降级：
#   - "bge" / "auto"：用 BAAI/bge-small-zh-v1.5（中文语义 Embedding，512 维）。
#                      首次会用 HF 镜像（hf-mirror.com）下载模型，需要联网一次。
#   - "hash"：纯 numpy 哈希嵌入基线，零依赖、离线可跑（架构演示用）。
#   没装 sentence-transformers 或模型下载失败时，"auto" 自动回退到 "hash"，绝不崩。

import os
import re
import math
import numpy as np
from kb import load_knowledge

# 国内走镜像下模型，不覆盖用户已有的 HF_ENDPOINT 配置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_bge_model = None  # 懒加载，避免 import 慢或失败时一启动就崩


def _get_bge_model():
    """懒加载 BGE 模型（单例）。第一次调用才下载/加载，之后复用。"""
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer(_BGE_MODEL_NAME)
    return _bge_model


# ============ 重排（cross-encoder 精排，可插拔） ============
# RRF 融合是「排名级」融合，仍可能把真正相关的条款压在后面；
# 再叠一层 cross-encoder 重排（BAAI/bge-reranker-base）做「相关性级」精排，
# 直接对 (query, 候选) 打分，纠正融合偏差。模型可用就增强、不可用就跳过。
_RERANKER_NAME = "BAAI/bge-reranker-base"
_reranker_model = None


def _get_reranker():
    """懒加载 cross-encoder 重排模型（单例）。失败抛异常，由调用方决定降级。"""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(_RERANKER_NAME)
    return _reranker_model


def tokenize(text):
    """中文 + 英文混合分词。
    中文没有空格，所以：每个汉字单独成 token + 相邻两字组成二元（bigram）token；
    英文/数字按单词。这样「电池容量」和「5000mAh」能分别被不同方式命中。"""
    text = (text or "").lower()
    en = re.findall(r"[a-z0-9]+", text)                      # 英文单词、数字
    cjk = re.findall(r"[一-鿿]", text)                         # 单个汉字
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]  # 汉字二元
    return en + cjk + bigrams


class Retrieval:
    def __init__(self, chunks=None, dense="auto", dense_dim=512, rerank="auto"):
        self.chunks = chunks if chunks is not None else load_knowledge()
        self.dense_dim = dense_dim
        self.dense_mode = self._resolve_dense(dense)   # "bge" 或 "hash"
        self.rerank_on = self._resolve_rerank(rerank)  # True / False
        self._build_bm25()
        self._build_dense()

    def _resolve_dense(self, dense):
        """决定稠密向量走哪条路。auto 时尝试加载 BGE，失败则降级哈希。"""
        if dense == "hash":
            return "hash"
        # dense == "bge" 或 "auto"：尝试加载 BGE
        try:
            _get_bge_model()
            return "bge"
        except Exception as e:
            if dense == "bge":
                raise  # 用户明确要求 bge 却失败，把错误抛出去，别偷偷降级
            print(f"[检索] BGE 加载失败，降级为哈希基线：{e.__class__.__name__}")
            return "hash"

    def _resolve_rerank(self, rerank):
        """决定要不要开 cross-encoder 精排。
        - "off"：永远不开（评测语义对比用，隔离重排影响）。
        - "on"：明确要求，加载失败直接抛错，不偷偷降级。
        - "auto"（默认）：模型可用就开，加载失败则降级关，核心流程永不崩。"""
        if rerank == "off":
            return False
        try:
            _get_reranker()
            return True
        except Exception as e:
            if rerank == "on":
                raise  # 用户明确要开却失败，如实报错
            print(f"[检索] reranker 加载失败，关闭精排：{e.__class__.__name__}")
            return False

    # ---------- BM25（手写，无需任何库） ----------
    def _build_bm25(self):
        self.docs_tok = [tokenize(c["text"]) for c in self.chunks]
        self.df = {}
        for toks in self.docs_tok:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs_tok)
        self.avgdl = sum(len(t) for t in self.docs_tok) / max(self.N, 1)
        # IDF：一个词越少见越重要
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def _bm25_scores(self, query, topk=20):
        qt = set(tokenize(query))
        out = []
        for i, dt in enumerate(self.docs_tok):
            dl = len(dt)
            freq = {}
            for t in dt:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for t in qt:
                if t in self.idf:
                    f = freq.get(t, 0)
                    if f > 0:
                        # 标准 BM25 公式（k1=1.5, b=0.75）
                        score += self.idf[t] * (f * 2.5) / (f + 1.5 * (1 - 0.75 + 0.75 * dl / self.avgdl))
            out.append(score)
        ranked = sorted(range(self.N), key=lambda i: -out[i])[:topk]
        return [(i, out[i]) for i in ranked]

    # ---------- 稠密向量 ----------
    def _embed(self, text):
        """把单条文本变成向量。
        bge 模式：真语义 Embedding（512 维，已归一化）。
        hash 模式：哈希技巧基线（纯 numpy，捕获词面重合，非语义）。"""
        if self.dense_mode == "bge":
            v = _get_bge_model().encode(text, normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)
        # hash 基线（离线、零依赖）
        # 注意：不能用内置 hash(t)，它每个进程随机加盐（PYTHONHASHSEED），
        # 会让基线向量跨进程不一致、评测不可复现。改用确定性 crc32。
        import zlib
        v = np.zeros(self.dense_dim, dtype=np.float32)
        for t in tokenize(text):
            v[zlib.crc32(t.encode("utf-8")) % self.dense_dim] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    def _build_dense(self):
        if self.dense_mode == "bge":
            # 批量编码所有知识块（一次性，之后检索只编码 query）
            texts = [c["text"] for c in self.chunks]
            vecs = _get_bge_model().encode(texts, normalize_embeddings=True)
            self.dense_vecs = np.asarray(vecs, dtype=np.float32)
        else:
            self.dense_vecs = np.stack([self._embed(c["text"]) for c in self.chunks])

    def _dense_scores(self, query, topk=20):
        qv = self._embed(query)
        sims = self.dense_vecs @ qv  # 余弦相似度（已归一化）
        ranked = np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in ranked]

    # ---------- RRF 融合 ----------
    def retrieve(self, query, topk=5):
        """返回融合后的 top-k 文档块，每个带 score。开启 rerank 时再做一遍精排。"""
        bm = self._bm25_scores(query, topk=20)
        ds = self._dense_scores(query, topk=20)
        k = 60  # RRF 常数，越大越削弱排名靠后项的影响
        fused = {}
        for rank, (i, _) in enumerate(bm):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
        for rank, (i, _) in enumerate(ds):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
        # 候选池：融合排序后的前 30 个进 rerank；rerank 也跑不动时直接用融合结果
        pool = sorted(fused.items(), key=lambda kv: -kv[1])[:30]
        if self.rerank_on and len(pool) > topk:
            try:
                docs = self._rerank(query, pool, topk)
            except Exception as e:
                print(f"[检索] rerank 打分失败，回退融合排序：{e.__class__.__name__}")
                docs = [dict(self.chunks[i], score=round(fused[i], 4)) for i, _ in pool[:topk]]
        else:
            docs = [dict(self.chunks[i], score=round(fused[i], 4)) for i, _ in pool[:topk]]
        return docs

    def _rerank(self, query, pool, topk):
        """用 cross-encoder 对融合候选池重排，纠正融合排序偏差。"""
        pairs = [(query, self.chunks[i]["text"]) for i, _ in pool]
        scores = _get_reranker().predict(pairs)
        ranked = sorted(range(len(pool)), key=lambda j: -float(scores[j]))[:topk]
        docs = []
        for j in ranked:
            i = pool[j][0]
            docs.append(dict(self.chunks[i], score=round(float(scores[j]), 4)))
        return docs


# ============ 重排（已实现，可插拔） ============
# 检索默认走 BM25 + 稠密 + RRF 融合；开启 rerank 时，再对候选池用
# BAAI/bge-reranker-base（cross-encoder）做精排，纠正融合排序偏差。
# rerank="auto" 下，模型可用就增强、不可用就跳过，核心流程永不崩。


if __name__ == "__main__":
    r = Retrieval()
    print(f"索引了 {len(r.chunks)} 个知识块，稠密向量模式：{r.dense_mode}")
    for d in r.retrieve("七天无理由退货怎么操作", topk=3):
        print(f"[{d['score']}] {d['title']} —— {d['text'][:40]}")
