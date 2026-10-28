# rewrite.py —— 查询改写（让口语化问题更适合检索）
#
# 为什么要改写？用户常说「这个玩意儿过保了坏了我自己掏钱修还是你们管」，
# 而知识库里写的是「质量问题退换」。直接拿原句检索，BM25 可能一个词都撞不上。
# 改写把口语翻成「检索友好」的查询：补全指代、去语气词、保留关键实体。
#
# 两级实现，自动降级：
#   - 有 DEEPSEEK_API_KEY：用 LLM 做语义改写（最准）。
#   - 无 Key / 调用失败：规则改写（保守地去前缀、清理标点），至少比原句干净。

import os
import re

from llm import call_deepseek

REWRITE_SYSTEM = (
    "你是检索查询改写器。把用户的口语化问题改写成适合向量/关键词检索的简洁查询。"
    "规则：1) 必须保留原意和关键实体（订单号、产品名、数字、政策名）；"
    "2) 补全明显的指代（如无可补则保留原词）；3) 去除问候、语气词、冗余表达；"
    "4) 只输出改写后的一句话，不要任何解释或引号。"
)

# 规则改写的常见口语前缀（去掉后更接近检索词）
_PREFIXES = [
    "请问", "我想知道", "我想问", "我想了解一下", "帮我查一下", "帮我看看",
    "怎么弄", "怎么搞", "如何", "怎么", "怎样", "我想", "可以", "能不能",
    "是否", "那个", "这个", "我想问下", "麻烦问下", "问一下",
]

# 末端的冗余语气词
_SUFFIXES = ["呢", "啊", "吧", "呀", "哦", "哈", "吗", "嘛"]


def _rule_rewrite(query):
    """无 Key 时的保守规则改写：去掉口语前缀、清理首尾标点与冗余语气词。"""
    q = (query or "").strip()
    # 循环去掉前缀（可能叠加「请问怎么」）
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if q.startswith(p):
                q = q[len(p):].strip()
                changed = True
                break
    # 去掉首尾标点/空白
    q = q.strip("？?。. 　，,；;：:、")
    # 去掉末尾单个语气词
    for s in _SUFFIXES:
        if q.endswith(s):
            q = q[:-len(s)].strip("？?。. 　，,；;：:、")
            break
    return q if q else (query or "")


def rewrite_query(query, force_llm=None):
    """返回改写后的查询。force_llm=True/False 可强制；默认按是否有 Key 决定。"""
    if force_llm is None:
        force_llm = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if force_llm:
        try:
            api_key = os.environ["DEEPSEEK_API_KEY"]
            out = call_deepseek(api_key, query, system=REWRITE_SYSTEM).strip()
            out = out.strip('"\'「」『』').strip()      # 清理可能包裹的引号
            out = out.split("\n")[0].strip()            # 只取首行，避免多行解释
            if out and len(out) >= 2:
                return out
        except Exception:
            pass
    return _rule_rewrite(query)


if __name__ == "__main__":
    for q in ["请问手机 X1 的电池容量是多少", "这个玩意儿过保了坏了我自己掏钱修还是你们管"]:
        print(f"原句: {q}\n改写: {rewrite_query(q)}\n")
