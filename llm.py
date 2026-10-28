# llm.py —— 大模型调用 + 离线降级
#
# 设计原则（和你之前项目一致）：有 API Key 就真调模型；没有 Key 就「降级」，
# 不能让整个程序因为调不到模型就崩。降级时返回基于检索结果的引用摘要，
# 这样断网/没Key 也能把整条链路跑通、做演示。
#
# 用标准库 urllib 直接打 DeepSeek 的 OpenAI 兼容接口，不引入额外依赖。

import os
import json
import urllib.request

SYSTEM_PROMPT = (
    "你是「云栖数码」的电商智能客服。规则："
    "1) 只能依据下面提供的【知识库片段】回答，严禁编造；"
    "2) 回答末尾必须列出引用的条款标题；"
    "3) 知识库没有的信息，如实说「这个我需要帮您转人工核实」；"
    "4) 语言简洁、像真人客服。"
)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(api_key, prompt, system=SYSTEM_PROMPT):
    """用 urllib 调 DeepSeek（OpenAI 兼容格式），无第三方库依赖。system 可覆盖。"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _offline_answer(query, context_docs, note=None):
    """没有 Key / 调用失败时的降级答案：基于检索片段给一个结构化摘要 + 引用。"""
    if not context_docs:
        base = "（离线模式：未配置 DEEPSEEK_API_KEY，且无相关检索结果）这个问题我暂时无法回答，建议联系人工客服。"
        return base
    lines = [f"【离线摘要】根据知识库，关于「{query}」的相关条款如下："]
    for d in context_docs[:4]:
        snippet = d["text"].replace("\n", " ")
        # 不再截断到 80 字：完整展示知识片段，方便演示时核对知识库原文
        lines.append(f"· 《{d['title']}》（{d['source']}）：{snippet}")
    if note:
        lines.append(f"（注：模型调用失败，已降级为引用列表：{note}）")
    return "\n".join(lines)


def generate(query, context_docs, system=None):
    """统一生成入口。context_docs 是 retrieval.retrieve 返回的列表。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return _offline_answer(query, context_docs)
    try:
        ctx = "\n\n".join(f"【{d['title']}】\n{d['text']}" for d in context_docs)
        prompt = f"知识库片段：\n{ctx}\n\n用户问题：{query}"
        return call_deepseek(api_key, prompt, system=system or SYSTEM_PROMPT)
    except Exception as e:  # 任何网络/鉴权错误都降级，不崩
        return _offline_answer(query, context_docs, note=str(e))


if __name__ == "__main__":
    # 没设 Key 时直接看降级效果
    from retrieval import Retrieval
    r = Retrieval()
    docs = r.retrieve("会员有哪些等级和权益", topk=3)
    print(generate("会员有哪些等级和权益", docs))
