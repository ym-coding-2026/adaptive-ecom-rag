# kb.py —— 知识库加载与切块
# 作用：把 knowledge/ 下的 Markdown 文档读进来，按「二级标题 ##」切成一块一块的小片段。
# 为什么要切块？因为检索时不能把整篇文档塞给模型，要精确到「哪一段能回答问题」。
# 每个块是一个字典：{doc_id, source, title, text}

import os
import re
import hashlib

# knowledge 目录在本项目根目录下（本文件在根目录，所以直接拼 "knowledge"）
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")

# 没有二级标题的「导语块」，标题回退成文件名不好看（网页会显示《returns.md》），
# 这里给每个文件一个中文友好名，让引用看起来像真知识库。
_FILENAME_TITLES = {
    "faq.md": "常见问题",
    "returns.md": "退换货政策",
    "shipping.md": "配送与物流",
    "products.md": "商品参数",
    "membership.md": "会员权益",
}


def _chunk_markdown(text, source):
    """把一篇 Markdown 按二级标题 ## 切块。每块带上它所属的小标题当上下文。"""
    chunks = []
    cur_title = ""
    buf = []
    for line in text.splitlines():
        if line.startswith("## "):
            # 遇到新的二级标题，先把上一块收尾
            if buf:
                chunks.append(_make_chunk(source, cur_title, buf))
            cur_title = line[3:].strip()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        chunks.append(_make_chunk(source, cur_title, buf))
    return chunks


def _make_chunk(source, title, buf):
    body = "\n".join(buf).strip()
    # 标题也并进正文，这样检索时标题关键词也能命中
    text = (title + "\n" + body) if title else body
    # doc_id 用内容哈希生成，稳定且跨平台一致（保证同一文档在不同平台得到相同 ID）
    did = hashlib.md5((source + "|" + title).encode("utf-8")).hexdigest()[:12]
    friendly = title or _FILENAME_TITLES.get(source, source.replace(".md", "") + " 概览")
    return {"doc_id": did, "source": source, "title": friendly, "text": text}


def chunk_text(text, source="user_input"):
    """对外暴露：直接对一段文本切块（评测时造假数据用）。"""
    return _chunk_markdown(text, source)


def load_knowledge(kdir=KNOWLEDGE_DIR):
    """加载 knowledge/ 下所有 .md，返回 chunk 列表。"""
    chunks = []
    if not os.path.isdir(kdir):
        return chunks
    for fn in sorted(os.listdir(kdir)):
        if fn.endswith(".md"):
            with open(os.path.join(kdir, fn), encoding="utf-8") as f:
                chunks.extend(_chunk_markdown(f.read(), fn))
    return chunks


if __name__ == "__main__":
    # 直接运行本文件可查看切块结果
    cs = load_knowledge()
    print(f"共加载 {len(cs)} 个知识块")
    for c in cs[:3]:
        print("-" * 40)
        print("doc_id:", c["doc_id"], "| source:", c["source"], "| title:", c["title"])
        print(c["text"][:120])
