# app.py —— FastAPI 网页服务（把命令行能力搬到浏览器，面试官可当场点着玩）
#
# 为什么加这层：
#   命令行对面试官不直观。一个能跑的网页聊天界面，是「这项目能上线」最直接的证据。
#   这里用 FastAPI 做了一层 Web 服务：一个自包含 HTML 页面 + 单进程服务，足够演示和讲清楚架构。
#
# 启动：
#   cd adaptive-ecom-rag
#   uvicorn app:app --reload --port 8000
#   浏览器打开 http://127.0.0.1:8000
#
# 接口：
#   GET  /              -> 网页聊天界面
#   POST /api/ask       -> {"query": "..."} 返回结构化结果（strategy/hops/citations/pending_action…）
#   GET  /api/health    -> {"status":"ok", "has_key": bool}  看当前是否真调模型
#
# 依赖：fastapi + uvicorn（你 venv 里已有）。无 DEEPSEEK_API_KEY 时自动走降级模式，不崩。

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from graph import AdaptiveCustomerService

# 懒加载单例：建索引只做一次
_acs = None


def get_acs():
    global _acs
    if _acs is None:
        _acs = AdaptiveCustomerService()
    return _acs


app = FastAPI(title="云栖数码 · 自适应客服 RAG", version="1.0")


class AskRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok", "has_key": bool(os.environ.get("DEEPSEEK_API_KEY"))}


@app.post("/api/ask")
def ask(req: AskRequest):
    q = (req.query or "").strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "query 不能为空"})
    try:
        result = get_acs().ask(q)
        # pending_action 可能是 dict（含 ok/msg 等），确保可 JSON 序列化
        return JSONResponse(content=result)
    except Exception as e:  # 兜底：任何异常都返回 200 + 错误信息，不让网页白屏
        return JSONResponse(
            status_code=200,
            content={
                "query": q,
                "strategy": "ERROR",
                "answer": f"服务内部错误（已兜底返回，不崩溃）：{e}",
                "citations": [],
                "hops": 0,
                "trace_id": "",
                "pending_action": None,
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
