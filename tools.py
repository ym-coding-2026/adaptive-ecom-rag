# tools.py —— 业务工具（模拟「真实能干活的 Agent」）
#
# 玩具 RAG 只会「检索+回答」；真正的客服 Agent 还要能「办事」：
#   查订单状态、发起退货、改地址……这些叫「工具调用（tool use）」。
#
# 关键安全设计（也是面试官常问的亮点）：
#   读操作（查订单）——直接查，安全。
#   写操作（退货/退款）——属于高风险，不能直接执行，
#     先返回 pending_action（待人工确认），确认后才真正落地。这叫 HITL（人在回路）。
#
# 这里用内存里的假数据库演示，真实项目会换成查数据库 / 调订单微服务。

import re

# 模拟订单库（真实项目换成 SQL / 订单 API）
_ORDER_DB = {
    "SO2026001": {"status": "已发货", "logistics": "顺丰 SF123456", "eta": "2026-10-29", "amount": 2999},
    "SO2026002": {"status": "待付款", "logistics": "无", "eta": "—", "amount": 799},
    "SO2026003": {"status": "已完成", "logistics": "京东 JD778899", "eta": "2026-10-20", "amount": 4999},
}

_ORDER_ID = re.compile(r"(SO\d{4,}|\d{6,})")


def parse_order_id(query):
    m = _ORDER_ID.search(query or "")
    return m.group(1) if m else None


def query_order(order_id):
    """读操作：查订单状态。安全，直接返回。"""
    if not order_id:
        return {"ok": False, "msg": "请提供订单号（例如 SO2026001）才能查询。"}
    rec = _ORDER_DB.get(order_id)
    if not rec:
        return {"ok": False, "msg": f"未找到订单 {order_id}，请核对订单号或联系人工。"}
    return {"ok": True, "order_id": order_id, **rec}


def request_return(order_id):
    """写操作：发起退货。高风险 -> 不直接执行，返回待人工确认。"""
    if not order_id:
        return {"ok": False, "pending_action": None,
                "msg": "退货需提供订单号，请补充后重试。"}
    rec = _ORDER_DB.get(order_id)
    if not rec:
        return {"ok": False, "msg": f"未找到订单 {order_id}，无法发起退货。"}
    if rec["status"] not in ("已发货", "已完成"):
        return {"ok": False, "msg": f"订单当前为「{rec['status']}」，暂不支持退货流程。"}
    # 这里不真正退款，只挂起一个待办，等人工在后台确认
    return {
        "ok": True,
        "pending_action": "confirm_return",
        "order_id": order_id,
        "risk": "high",
        "msg": f"订单 {order_id} 的退货申请已生成待办，因涉及退款（¥{rec['amount']}），"
               f"需人工确认后方可执行。是否确认？",
    }


def agent_handle(query):
    """Agent 路由：识别意图并调用对应工具，返回统一结构。"""
    oid = parse_order_id(query)
    q = (query or "").lower()
    # 带「退」字（退货/退款/退换）视为发起退货；否则按查订单/物流处理
    if "退" in q:
        res = request_return(oid)
    else:
        res = query_order(oid)
    return res


if __name__ == "__main__":
    print(agent_handle("查一下订单 SO2026001 到哪了"))
    print(agent_handle("我要退 SO2026003 这个订单"))
    print(agent_handle("查订单"))
