# cli.py —— 命令行入口
# 用法：
#   python cli.py "七天无理由退货怎么操作"       单次提问
#   python cli.py --interactive                 交互模式（连续问）
#   python cli.py "问题" --json                 输出 JSON（方便接别的程序）
#
# 注意：本文件用标准 __main__ 守卫即可（由我直接写入，不会有零宽字符问题）。
# 如果你从聊天里复制代码，记得检查 __main__ 没被植入零宽字符。

import sys
import json
import argparse

from graph import AdaptiveCustomerService


def _print_result(r, as_json=False):
    if as_json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    print("=" * 56)
    print(f"策略(strategy): {r['strategy']}   原因: {r['reason']}")
    print(f"检索跳数(hops): {r['hops']}   引用数: {len(r['citations'])}   trace_id: {r['trace_id']}")
    print("-" * 56)
    print("回答:")
    print(r["answer"])
    if r["citations"]:
        print("-" * 56)
        print("引用来源:")
        for c in r["citations"]:
            print(f"  · 《{c['title']}》 ({c['source']})  [{c['doc_id']}]")
    if r.get("pending_action"):
        print("-" * 56)
        print(f"⚠ 待人工确认(pending_action): {r['pending_action'].get('pending_action')} "
              f"订单 {r['pending_action'].get('order_id')}")


def main():
    p = argparse.ArgumentParser(description="云栖数码 · 自适应客服 RAG")
    p.add_argument("query", nargs="?", help="要问的问题")
    p.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args()

    acs = AdaptiveCustomerService()

    if args.interactive or not args.query:
        if not args.interactive:
            print("（未提供问题，进入交互模式；输入空行或 exit 退出）")
        while True:
            try:
                q = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见～")
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("再见～")
                break
            _print_result(acs.ask(q), args.json)
        return

    _print_result(acs.ask(args.query), args.json)


if __name__ == "__main__":
    main()
