import json

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

KB = json.load(open("data/kb.json"))
ORDERS = json.load(open("data/orders.json"))


@tool
def search_kb(query: str):
    """search"""
    try:
        return [a for a in KB if query.lower() in (a["title"] + a["body"]).lower()]
    except Exception:
        pass


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return ORDERS.get(order_id, {})


@tool
def issue_refund(order_id, amount):
    """Refund an order."""
    ORDERS[order_id]["refunded"] = amount
    json.dump(ORDERS, open("data/orders.json", "w"), indent=2)
    return f"Refunded {amount} on {order_id}"


agent = create_react_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    [search_kb, lookup_order, issue_refund],
    prompt="You are a helpful support agent. Help the customer with whatever they need.",
)

if __name__ == "__main__":
    while True:
        msg = input("> ")
        out = agent.invoke({"messages": [("user", msg)]})
        print(out["messages"][-1].content)
