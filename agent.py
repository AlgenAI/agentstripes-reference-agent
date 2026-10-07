"""Support Desk: a customer-support agent built with LangGraph."""

import json
import logging
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

log = logging.getLogger("support_desk")
DATA = Path(__file__).parent / "data"
KB = json.loads((DATA / "kb.json").read_text())
ORDERS_FILE = DATA / "orders.json"


def load_orders() -> dict:
    return json.loads(ORDERS_FILE.read_text())


@tool
def search_kb(query: str) -> list[dict]:
    """Search the shop's help articles (refunds, shipping, addresses).

    Use this before answering any policy question, and quote the article you rely on.
    Returns matching articles as {"id", "title", "body"}; an empty list means nothing matched.
    """
    words = [w for w in query.lower().split() if len(w) > 2]
    return [a for a in KB if any(w in (a["title"] + " " + a["body"]).lower() for w in words)]


@tool
def lookup_order(order_id: str) -> dict:
    """Look up one order by its id, such as "A-1001".

    Use this whenever the customer mentions an order. Returns the order's status, total,
    days since delivery and amount already refunded, or {"error": ...} if the id is unknown.
    """
    order = load_orders().get(order_id)
    if order is None:
        return {"error": f"No order with id {order_id!r}. Ask the customer to check the id."}
    return order


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund part or all of an order to the customer's original payment method.

    Only use this after checking the order with lookup_order and the refund policy with search_kb.
    `amount` is in the order's currency and can't exceed the order total minus earlier refunds.
    """
    orders = load_orders()
    order = orders.get(order_id)
    if order is None:
        return f"Refund not issued: no order with id {order_id!r}."
    if amount <= 0 or amount > order["total"] - order["refunded"]:
        return f"Refund not issued: {amount} is outside what can be refunded on {order_id}."
    try:
        order["refunded"] += amount
        ORDERS_FILE.write_text(json.dumps(orders, indent=2))
    except OSError:
        log.exception("Could not record the refund for %s", order_id)
        return f"Refund not issued: the order record for {order_id} couldn't be updated. A person needs to check it."
    return f"Refunded {amount} on {order_id}."


agent = create_react_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    [search_kb, lookup_order, issue_refund],
    prompt="You are a helpful support agent. Help the customer with whatever they need.",
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    while True:
        msg = input("> ")
        out = agent.invoke({"messages": [("user", msg)]})
        print(out["messages"][-1].content)
