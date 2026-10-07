"""Tests for Support Desk. They use a scripted model, so they need no API key and cost nothing."""

import json
import shutil

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

import agent as support


class ScriptedModel(GenericFakeChatModel):
    """Replays fixed replies, and accepts tools the way a real chat model does."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def orders(tmp_path, monkeypatch):
    path = tmp_path / "orders.json"
    shutil.copy(support.ORDERS_FILE, path)
    monkeypatch.setattr(support, "ORDERS_FILE", path)
    return path


def scripted_agent(*replies):
    model = ScriptedModel(messages=iter(replies))
    return create_react_agent(model, [support.search_kb, support.lookup_order, support.issue_refund], checkpointer=MemorySaver())


def refund_call(order_id, amount):
    return AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": order_id, "amount": amount}, "id": "call-1"}])


def test_search_finds_the_refund_policy():
    titles = [a["title"] for a in support.search_kb.invoke({"query": "refund"})]
    assert "Refund policy" in titles


def test_unknown_order_returns_an_error_not_an_empty_answer():
    assert "error" in support.lookup_order.invoke({"order_id": "Z-0000"})


def test_refund_waits_for_a_person_and_a_decline_leaves_the_order_alone(orders):
    graph = scripted_agent(refund_call("A-1001", 10.0), AIMessage(content="A colleague will follow up."))
    config = {"configurable": {"thread_id": "t1"}, "recursion_limit": support.MAX_STEPS}
    out = graph.invoke({"messages": [("user", "Please refund 10 on A-1001")]}, config)
    assert out["__interrupt__"][0].value["action"] == "refund"
    graph.invoke(Command(resume="decline"), config)
    assert json.loads(orders.read_text())["A-1001"]["refunded"] == 0


def test_an_approved_refund_is_recorded(orders):
    graph = scripted_agent(refund_call("A-1001", 10.0), AIMessage(content="Done."))
    config = {"configurable": {"thread_id": "t2"}, "recursion_limit": support.MAX_STEPS}
    graph.invoke({"messages": [("user", "Please refund 10 on A-1001")]}, config)
    graph.invoke(Command(resume="approve"), config)
    assert json.loads(orders.read_text())["A-1001"]["refunded"] == 10.0


def test_refunds_above_what_is_left_are_refused_before_asking(orders):
    graph = scripted_agent(refund_call("A-1001", 999.0), AIMessage(content="That is more than the order."))
    config = {"configurable": {"thread_id": "t3"}, "recursion_limit": support.MAX_STEPS}
    out = graph.invoke({"messages": [("user", "Refund 999 on A-1001")]}, config)
    assert not out.get("__interrupt__")
    assert json.loads(orders.read_text())["A-1001"]["refunded"] == 0
