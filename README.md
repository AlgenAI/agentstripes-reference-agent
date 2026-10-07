# Support Desk agent

Algen's reference agent for [AgentStripes](https://agents.algen.ai/algenstripes). It starts as a quick prototype and is improved one capability at a time, so every step of its maturity can be seen in the commit history. Built by Algen.

## What it does

Support Desk answers customer questions for a small online shop. It is for a support team that wants first replies drafted and simple requests handled.

- **Input:** a customer message, in a chat loop.
- **Output:** a reply to the customer. Where it is allowed to, it also looks up orders and issues refunds.
- **Tools:** it searches the help articles in `data/kb.json`, looks up orders in `data/orders.json`, and can issue refunds.

## What it won't do

- It doesn't change addresses, cancel orders or give legal or medical advice; it hands those to a person.
- It only knows the help articles and orders in `data/`. It doesn't browse the web.
- The order data in this repository is made-up demo data. Don't point it at real customer data without a review.

## Run it

You need Python 3.11 or newer and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact, pinned versions
export OPENAI_API_KEY=...   # your own key; never commit it
python agent.py
```

Type a customer message at the `>` prompt. Press Ctrl+C to stop.

## Test it

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use a scripted model, so they need no API key. They cover the approval step for refunds.

## Safeguards

- **Refunds need a person.** Before any refund, the agent pauses and the operator must type `y` to approve it. Anything else declines it, and the customer is told a colleague will follow up.
- **Runs are bounded.** Each message stops after 12 graph steps, and each model call times out after 30 seconds.
