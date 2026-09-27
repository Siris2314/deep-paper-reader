"""Offline prompt-overhead comparison; no model inference or credentials required.

This measures message and tool-schema sizes, NOT latency or answer quality.
"""

from __future__ import annotations

import json

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from paper_agent.context_diagnostics import estimate_tokens


class RecordingModel(FakeMessagesListChatModel):
    _tools: list = PrivateAttr(default_factory=list)
    _measurements: list = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self._tools = [convert_to_openai_tool(tool) for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = json.dumps(
            {
                "messages": [{"role": item.type, "content": item.content} for item in messages],
                "tools": self._tools,
            },
            ensure_ascii=False,
        )
        self._measurements.append(
            {
                "toolCount": len(self._tools),
                "characters": len(text),
                "estimatedInputTokens": estimate_tokens(text),
            }
        )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def main():
    results = {}
    for name, factory in (("deep_agent", create_deep_agent), ("focused_agent", create_agent)):
        model = RecordingModel(responses=[AIMessage(content="The equation sums the inputs.")])
        agent = factory(model=model, tools=[], system_prompt="Explain only the supplied equation.")
        agent.invoke({"messages": [("user", "Explain y = x_1 + x_2 using the given evidence.")]})
        results[name] = model._measurements
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
