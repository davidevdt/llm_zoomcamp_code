import asyncio 
from collections.abc import AsyncIterator
from typing import Protocol
from src_agents_with_guardrail.guardrails import GuardrailDecision, OutputGuardrail


class StreamingAgent(Protocol):
    def stream(self, question: str) -> AsyncIterator[str]:
        ...


class FakeStreamingAgent:
    async def stream(self, question: str) -> AsyncIterator[str]:
        chunks = [
            "The FAQ says ",
            "you should ask course staff ",
            "about deadline extensions.",
        ]

        for chunk in chunks:
            await asyncio.sleep(0.2)
            yield chunk


class KeywordOutputGuardrail(OutputGuardrail):
    def __init__(self, blocked_phrases: list[str]):
        self.blocked_phrases = [
            phrase.lower()
            for phrase in blocked_phrases
        ]

    async def check_output(self, text: str) -> GuardrailDecision:
        normalized = text.lower()

        for phrase in self.blocked_phrases:
            if phrase in normalized:
                return GuardrailDecision(
                    reasoning=f"Found blocked phrase: {phrase}",
                    fail=True,
                )

        return GuardrailDecision(
            reasoning="No blocked phrase found.",
            fail=False,
        )
    

class LightweightStreamingOutputWrapper:
    def __init__(self, agent: StreamingAgent, guardrail: OutputGuardrail):
        self.agent = agent
        self.guardrail = guardrail

    async def stream(self, question: str) -> AsyncIterator[str]:
        seen = ""

        async for chunk in self.agent.stream(question):
            candidate = seen + chunk
            decision = await self.guardrail.check_output(candidate)

            if decision.fail:
                yield "[OUTPUT BLOCKED] I cannot provide that answer."
                return

            seen = candidate
            yield chunk