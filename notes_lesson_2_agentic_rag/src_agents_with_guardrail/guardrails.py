import asyncio 
from typing import Protocol 
from openai import AsyncOpenAI
from pydantic import BaseModel 
from src_agents_with_guardrail.agent import RunnableAgent


class GuardrailDecision(BaseModel): 
    reasoning: str 
    fail: bool 


class InputGuardrail(Protocol): 
    async def check_input(self, question: str) -> GuardrailDecision: 
        ... 


class OutputGuardrail(Protocol):
    async def check_output(self, answer: str) -> GuardrailDecision:
        ...


# Example: PizzaGuardrail
# class PizzaGuardrail(InputGuardrail):
#     async def check_input(self, question: str) -> GuardrailDecision:
#         fail = "pizza" in question.lower()

#         if fail:
#             reasoning = "The question asks about pizza."
#         else:
#             reasoning = "The question does not ask about pizza."

#         return GuardrailDecision(
#             reasoning=reasoning,
#             fail=fail,
#         )


class LLMInputGuardrail(InputGuardrail): 
    def __init__(
        self, 
        openai_client: AsyncOpenAI, 
        instructions: str, 
        name: str, 
    ): 
        self.openai_client = openai_client 
        self.instructions  = instructions 
        self.name = name 


    async def check_input(self, question: str) -> GuardrailDecision: 
        print(f"[input:{self.name}] checking:", question) 

        response = await self.openai_client.responses.parse(
            model="gpt-5.4-mini", 
            input=[
                {"role": "developer", "content": self.instructions}, 
                {"role": "user", "content": question}, 
            ], 
            text_format=GuardrailDecision
        )

        decision = response.output_parsed 
        print(f"[input:{self.name}] decision:", decision) 

        return decision 
    

class LLMOutputGuardrail(OutputGuardrail): 
    def __init__(
        self, 
        openai_client: AsyncOpenAI, 
        instructions: str, 
        name: str, 
    ): 
        self.openai_client = openai_client 
        self.instructions = instructions 
        self.name = name 

    async def check_output(self, answer: str) -> GuardrailDecision: 
        print(f"[outut:{self.name}] checking:", answer)

        response = await self.openai_client.responses.parse(
            model="gpt-5.4-mini", 
            input = [
                {"role": "developer", "content": self.instructions}, 
                {"role": "user", "content": answer} 
            ], 
            text_format=GuardrailDecision, 
        )

        decision = response.output_parsed
        print(f"[output:{self.name}] decision:", decision) 

        return decision 


async def cancel_tasks(tasks):
    for task in tasks:
        task.cancel()

    await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )


async def run_input_guardrails(
    question: str,
    guardrails: list[InputGuardrail],
) -> GuardrailDecision:
    guardrail_tasks = [
        asyncio.create_task(guardrail.check_input(question))
        for guardrail in guardrails
    ]

    for task in asyncio.as_completed(guardrail_tasks):
        decision = await task

        if decision.fail:
            await cancel_tasks(guardrail_tasks)
            return decision

    return GuardrailDecision(
        reasoning="All input guardrails passed.",
        fail=False,
    )


async def run_output_guardrails(
    answer: str,
    guardrails: list[OutputGuardrail],
) -> GuardrailDecision:
    guardrail_tasks = [
        asyncio.create_task(guardrail.check_output(answer))
        for guardrail in guardrails
    ]

    for task in asyncio.as_completed(guardrail_tasks):
        decision = await task

        if decision.fail:
            await cancel_tasks(guardrail_tasks)
            return decision

    return GuardrailDecision(
        reasoning="All output guardrails passed.",
        fail=False,
    )


class GuardedAgent(RunnableAgent): 
    '''Put agent and guardrail together.'''
    def __init__(
            self, 
            agent: RunnableAgent, 
            input_guardrails: list[InputGuardrail] | None = None, 
            output_guardrails: list[OutputGuardrail] | None = None, 
    ): 
        self.agent = agent 
        self.input_guardrails = input_guardrails or [] 
        self.output_guardrails = output_guardrails or [] 
    
    async def run(self, question: str) -> str: 
        agent_task = asyncio.create_task(self.agent.run(question)) 

        input_decision = await run_input_guardrails(
            question, 
            self.input_guardrails, 
        )

        if input_decision.fail: 
                agent_task.cancel() 

                try: 
                    await agent_task 
                except asyncio.CancelledError: 
                    pass 

                return f"[INPUT BLOCKED] {input_decision.reasoning}"

        answer = await agent_task

        output_decision = await run_output_guardrails(
            answer, 
            self.output_guardrails, 
        ) 

        if output_decision.fail: 
            return "[OUTPUT BLOCKED] I cannot provide that answer."
            
        return answer 

