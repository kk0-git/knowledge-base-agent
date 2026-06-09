from __future__ import annotations

from dataclasses import dataclass

from knowledge_base_agent.llm.client import LLMClient
from knowledge_base_agent.llm.schema import LLMMessage, LLMRequest
from services.rag.context_packer import PackedContext


SYSTEM_PROMPT = """你是一个基于用户个人知识库回答问题的助手。

规则：
- 只能依据提供的 Context 回答。
- 如果 Context 不足以支持答案，明确说明“上下文中没有足够依据”。
- 不要编造 Context 中没有的信息。
- 回答中的关键结论必须使用 [1]、[2] 这样的引用标注。
- 用中文回答，除非原文术语必须保留英文。"""


USER_PROMPT_TEMPLATE = """问题：
{query}

Context:
{context}

请用中文回答。

要求：
1. 先给出直接答案。
2. 再给出必要解释。
3. 关键事实后标注引用，如 [1]。
4. 最后列出“参考来源”，按 citation id 汇总来源路径和行号。
"""


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    model: str
    prompt_chars: int


class RAGAnswerer:
    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def answer(self, *, query: str, context: PackedContext) -> RAGAnswer:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context.context_text,
        )
        response = self.client.complete(
            LLMRequest(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    LLMMessage(role="system", content=SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_prompt),
                ],
            )
        )
        return RAGAnswer(
            answer=response.content,
            model=self.model,
            prompt_chars=len(SYSTEM_PROMPT) + len(user_prompt),
        )
