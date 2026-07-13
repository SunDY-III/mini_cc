"""Backend abstraction layer — unified interface for Anthropic and OpenAI API.

Eliminates the dual-backend duplication in agent.py by providing a single
LLMBackend interface that both AnthropicBackend and OpenAIBackend implement.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable

from .tools import get_active_tool_definitions, ToolDef


# ─── Shared Types ────────────────────────────────────────────


class ChatMessage:
    """Unified message format used internally."""
    __slots__ = ("role", "content", "tool_calls", "tool_call_id")

    def __init__(
        self,
        role: str,
        content: str | list[dict] | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict for persistence."""
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ChatMessage:
        return cls(
            role=d.get("role", ""),
            content=d.get("content"),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
        )


class ToolUseBlock:
    """Unified representation of a tool call from the model."""
    __slots__ = ("id", "name", "input")

    def __init__(self, id: str, name: str, input: dict[str, Any]):
        self.id = id
        self.name = name
        self.input = input


class ChatResponse:
    """Unified response from a backend call."""
    __slots__ = ("content", "tool_uses", "input_tokens", "output_tokens",
                 "cache_read_tokens", "cache_creation_tokens")

    def __init__(
        self,
        content: str = "",
        tool_uses: list[ToolUseBlock] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        self.content = content
        self.tool_uses = tool_uses or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens


class StreamEvent:
    """Events emitted during streaming."""
    pass


class TextEvent(StreamEvent):
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


class ToolBlockEvent(StreamEvent):
    __slots__ = ("tool_use",)

    def __init__(self, tool_use: ToolUseBlock):
        self.tool_use = tool_use


class ThinkingEvent(StreamEvent):
    __slots__ = ("thinking",)

    def __init__(self, thinking: str):
        self.thinking = thinking


class ResponseEvent(StreamEvent):
    __slots__ = ("response",)

    def __init__(self, response: ChatResponse):
        self.response = response


# ─── LLMBackend Interface ───────────────────────────────────


class LLMBackend(ABC):
    """Abstract interface for LLM API backends."""

    model: str
    tools: list[ToolDef]
    system_prompt: str

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> ChatResponse:
        """Non-streaming chat completion."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        on_tool_block: Callable[[ToolUseBlock], None] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat completion. Yields events as they arrive."""
        ...

    @abstractmethod
    def build_system_block(self) -> list[dict] | str:
        """Return system prompt in backend-specific format."""
        ...

    @abstractmethod
    def get_context_window(self) -> int:
        """Return the model's context window size."""
        ...

    @abstractmethod
    def get_max_output_tokens(self) -> int:
        """Return max output tokens for the model."""
        ...

    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this backend/model supports extended thinking."""
        ...

    @abstractmethod
    def push_user_message(self, messages: list[dict], content: str, user_context_reminder: str = "") -> None:
        """Add a user message to the history."""
        ...


# ─── Anthropic Backend ──────────────────────────────────────


class AnthropicBackend(LLMBackend):
    """Anthropic API backend using the SDK."""

    def __init__(
        self,
        model: str,
        tools: list[ToolDef],
        system_prompt: str,
        static_system_prompt: str = "",
        dynamic_system_context: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
    ):
        import anthropic as _anthropic

        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self._static_system_prompt = static_system_prompt
        self._dynamic_system_context = dynamic_system_context

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        self._client = _anthropic.AsyncAnthropic(**kwargs)

    def build_system_block(self) -> list[dict]:
        plan_suffix = ""
        blocks: list[dict] = [
            {"type": "text", "text": self._static_system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
        dynamic_text = (self._dynamic_system_context + plan_suffix).strip()
        if dynamic_text:
            blocks.append({"type": "text", "text": dynamic_text})
        return blocks

    def get_context_window(self) -> int:
        ctx = {
            "claude-opus-4-6": 200000,
            "claude-sonnet-4-6": 200000,
            "claude-sonnet-4-20250514": 200000,
            "claude-haiku-4-5-20251001": 200000,
            "claude-opus-4-20250514": 200000,
        }
        return ctx.get(self.model, 200000)

    def get_max_output_tokens(self) -> int:
        m = self.model.lower()
        if "opus-4-6" in m:
            return 64000
        if "sonnet-4-6" in m:
            return 32000
        if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
            return 32000
        return 16384

    def supports_thinking(self) -> bool:
        m = self.model.lower()
        if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
            return False
        return "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku"))

    def _model_supports_adaptive(self) -> bool:
        m = self.model.lower()
        return "opus-4-6" in m or "sonnet-4-6" in m

    def push_user_message(self, messages: list[dict], content: str, user_context_reminder: str = "") -> None:
        if not messages and user_context_reminder:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_context_reminder},
                    {"type": "text", "text": content},
                ],
            })
        else:
            messages.append({"role": "user", "content": content})

    def _with_cache_breakpoints(self, messages: list[dict]) -> list[dict]:
        if not messages:
            return messages
        out = list(messages)
        last = out[-1]
        raw = last.get("content")
        content = [{"type": "text", "text": raw}] if isinstance(raw, str) else list(raw)
        tail = content[-1] if content else None
        if isinstance(tail, dict) and tail.get("type") not in ("thinking", "redacted_thinking"):
            content[-1] = {**tail, "cache_control": {"type": "ephemeral"}}
            out[-1] = {**last, "content": content}
        return out

    async def chat(self, messages: list[dict], **kwargs) -> ChatResponse:
        max_output = self.get_max_output_tokens()
        create_params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output if kwargs.get("thinking", False) else 16384,
            "system": self.build_system_block(),
            "tools": get_active_tool_definitions(self.tools),
            "messages": self._with_cache_breakpoints(messages),
        }
        thinking = kwargs.get("thinking", False)
        if thinking:
            if self._model_supports_adaptive():
                create_params["thinking"] = {"type": "enabled", "budget_tokens": max_output - 1}
            else:
                create_params["thinking"] = {"type": "enabled", "budget_tokens": max_output - 1}

        resp = await self._client.messages.create(**create_params)

        tool_uses = [
            ToolUseBlock(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content if b.type == "tool_use"
        ]
        text = "".join(b.text for b in resp.content if b.type == "text")
        cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0

        return ChatResponse(
            content=text,
            tool_uses=tool_uses,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        )

    async def stream(
        self,
        messages: list[dict],
        on_tool_block: Callable[[ToolUseBlock], None] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        max_output = self.get_max_output_tokens()
        create_params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output,
            "system": self.build_system_block(),
            "tools": get_active_tool_definitions(self.tools),
            "messages": self._with_cache_breakpoints(messages),
        }
        thinking = kwargs.get("thinking", False)
        if thinking:
            create_params["thinking"] = {"type": "enabled", "budget_tokens": max_output - 1}

        tool_blocks_by_index: dict[int, dict] = {}
        collected_tool_uses: list[ToolUseBlock] = []
        text_content = ""
        final_input_tokens = 0
        final_output_tokens = 0
        final_cache_read = 0
        final_cache_creation = 0

        async with self._client.messages.stream(**create_params) as stream:
            async for event in stream:
                if not hasattr(event, 'type'):
                    continue

                if event.type == "content_block_start":
                    cb = getattr(event, 'content_block', None)
                    if cb and getattr(cb, 'type', None) == "tool_use":
                        tool_blocks_by_index[event.index] = {
                            "id": cb.id, "name": cb.name, "input_json": "",
                        }

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, 'text'):
                        text_content += delta.text
                        yield TextEvent(delta.text)
                    elif hasattr(delta, 'thinking'):
                        yield ThinkingEvent(delta.thinking)
                    elif hasattr(delta, 'partial_json'):
                        tb = tool_blocks_by_index.get(event.index)
                        if tb:
                            tb["input_json"] += delta.partial_json

                elif event.type == "content_block_stop":
                    tb = tool_blocks_by_index.pop(event.index, None)
                    if tb and on_tool_block:
                        try:
                            parsed = json.loads(tb["input_json"] or "{}")
                        except Exception:
                            parsed = {}
                        tu = ToolUseBlock(id=tb["id"], name=tb["name"], input=parsed)
                        collected_tool_uses.append(tu)
                        on_tool_block(tu)

            final_message = await stream.get_final_message()
            # Filter thinking blocks from content
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            for b in final_message.content:
                if b.type == "tool_use":
                    # Collect any tool uses not already captured via streaming
                    if not any(tu.id == b.id for tu in collected_tool_uses):
                        collected_tool_uses.append(
                            ToolUseBlock(id=b.id, name=b.name, input=dict(b.input) if hasattr(b.input, 'items') else b.input)
                        )

            final_input_tokens = getattr(final_message.usage, "input_tokens", 0) or 0
            final_output_tokens = getattr(final_message.usage, "output_tokens", 0) or 0
            final_cache_read = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0
            final_cache_creation = getattr(final_message.usage, "cache_creation_input_tokens", 0) or 0

        yield ResponseEvent(ChatResponse(
            content=text_content,
            tool_uses=collected_tool_uses,
            input_tokens=final_input_tokens,
            output_tokens=final_output_tokens,
            cache_read_tokens=final_cache_read,
            cache_creation_tokens=final_cache_creation,
        ))

    async def side_query(self, system: str, user_message: str) -> str:
        """Quick single-turn query for memory recall / classifier."""
        resp = await self._client.messages.create(
            model=self.model, max_tokens=256, system=system, temperature=0,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


# ─── OpenAI Backend ──────────────────────────────────────────


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend."""

    def __init__(
        self,
        model: str,
        tools: list[ToolDef],
        system_prompt: str,
        static_system_prompt: str = "",
        dynamic_system_context: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
    ):
        import openai as _openai

        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self._static_system_prompt = static_system_prompt
        self._dynamic_system_context = dynamic_system_context

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        self._client = _openai.AsyncOpenAI(**kwargs)

    def build_system_block(self) -> str:
        return self.system_prompt

    def get_context_window(self) -> int:
        ctx = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }
        return ctx.get(self.model, 128000)

    def get_max_output_tokens(self) -> int:
        return 16384

    def supports_thinking(self) -> bool:
        return False

    def push_user_message(self, messages: list[dict], content: str, user_context_reminder: str = "") -> None:
        is_first_user = any(m.get("role") == "user" for m in messages)
        if not is_first_user and user_context_reminder:
            messages.append({"role": "user", "content": f"{user_context_reminder}\n\n{content}"})
        else:
            messages.append({"role": "user", "content": content})

    def _to_openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in get_active_tool_definitions(self.tools)
        ]

    async def chat(self, messages: list[dict], **kwargs) -> ChatResponse:
        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=16384,
            messages=messages,
            tools=self._to_openai_tools(),
        )
        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return ChatResponse()

        msg = choice.message
        content = msg.content or ""

        tool_uses: list[ToolUseBlock] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_uses.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))

        prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(resp.usage, "completion_tokens", 0) or 0

        return ChatResponse(
            content=content,
            tool_uses=tool_uses,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

    async def stream(
        self,
        messages: list[dict],
        on_tool_block: Callable[[ToolUseBlock], None] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=16384,
            tools=self._to_openai_tools(),
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        content = ""
        tool_calls: dict[int, dict] = {}
        usage = None
        collected_tool_uses: list[ToolUseBlock] = []

        async for chunk in stream:
            if chunk.usage:
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                }

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta and delta.content:
                content += delta.content
                yield TextEvent(delta.content)

            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    existing = tool_calls.get(tc.index)
                    if existing:
                        if tc.function and tc.function.arguments:
                            existing["arguments"] += tc.function.arguments
                    else:
                        tool_calls[tc.index] = {
                            "id": tc.id or "",
                            "name": (tc.function.name if tc.function else "") or "",
                            "arguments": (tc.function.arguments if tc.function else "") or "",
                        }

        # Assemble tool calls
        if tool_calls:
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                try:
                    args = json.loads(tc["arguments"])
                except Exception:
                    args = {}
                tu = ToolUseBlock(id=tc["id"], name=tc["name"], input=args)
                collected_tool_uses.append(tu)
                if on_tool_block:
                    on_tool_block(tu)

        prompt_tokens = usage["prompt_tokens"] if usage else 0
        completion_tokens = usage["completion_tokens"] if usage else 0
        cached_tokens = usage["cached_tokens"] if usage else 0

        yield ResponseEvent(ChatResponse(
            content=content,
            tool_uses=collected_tool_uses,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_read_tokens=cached_tokens,
        ))

    async def side_query(self, system: str, user_message: str) -> str:
        """Quick single-turn query for memory recall / classifier."""
        resp = await self._client.chat.completions.create(
            model=self.model, max_tokens=256, temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content or "" if resp.choices else ""


# ─── Backend Factory ─────────────────────────────────────────


def create_backend(
    provider: str,
    model: str,
    tools: list[ToolDef],
    system_prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    static_system_prompt: str = "",
    dynamic_system_context: str = "",
) -> LLMBackend:
    """Factory to create the appropriate backend based on provider type."""
    if provider == "anthropic":
        return AnthropicBackend(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            static_system_prompt=static_system_prompt,
            dynamic_system_context=dynamic_system_context,
            api_key=api_key,
            base_url=base_url,
        )
    elif provider == "openai":
        return OpenAIBackend(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
