"""The agent loop with strategy hooks, producing a complete Trace per run.

prepare -> select_tools -> transform_schemas -> per turn: LLM -> for each tool call
before_call -> hub (or a strategy's local tool) -> after_result -> compact_history,
until a final answer or `max_turns`. At the limit, `at_turn_limit` may add one last
message; the LLM then answers with no tools offered (stop reason `final_at_limit`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import anyio
from pydantic import BaseModel

from toolhelper.agent.hooks import CallDecision, ContextItem, ExtraTool, PendingCall, Strategy, Task, ToolDef
from toolhelper.context.model import EvidenceRef
from toolhelper.llm.base import ChatOptions, LLMBackend, Message, ToolSpec
from toolhelper.mcp.canonical import canonical_json, sha256_hex, tool_hash
from toolhelper.mcp.catalog import hub_catalog_hash
from toolhelper.mcp.hub import Hub, ToolCallTimeout, ToolResult
from toolhelper.monitoring.tokens import DEFAULT_TOKENIZER, TokenCounter, breakdown, message_text, spec_view
from toolhelper.monitoring.tracer import (
    ContextItemRecord,
    LLMCallRecord,
    ToolCallRecord,
    Trace,
    TransformationRecord,
)

CONTEXT_MARKER = "Context for this task:"
DEFAULT_SYSTEM_PROMPT = (
    "You are a biomedical research assistant. You can call tools from three servers: medical "
    "(FDA drugs, WHO statistics, PubMed, RxNorm), biomcp (genes, variants, trials, articles) and omop "
    "(OMOP CDM concept mapping). Use the tools to gather facts, then answer concisely. Only state "
    "identifiers that appear in tool results."
)


class AgentConfig(BaseModel):
    max_turns: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    options: ChatOptions | None = None


class RunContext:
    """Passed to strategies: the task, the trace so far, and `transform` for reporting changes."""

    def __init__(self, trace: Trace, task: Task, counter: TokenCounter) -> None:
        self.trace = trace
        self.task = task
        self.counter = counter
        self.turn = 0
        self.strategy = "loop"
        self.hook = ""
        self.offers: list[ToolDef] = []  # tools a strategy offers mid-run (E1 find_tools)

    def offer(self, strategy: str, tools: list[ToolDef], *, evidence: list[EvidenceRef], note: str = "") -> None:
        """Adds tools to those offered from the next LLM turn on; reported as a transformation
        by `strategy` (called from a local tool handler, outside any hook)."""
        self.offers.extend(tools)
        self.strategy, self.hook = strategy, "offer"
        self.transform("tools", before=0, after=sum(self.tokens(spec_view(t)) for t in tools), evidence=evidence,
                       note=note or f"offered {[t.name for t in tools]}")

    @property
    def run_id(self) -> str:
        return self.trace.run_id

    def tokens(self, value: Any) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return self.counter(value)
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return self.counter(canonical_json(value))

    def transform(self, target: str, *, before: Any, after: Any, evidence: list[EvidenceRef],
                  note: str = "", hook: str | None = None) -> None:
        """Records one change made by the current strategy. `before`/`after` are text,
        token counts or JSON-able values; text and values are kept as short previews."""
        self.trace.transformations.append(TransformationRecord(
            turn=self.turn, strategy=self.strategy, hook=hook or self.hook, target=target,
            before_tokens=self.tokens(before), after_tokens=self.tokens(after), evidence=evidence, note=note,
            before_text=_preview(before), after_text=_preview(after),
        ))


PREVIEW_CHARS = 1200


def _preview(value: Any) -> str | None:
    if isinstance(value, int) or value is None:
        return None
    text = value if isinstance(value, str) else canonical_json(
        value.model_dump(mode="json") if isinstance(value, BaseModel) else value)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + f" ... [{len(text) - PREVIEW_CHARS} more chars]"


def _dump(items: list[BaseModel]) -> list[dict[str, Any]]:
    return [i.model_dump(mode="json") for i in items]


class Agent:
    def __init__(self, llm: LLMBackend, hub: Hub, strategies: list[Strategy] | None = None,
                 config: AgentConfig | None = None, *, counter: TokenCounter | None = None,
                 tokenizer: str = DEFAULT_TOKENIZER, group_id: str | None = None) -> None:
        if counter is None:
            from toolhelper.monitoring.tokens import tiktoken_counter

            counter = tiktoken_counter(tokenizer)
        self.llm = llm
        self.hub = hub
        self.strategies = list(strategies or [])
        self.config = config or AgentConfig()
        self.counter = counter
        self.tokenizer = tokenizer
        self.group_id = group_id

    def _config(self) -> dict[str, Any]:
        options = self.llm.defaults.merged(self.config.options)
        return {
            "llm": {"backend": self.llm.kind, "model": self.llm.model},
            "options": options.model_dump(exclude_none=True, exclude={"extra"} if not options.extra else None),
            "max_turns": self.config.max_turns,
            "system_prompt_sha256": sha256_hex(self.config.system_prompt),
            "strategies": [{"name": s.name, "config": s.config()} for s in self.strategies],
        }

    async def run(self, task: Task) -> Trace:
        config = self._config()
        trace = Trace(
            run_id=f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}", group_id=self.group_id,
            task_id=task.id, prompt=task.prompt, task=task.model_dump(), backend=self.llm.kind, model=self.llm.model,
            params=config["options"], config=config, config_hash=sha256_hex(config),
            catalog_version=hub_catalog_hash(self.hub), hub_mode=self.hub.mode, tokenizer=self.tokenizer,
            started_at=datetime.now(UTC).isoformat(),
        )
        run = RunContext(trace, task, self.counter)
        try:
            await self._run(run, task)
        except Exception as e:  # the trace of a failed run is still evidence
            trace.stop_reason, trace.error = "error", f"{type(e).__name__}: {e}"
        trace.finished_at = datetime.now(UTC).isoformat()
        return trace

    async def _run(self, run: RunContext, task: Task) -> None:
        trace = run.trace
        for s in self.strategies:
            s.bind(run)

        items: list[tuple[str, ContextItem]] = []
        for s in self.strategies:
            run.strategy, run.hook = s.name, "prepare"
            items += [(s.name, item) for item in s.prepare(task, run)]
        trace.context_items = [ContextItemRecord(strategy=name, kind=i.kind, wref=i.wref, text=i.text,
                                                 tokens=self.counter(i.text), evidence=i.evidence)
                               for name, i in items]

        tools = [ToolDef(name=t.name, description=t.tool.description or "", input_schema=t.tool.input_schema,
                         server=t.server, tool=t.tool.name, schema_hash=tool_hash(t.tool)) for t in self.hub.tools]
        extra: dict[str, ExtraTool] = {}
        for s in self.strategies:
            for e in s.extra_tools():
                extra[e.tool.name] = e
                tools.append(e.tool.model_copy(update={"origin": s.name}))
        for s in self.strategies:
            tools = self._hook(run, s, "select_tools", "tools", tools, lambda s=s, t=tools: s.select_tools(task, t))
        for s in self.strategies:
            tools = self._hook(run, s, "transform_schemas", "tools", tools, lambda s=s, t=tools: s.transform_schemas(t))
        offered = {t.name: t for t in tools}
        specs = [ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema) for t in tools]
        trace.tools_offered = list(offered)

        messages = [Message(role="system", content=self.config.system_prompt)]
        if items:
            messages.append(Message(role="system", content="\n\n".join(
                [CONTEXT_MARKER] + [f"[{i.kind}] {i.text}" for _, i in items])))
        messages.append(Message(role="user", content=task.prompt))

        seen_ids: set[str] = set()
        for turn in range(1, self.config.max_turns + 1):
            run.turn = turn
            result = await anyio.to_thread.run_sync(self.llm.chat, messages, specs, self.config.options)
            message = result.message
            calls = []
            for i, c in enumerate(message.tool_calls):
                # Ollama and the scripted backend number calls per turn; make ids unique per run.
                call_id = c.id if c.id and c.id not in seen_ids else f"t{turn}_{c.id or i}"
                seen_ids.add(call_id)
                calls.append(c.model_copy(update={"id": call_id}))
            message = message.model_copy(update={"tool_calls": calls})
            trace.llm_calls.append(LLMCallRecord(
                turn=turn, prompt_tokens=result.usage.prompt, completion_tokens=result.usage.completion,
                measured=result.usage.measured, thinking_tokens=result.usage.thinking,
                cached_prompt_tokens=result.usage.cached_prompt,
                est_breakdown=breakdown(messages, specs, self.counter, CONTEXT_MARKER),
                latency_ms=result.latency_ms, done_reason=result.stop_reason, tools_offered=list(offered),
                messages_sent=len(messages),
                tool_results_in_context={m.tool_call_id: self.counter(m.content) for m in messages
                                         if m.role == "tool" and m.tool_call_id},
                applied_options=result.applied_options, ignored_options=result.ignored_options,
                retries=result.retries, waited_s=result.waited_s,
                tool_calls=[c.id for c in calls],
            ))
            messages.append(message)
            if not calls:
                trace.stop_reason, trace.answer = "final", message.content
                break
            for c in calls:
                pending = PendingCall(id=c.id, turn=turn, name=c.name, arguments=c.arguments)
                messages.append(await self._tool_call(run, pending, offered, extra))
            if run.offers:
                # Tools offered mid-run get the same schema transforms as the initial ones (E2).
                new = [t for t in run.offers if t.name not in offered]
                for s in self.strategies:
                    new = self._hook(run, s, "transform_schemas", "tools", new, lambda s=s, t=new: s.transform_schemas(t))
                for t in new:
                    offered[t.name] = t
                    specs.append(ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema))
                run.offers = []
                trace.tools_offered = list(offered)
            for s in self.strategies:
                messages = self._hook(run, s, "compact_history", "history", messages,
                                      lambda s=s, m=messages: s.compact_history(m))
        else:
            trace.stop_reason = "max_turns"
            await self._at_turn_limit(run, task, messages)
        trace.messages = messages

    async def _at_turn_limit(self, run: RunContext, task: Task, messages: list[Message]) -> None:
        """One extra LLM call with no tools, if a strategy asks for it (E9)."""
        run.turn = self.config.max_turns + 1
        closing = None
        for s in self.strategies:
            run.strategy, run.hook = s.name, "at_turn_limit"
            start = len(run.trace.transformations)
            closing = s.at_turn_limit(task, messages)
            if closing is not None:
                if not any(t.strategy == s.name for t in run.trace.transformations[start:]):
                    run.trace.transformations.append(TransformationRecord(
                        turn=run.turn, strategy=s.name, hook="at_turn_limit", target="history", before_tokens=0,
                        after_tokens=self.counter(closing.content), note="change not reported by the strategy",
                        reported=False))
                break
        if closing is None:
            return
        messages.append(closing)
        result = await anyio.to_thread.run_sync(self.llm.chat, messages, [], self.config.options)
        # No tools were offered; any tool calls in the reply are dropped, not executed.
        message = result.message.model_copy(update={"tool_calls": []})
        run.trace.llm_calls.append(LLMCallRecord(
            turn=run.turn, prompt_tokens=result.usage.prompt, completion_tokens=result.usage.completion,
            measured=result.usage.measured, thinking_tokens=result.usage.thinking,
            cached_prompt_tokens=result.usage.cached_prompt,
            est_breakdown=breakdown(messages, [], self.counter, CONTEXT_MARKER),
            latency_ms=result.latency_ms, done_reason=result.stop_reason, tools_offered=[],
            messages_sent=len(messages),
            tool_results_in_context={m.tool_call_id: self.counter(m.content) for m in messages
                                     if m.role == "tool" and m.tool_call_id},
            applied_options=result.applied_options, ignored_options=result.ignored_options,
                retries=result.retries, waited_s=result.waited_s, tool_calls=[],
        ))
        messages.append(message)
        run.trace.stop_reason, run.trace.answer = "final_at_limit", message.content

    def _hook(self, run: RunContext, s: Strategy, hook: str, target: str, before: list, call) -> list:
        """Runs one hook; records an unreported change without evidence."""
        run.strategy, run.hook = s.name, hook
        start = len(run.trace.transformations)
        after = call()
        reported = any(t.strategy == s.name and t.hook == hook for t in run.trace.transformations[start:])
        if not reported and _dump(after) != _dump(before):
            run.trace.transformations.append(TransformationRecord(
                turn=run.turn, strategy=s.name, hook=hook, target=target,
                before_tokens=self._tokens_of(before), after_tokens=self._tokens_of(after),
                note="change not reported by the strategy", reported=False,
            ))
        return after

    def _tokens_of(self, values: list) -> int:
        return sum(self.counter(spec_view(v) if isinstance(v, ToolSpec) else message_text(v)) for v in values)

    async def _tool_call(self, run: RunContext, call: PendingCall, offered: dict[str, ToolDef],
                         extra: dict[str, ExtraTool]) -> Message:
        args = call.arguments
        decision = CallDecision()
        for s in self.strategies:
            run.strategy, run.hook = s.name, "before_call"
            d = s.before_call(call.model_copy(update={"arguments": args}))
            if d.action == "proceed":
                continue
            run.transform(call.name, before=args, after=d.arguments or d.hint or (d.result.text if d.result else ""),
                          evidence=d.evidence, note=d.note or d.action)
            decision = d
            if d.action == "repaired":
                args = d.arguments or args
                continue
            break

        tool = offered.get(call.name)
        result, error_class, text = None, None, None
        if decision.action == "rejected":
            text, error_class = decision.hint or "Call rejected.", "rejected"
        elif decision.action == "cached":
            result = decision.result
        elif tool is None:
            known = call.name in extra or any(t.name == call.name for t in self.hub.tools)
            error_class = "not_offered" if known else "unknown_tool"
            text = f"Error: tool {call.name!r} is not available."
        else:
            try:
                if call.name in extra:
                    result = await extra[call.name].handler(call.model_copy(update={"arguments": args}))
                else:
                    result = await self.hub.call(call.name, args)
            except ToolCallTimeout as e:
                text, error_class = f"Error: {e}", "timeout"

        shaped = result
        if result is not None:
            if result.is_error:
                error_class = "protocol_error" if result.protocol_error else "tool_error"
            for s in self.strategies:
                shaped = self._shape(run, s, call, shaped)
            text = shaped.text

        is_error = error_class is not None
        run.trace.tool_calls.append(ToolCallRecord(
            turn=call.turn, call_id=call.id, name=call.name, server=tool.server if tool else None,
            tool=tool.tool if tool else None, arguments=call.arguments, final_arguments=args,
            args_tokens=self.counter(canonical_json(args)), decision=decision.action,
            raw_result_tokens=self.counter(result.text) if result is not None else None,
            shaped_result_tokens=self.counter(text), latency_ms=result.latency_ms if result else 0.0,
            is_error=is_error, error_class=error_class, cache_hit=decision.action == "cached",
            replayed=bool(result and result.source == "replay"), source=result.source if result else None,
            recorded_at=result.recorded_at if result else None,
            raw_text=result.text if result is not None else None, shaped_text=text,
        ))
        return Message(role="tool", tool_call_id=call.id, tool_name=call.name, content=text, is_error=is_error)

    def _shape(self, run: RunContext, s: Strategy, call: PendingCall, result: ToolResult) -> ToolResult:
        run.strategy, run.hook = s.name, "after_result"
        start = len(run.trace.transformations)
        shaped = s.after_result(call, result)
        reported = any(t.strategy == s.name and t.hook == "after_result"
                       for t in run.trace.transformations[start:])
        if not reported and (shaped.text, shaped.structured) != (result.text, result.structured):
            run.trace.transformations.append(TransformationRecord(
                turn=run.turn, strategy=s.name, hook="after_result", target=call.name,
                before_tokens=self.counter(result.text), after_tokens=self.counter(shaped.text),
                note="change not reported by the strategy", reported=False,
            ))
        return shaped
