"""CodingAgent: ReAct-style tool loop over OpenAI-compatible API (base_url aware).

Same safeguards as starter-code/template.py:
- bounded `max_iterations`, per-turn `trace[]`
- tool-name `.strip().lower()` normalization (in ToolRegistry.invoke)
- invalid-tool / error observations fed back instead of looping forever
- no API key -> deterministic offline mock (so `python main.py` works before key setup)
"""

from __future__ import annotations

import json
from typing import Any

from config import Config
from memory import LongTermMemory, ShortTermMemory
from tools import ToolRegistry

SYSTEM_PROMPT = """You are a CLI coding agent sandboxed to the workspace root.
Rules:
- Prefer read_file/list_dir/search_files before editing. Never invent file contents.
- run_bash runs non-interactively inside the workspace (cwd=workdir, timeout-capped).
  Prefer it for tests/builds/git status; never for interactive editors or sudo.
  It always asks the user for approval — state the exact command first.
- GitHub work uses the full `github_*` MCP tools (upstream server). Prefer
  read-only tools (list/get/search) before any create/update/merge, which need
  user approval. Never invent issue/PR numbers.
- To recall durable facts, read MEMORY.md (path: {memory_path}).
- To save durable user/project facts, call add_memory(reason_to_add, memory_to_add).
- Keep answers concise; show file paths and code blocks when relevant.
- If a tool returns an Error, explain it and stop retrying after 2 failures.
"""


class CodingAgent:
    def __init__(
        self,
        config: Config,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        tools: ToolRegistry,
    ):
        self.config = config
        self.short_term = short_term
        self.long_term = long_term
        self.tools = tools
        self.trace: list[dict[str, Any]] = []
        self._client = None
        if config.has_api_key:
            try:
                from openai import OpenAI  # type: ignore

                # base_url makes DeepSeek / OpenRouter / Ollama work with same code.
                self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
            except Exception:
                self._client = None

    @property
    def online(self) -> bool:
        return self._client is not None

    # -- context ------------------------------------------------------------
    def _build_messages(self, session_id: str, user_input: str) -> list[dict[str, Any]]:
        mem_excerpt = self.long_term.read(limit_chars=3000)
        msgs: list[dict[str, Any]] = [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(memory_path=str(self.config.memory_path))
             + f"\n<Long-term memory>\n{mem_excerpt}\n</long-term memory>"},
        ]
        for m in self.short_term.get_window(session_id, self.config.short_term_window):
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_input})
        return msgs

    # -- public --------------------------------------------------------------
    def run(self, session_id: str, user_input: str) -> dict[str, Any]:
        """Non-streaming wrapper (backwards compatible). Collects stream()."""
        try:
            final: dict[str, Any] | None = None
            for ev in self.stream(session_id, user_input):
                if ev["type"] == "done":
                    final = ev
            assert final is not None
            return {"status": final["status"], "iterations": final.get("iterations", 1),
                    "answer": final["answer"], "trace": final["trace"],
                    **({"mock": True} if final.get("mock") else {})}
        except Exception as e:
            # Never crash the REPL (e.g. bad base_url) -> mock fallback.
            return self._run_mock(session_id, user_input, online_error=str(e))

    def stream(self, session_id: str, user_input: str):
        """Yield live events: message_start, content_delta, tool_start,
        tool_end, done. Mock path (no key) yields synthetic content deltas."""
        self.trace = []
        if self.online:
            try:
                yield from self._stream_online(session_id, user_input)
                return
            except Exception as e:
                yield from self._stream_mock(session_id, user_input, online_error=str(e))
                return
        yield from self._stream_mock(session_id, user_input)

    # -- streaming online loop --------------------------------------------------
    def _stream_online(self, session_id: str, user_input: str):
        assert self._client is not None
        messages = self._build_messages(session_id, user_input)
        self.short_term.add_message(session_id, "user", user_input)
        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1
            yield {"type": "message_start", "iteration": iteration}
            try:
                chunk_iter = self._client.chat.completions.create(
                    model=self.config.model, messages=messages,
                    tools=self.tools.openai_tools, stream=True,
                )
            except Exception:
                # Stream setup failed -> single non-streaming turn fallback.
                yield from self._nonstream_turn(messages, session_id, iteration)
                if self.trace and "final_answer" in self.trace[-1]:
                    yield {"type": "done", "status": "completed",
                           "iterations": iteration,
                           "answer": self.trace[-1]["final_answer"], "trace": self.trace}
                    return
                continue

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            calls: dict[int, dict[str, Any]] = {}
            announced: set[int] = set()
            for chunk in chunk_iter:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "content_delta", "text": delta.content}
                # DeepSeek reasoning_content rides along as an extra field when
                # the model sends it; otherwise plain getattr -> None. No extractor.
                r = getattr(delta, "reasoning_content", None)
                if r:
                    reasoning_parts.append(r)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": []})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = tc.function
                    if fn:
                        if fn.name:
                            slot["name"] = fn.name if not slot["name"] else slot["name"]
                        if fn.arguments:
                            slot["args"].append(fn.arguments)
                    if tc.index not in announced and slot["name"].strip():
                        announced.add(tc.index)
                        yield {"type": "tool_start", "name": slot["name"].strip()}

            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            tool_calls = [{"id": s["id"], "name": s["name"].strip(),
                           "arguments": "".join(s["args"])}
                          for s in calls.values() if s["name"].strip()]
            if not tool_calls:
                self.trace.append({"iteration": iteration,
                                   "thought": reasoning or "final answer",
                                   "final_answer": content})
                self.short_term.add_message(session_id, "assistant", content)
                yield {"type": "done", "status": "completed",
                       "iterations": iteration, "answer": content, "trace": self.trace}
                return
            messages.append({"role": "assistant", "content": content or "",
                             "tool_calls": [
                                 {"id": t["id"], "type": "function",
                                  "function": {"name": t["name"], "arguments": t["arguments"]}}
                                 for t in tool_calls]})
            for t in tool_calls:
                name = (t["name"] or "").strip()
                try:
                    args = json.loads(t["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                    obs = "Observation: Invalid JSON format in tool arguments."
                else:
                    obs = self.tools.invoke(name, args)
                key = name.strip().lower()
                self.trace.append({"iteration": iteration,
                                   "thought": reasoning[:500] if reasoning else f"call {key}",
                                   "action": {"name": key, "args": args}, "observation": obs})
                yield {"type": "tool_end", "name": key,
                       "observation": obs if isinstance(obs, str) else str(obs)}
                messages.append({"role": "tool", "tool_call_id": t["id"],
                                 "content": obs if isinstance(obs, str) else str(obs)})
        answer = "Stopped: max_iterations reached without a final answer."
        self.short_term.add_message(session_id, "assistant", answer)
        yield {"type": "done", "status": "max_iterations_reached",
               "iterations": iteration, "answer": answer, "trace": self.trace}

    def _nonstream_turn(self, messages: list[dict], session_id: str, iteration: int):
        """One blocking turn; used only if streaming setup fails mid-loop."""
        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.config.model, messages=messages, tools=self.tools.openai_tools)
        choice = resp.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None) or []
        if not tool_calls:
            answer = choice.content or ""
            yield {"type": "content_delta", "text": answer}
            self.trace.append({"iteration": iteration, "thought": "final answer",
                               "final_answer": answer})
            self.short_term.add_message(session_id, "assistant", answer)
            return
        messages.append({"role": "assistant", "content": choice.content or "",
                         "tool_calls": [
                             {"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name,
                                           "arguments": tc.function.arguments}}
                             for tc in tool_calls]})
        for tc in tool_calls:
            name = (tc.function.name or "").strip()
            yield {"type": "tool_start", "name": name.strip().lower()}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                obs = "Observation: Invalid JSON format in tool arguments."
            else:
                obs = self.tools.invoke(name, args)
            key = name.strip().lower()
            self.trace.append({"iteration": iteration, "thought": f"call {key}",
                               "action": {"name": key, "args": args}, "observation": obs})
            yield {"type": "tool_end", "name": key,
                   "observation": obs if isinstance(obs, str) else str(obs)}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": obs if isinstance(obs, str) else str(obs)})

    def _stream_mock(self, session_id: str, user_input: str,
                     online_error: str | None = None):
        """Offline path: reuse _run_mock, then replay answer as deltas."""
        result = self._run_mock(session_id, user_input, online_error=online_error)
        answer = result["answer"]
        yield {"type": "message_start", "iteration": 1}
        for i in range(0, len(answer), 120):
            yield {"type": "content_delta", "text": answer[i:i + 120]}
        yield {"type": "done", "status": result["status"],
               "iterations": result.get("iterations", 1),
               "answer": answer, "trace": result["trace"], "mock": True}

    # -- online loop ----------------------------------------------------------
    def _run_online(self, session_id: str, user_input: str) -> dict[str, Any]:
        assert self._client is not None
        messages = self._build_messages(session_id, user_input)
        self.short_term.add_message(session_id, "user", user_input)
        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1
            resp = self._client.chat.completions.create(
                model=self.config.model, messages=messages, tools=self.tools.openai_tools,
            )
            choice = resp.choices[0].message
            tool_calls = getattr(choice, "tool_calls", None) or []
            if not tool_calls:
                answer = choice.content or ""
                self.trace.append({"iteration": iteration, "thought": "final answer",
                                   "final_answer": answer})
                self.short_term.add_message(session_id, "assistant", answer)
                return {"status": "completed", "iterations": iteration,
                        "answer": answer, "trace": self.trace}
            # Execute each tool call, feed observations back.
            messages.append({"role": "assistant", "content": choice.content or "",
                             "tool_calls": [
                                 {"id": tc.id, "type": "function",
                                  "function": {"name": tc.function.name,
                                               "arguments": tc.function.arguments}}
                                 for tc in tool_calls]})
            for tc in tool_calls:
                name = (tc.function.name or "").strip()
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                    obs = "Observation: Invalid JSON format in tool arguments."
                else:
                    obs = self.tools.invoke(name, args)
                key = name.strip().lower()
                self.trace.append({"iteration": iteration, "thought": f"call {key}",
                                   "action": {"name": key, "args": args}, "observation": obs})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": obs if isinstance(obs, str) else str(obs)})
        answer = "Stopped: max_iterations reached without a final answer."
        self.short_term.add_message(session_id, "assistant", answer)
        return {"status": "max_iterations_reached", "iterations": iteration,
                "answer": answer, "trace": self.trace}

    # -- offline mock (no key / no openai pkg) ---------------------------------
    def _run_mock(self, session_id: str, user_input: str,
                  online_error: str | None = None) -> dict[str, Any]:
        """Keyword-driven demo so CLI + memory + tools are testable offline."""
        self.short_term.add_message(session_id, "user", user_input)
        low = user_input.lower()
        results: list[dict[str, Any]] = []

        def _do(name: str, args: dict) -> None:
            obs = self.tools.invoke(name, args)
            results.append({"tool": name, "observation": obs})
            self.trace.append({"iteration": len(results), "thought": f"mock call {name}",
                               "action": {"name": name, "args": args}, "observation": obs})

        if "memory" in low:
            _do("read_file", {"path": str(self.config.memory_path)})
        if "list" in low or "ls" in low or "files" in low:
            _do("list_dir", {"path": "."})
        # naive "read <path>" / "open <path>"
        for kw in ("read ", "open ", "show "):
            if kw in low:
                frag = user_input.lower().split(kw, 1)[1].strip().split()[0].strip("'\"`")
                _do("read_file", {"path": frag})
                break

        if results:
            parts = [f"- `{r['tool']}` →\n```\n{str(r['observation'])[:1500]}\n```" for r in results]
            answer = ("[offline mock — add `OPENAI_API_KEY` to `.env` for real answers]"
                      + (f"\nOnline error: {online_error}" if online_error else "")
                      + "\n" + "\n".join(parts))
        else:
            answer = ("[offline mock] I need `OPENAI_API_KEY` in `.env` for real answers. "
                      "Try `/tools`, `/memory`, `list files`, or `read MEMORY.md`.")
            self.trace.append({"iteration": 1, "thought": "no tool needed",
                               "final_answer": answer})
        self.short_term.add_message(session_id, "assistant", answer)
        return {"status": "completed", "iterations": max(1, len(results)),
                "answer": answer, "trace": self.trace, "mock": True}
