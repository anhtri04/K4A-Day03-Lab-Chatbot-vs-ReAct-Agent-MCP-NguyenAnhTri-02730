"""Generic MCP (Model Context Protocol) stdio bridge + GitHub wiring.

Design (keeps cli-agent philosophy: optional deps, never crash the REPL):
- `mcp` python package is OPTIONAL, like `langchain-community`. If missing,
  the bridge reports `available=False` and ToolRegistry simply exposes no
  `github_*` tools (LLM gets an `Error:` string if it tries anyway).
- Full GitHub MCP = upstream server `@modelcontextprotocol/server-github`
  spawned per call via `npx -y ...` over stdio. No vendored re-implementation
  of the GitHub API; we just adapt MCP tools -> OpenAI function schemas.
- v1 uses per-call spawn (one `npx` boot for list_tools, one per tool call).
  Slower (~2-5s cold) but avoids holding an async stdio session across the
  sync REPL (no loop-thread juggling). Tool schemas are cached after first
  successful list. TODO: persistent session if latency matters.
- All output is truncated; all errors become strings (never exceptions).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any


# Heuristic: MCP tool names containing these substrings mutate GitHub state
# and therefore go through the human-in-the-loop confirm gate in ToolRegistry.
MCP_WRITE_HINTS = (
    "create",
    "update",
    "delete",
    "merge",
    "push",
    "comment",
    "add_",
    "close",
    "reopen",
    "lock",
    "fork",
    "star",
    "apply",
    "set_",
    "request",
    "dismiss",
    "submit",
)


def is_mcp_write_tool(openai_name: str) -> bool:
    n = (openai_name or "").strip().lower()
    return any(h in n for h in MCP_WRITE_HINTS)


def _sanitize_tool_name(server: str, remote: str) -> str:
    """`github` + `create_issue` -> `github_create_issue` (OpenAI-safe)."""
    raw = f"{server}_{remote}".strip().lower()
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "mcp_tool"


@dataclass
class MCPServerConfig:
    name: str = "github"  # prefix for exposed OpenAI tool names
    command: str = "npx"
    args: list[str] = field(default_factory=lambda: ["-y", "@modelcontextprotocol/server-github"])
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout: int = 60  # seconds per list/call (covers cold npx boot)


class MCPBridge:
    """Sync wrapper around one MCP stdio server. Lazy, cached, never raises."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.disable_reason: str = ""
        self._schemas: list[dict] | None = None
        self._remote_by_openai: dict[str, str] = {}
        if not config.enabled:
            self.disable_reason = "disabled by config (MCP_GITHUB_ENABLED=0)"

    @property
    def available(self) -> bool:
        return not self.disable_reason and self._check_deps() is None

    def _check_deps(self) -> str | None:
        try:
            import mcp  # noqa: F401
            from mcp.client.stdio import stdio_client  # noqa: F401
        except Exception as e:
            return f"mcp package not installed ({e}); pip install mcp"
        if not self.config.command:
            return "no MCP command configured"
        return None

    # -- async guts (one spawned server per call) ---------------------------
    async def _list_remote(self) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = {**os.environ, **(self.config.env or {})}
        params = StdioServerParameters(
            command=self.config.command, args=list(self.config.args or []), env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.list_tools()

    async def _call_remote(self, remote_name: str, args: dict) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = {**os.environ, **(self.config.env or {})}
        params = StdioServerParameters(
            command=self.config.command, args=list(self.config.args or []), env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(remote_name, args or {})

    def _run(self, coro_factory, timeout: int | None = None):
        """Run an async factory in a fresh loop; converts errors to strings."""
        async def _runner():
            return await coro_factory()

        try:
            return asyncio.run(
                asyncio.wait_for(_runner(), timeout or self.config.timeout)
            )
        except Exception as e:
            return e

    # -- sync public API -----------------------------------------------------
    def list_openai_tools(self) -> list[dict]:
        """OpenAI function schemas for this server; [] when unavailable."""
        if self.disable_reason:
            return []
        dep_err = self._check_deps()
        if dep_err:
            self.disable_reason = dep_err
            return []
        if self._schemas is not None:
            return self._schemas
        res = self._run(self._list_remote)
        if isinstance(res, Exception):
            self.disable_reason = f"github MCP list_tools failed: {res}"
            return []
        try:
            tools = getattr(res, "tools", res) or []
            schemas: list[dict] = []
            mapping: dict[str, str] = {}
            for t in tools:
                remote = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                if not remote:
                    continue
                desc = getattr(t, "description", None) or (t.get("description") if isinstance(t, dict) else "") or ""
                schema = getattr(t, "inputSchema", None) or (t.get("inputSchema") if isinstance(t, dict) else None) or {}
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    schema = {"type": "object", "properties": dict(schema.get("properties", {}) if isinstance(schema, dict) else {})}
                openai_name = _sanitize_tool_name(self.config.name, str(remote))
                mapping[openai_name] = str(remote)
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": openai_name,
                        "description": f"[github MCP] {desc}".strip()[:1000],
                        "parameters": schema,
                    },
                })
            schemas.sort(key=lambda s: s["function"]["name"])
            self._schemas = schemas
            self._remote_by_openai = mapping
            # Clear any transient failure once we succeed.
            if self.disable_reason.startswith("github MCP list_tools failed"):
                self.disable_reason = ""
            return schemas
        except Exception as e:
            self.disable_reason = f"github MCP schema parse failed: {e}"
            return []

    def call_tool_sync(self, openai_name: str, args: dict | None = None) -> str:
        key = (openai_name or "").strip().lower()
        if self.disable_reason and self._schemas is None:
            # Allow retry once in case failure was transient (e.g. cold npx).
            if self.disable_reason.startswith("github MCP list_tools failed"):
                self.disable_reason = ""
                self._schemas = None
            else:
                return f"Error: github MCP unavailable: {self.disable_reason}"
        dep_err = self._check_deps()
        if dep_err:
            return f"Error: github MCP unavailable: {dep_err}"
        if self._schemas is None:
            self.list_openai_tools()
            if self._schemas is None:
                return f"Error: github MCP unavailable: {self.disable_reason or 'unknown'}"
        remote = self._remote_by_openai.get(key)
        if remote is None:
            return f"Error: unknown github MCP tool {openai_name!r}."
        res = self._run(lambda: self._call_remote(remote, dict(args or {})))
        if isinstance(res, Exception):
            return f"Error: github MCP call `{remote}` failed: {_format_mcp_error(res)}"
        try:
            return _format_mcp_result(res)
        except Exception as e:
            return f"Error formatting github MCP result: {e}"

    @property
    def status(self) -> str:
        n = len(self._schemas) if self._schemas is not None else 0
        if self.disable_reason:
            return f"github MCP: disabled ({self.disable_reason})"
        if self._schemas is None:
            return "github MCP: not yet connected (lazy)"
        return f"github MCP: {n} tools via {self.config.command} {' '.join(self.config.args)}"


def _format_mcp_error(e: BaseException) -> str:
    """Unwrap TaskGroup/ExceptionGroup so auth failures are readable."""
    seen: list[str] = []
    stack: list[BaseException] = [e]
    while stack:
        cur = stack.pop()
        if getattr(cur, "exceptions", None):
            stack.extend(cur.exceptions)  # type: ignore[attr-defined]
            continue
        msg = str(cur).strip()
        if msg and msg not in seen:
            seen.append(msg)
    detail = "; ".join(seen) or repr(e)
    return detail[:1500]


def _format_mcp_result(res: Any, limit: int = 12000) -> str:
    """Flatten MCP CallToolResult content blocks to text."""
    parts: list[str] = []
    blocks = getattr(res, "content", None)
    if blocks is None and isinstance(res, dict):
        blocks = res.get("content", [])
    for b in blocks or []:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(str(b.get("text", "")))
            elif t == "resource":
                parts.append(str(b.get("resource", "")))
            else:
                parts.append(str(b))
        else:
            text = getattr(b, "text", None)
            if text is not None:
                parts.append(str(text))
            else:
                parts.append(str(b))
    # Some servers also set isError flag.
    if getattr(res, "isError", False) and not parts:
        parts.append(f"MCP tool error: {res}")
    out = "\n".join(parts).strip() or "(empty MCP result)"
    if len(out) > limit:
        out = out[:limit] + f"\n…[truncated {len(out) - limit} chars]"
    return out


def build_github_bridge(
    github_token: str = "",
    enabled: bool = True,
    timeout: int = 60,
) -> MCPBridge | None:
    """Full GitHub MCP server via npx. Returns None when token/flag missing.

    Upstream server auth: GITHUB_PERSONAL_ACCESS_TOKEN env var.
    """
    token = (github_token or "").strip()
    if not enabled:
        b = MCPBridge(MCPServerConfig(enabled=False))
        b.disable_reason = "disabled by config (MCP_GITHUB_ENABLED=0)"
        return b
    if not token:
        b = MCPBridge(MCPServerConfig(enabled=True))
        b.disable_reason = "no GITHUB token (set GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN)"
        return b
    return MCPBridge(MCPServerConfig(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token},
        enabled=True,
        timeout=timeout,
    ))
