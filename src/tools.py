"""Filesystem + shell + GitHub-MCP tools.

Canonical tools exposed to the LLM (OpenAI function-calling):
  read_file(path) | write_file(path, content) | edit_file(path, old_string, new_string)
  | list_dir(path=".") | search_files(pattern, path=".") | add_memory(reason_to_add, memory_to_add)
  | run_bash(command, workdir=".", timeout=30)
  | github_* (full GitHub MCP server, when configured — see mcp_tools.py)

Strategy (per user request "use prebuilt langchain tools"):
- If `langchain_community` is installed, read/write/list delegate to its
  FileManagementToolkit tools (ReadFileTool, WriteFileTool, ListDirectoryTool,
  + FileSearchTool when present), sandboxed with root_dir=workspace_root.
- `edit_file`, `search_files` (content grep), `run_bash` and `add_memory` have
  no direct prebuilt equivalent, so they are small native tools with the same
  interface. This keeps offline/CI working without the heavy dep too.
- `github_*` tools are NOT native: they come from the upstream GitHub MCP
  server (`@modelcontextprotocol/server-github` via npx stdio) bridged through
  `mcp_tools.MCPBridge` into OpenAI function schemas. Optional `mcp` package;
  when missing/unconfigured the tools are simply not advertised.
- Tool-name lookup always uses `.strip().lower()` (same trap fix as Lab #3).
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------- sandbox
DENIED_WRITE_NAMES = {".env", "agent.db"}
DENIED_PARTS = {".git"}

# ------------------------------------------------- human-in-the-loop gate
# Tools that mutate the env need explicit user approval (Option A:
# server-side confirmation, NOT an LLM-controlled boolean param which
# the model could just set to true and bypass).
MUTATING_TOOLS = {"add_memory", "write_file", "edit_file", "run_bash"}
# run_bash is always gated (even `ls`) because the model controls the string.
# github_* MCP writes are gated heuristically via mcp_tools.is_mcp_write_tool.

ConfirmFn = Callable[[str, str], bool]


def _default_confirm(tool_name: str, preview: str) -> bool:
    """Fallback y/n prompt. Used only when a confirm_fn is wired (CLI)."""
    print(f"\n[confirm] Agent wants to call `{tool_name}`:\n{preview}\nAllow? [y/N]: ", end="")
    try:
        return input().strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _resolve(user_path: str, root: Path) -> Path:
    root = root.resolve()
    # Empty means workspace root (for list_dir).
    target = (root / (user_path or ".")).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes workspace root: {user_path!r}")
    return target


def _deny_write(target: Path, root: Path) -> str | None:
    rel = target.resolve().relative_to(root.resolve())
    if rel.name in DENIED_WRITE_NAMES:
        return f"Refusing to write protected file: {rel}"
    if DENIED_PARTS & set(rel.parts):
        return f"Refusing to write inside protected dir: {rel}"
    return None


# ------------------------------------------------------- native fallbacks
def _native_read(root: Path, path: str) -> str:
    t = _resolve(path, root)
    if not t.is_file():
        return f"Error: not a file: {path!r}"
    try:
        return t.read_text(encoding="utf-8")[:20000]
    except Exception as e:
        return f"Error reading {path!r}: {e}"


def _native_write(root: Path, path: str, content: str) -> str:
    t = _resolve(path, root)
    denied = _deny_write(t, root)
    if denied:
        return f"Error: {denied}"
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(content or "", encoding="utf-8")
    return f"Wrote {len(content or '')} chars to {t.relative_to(root.resolve())}"


def _native_edit(root: Path, path: str, old_string: str, new_string: str) -> str:
    t = _resolve(path, root)
    denied = _deny_write(t, root)
    if denied:
        return f"Error: {denied}"
    if not t.is_file():
        return f"Error: not a file: {path!r}"
    text = t.read_text(encoding="utf-8")
    if old_string not in text:
        return "Error: old_string not found in file."
    if text.count(old_string) > 1:
        return "Error: old_string matches multiple times; be more specific."
    t.write_text(text.replace(old_string, new_string), encoding="utf-8")
    return f"Edited {t.relative_to(root.resolve())}"


def _native_list(root: Path, path: str = ".") -> str:
    t = _resolve(path, root)
    if not t.exists():
        return f"Error: path not found: {path!r}"
    if t.is_file():
        return str(t.relative_to(root.resolve()))
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in t.iterdir())
    return "\n".join(entries) if entries else "(empty directory)"


def _native_search(root: Path, pattern: str, path: str = ".") -> str:
    """Content grep (regex) + filename glob fallback, capped at 50 hits."""
    t = _resolve(path, root)
    base = t if t.is_dir() else t.parent
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    hits: list[str] = []
    for p in base.rglob("*"):
        if len(hits) >= 50:
            break
        if not p.is_file() or ".git" in p.parts or ".venv" in p.parts:
            continue
        if p.suffix in {".db", ".pyc"} or "__pycache__" in p.parts:
            continue
        try:
            if fnmatch.fnmatch(p.name, pattern):
                hits.append(f"{p.relative_to(root.resolve())} (filename match)")
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{p.relative_to(root.resolve())}:{i}: {line.strip()[:160]}")
                if len(hits) >= 50:
                    break
    return "\n".join(hits) if hits else "No matches."


# ------------------------------------------------------------- bash tool
# Native subprocess (NOT langchain ShellTool: no sandbox/approval there).
BASH_DEFAULT_TIMEOUT = 30
BASH_MAX_OUTPUT = 8000
# Hard-refuse patterns: interactive shells, privilege escalation, destructive
# wipes, fork bombs, pipe-to-shell. LLM gets an Error string to self-correct.
BASH_DENY = (
    r"(?<!\w)sudo\b",
    r"rm\s+-rf\s+/(?:\s|$)",
    r"mkfs\b",
    r":\(\)\s*\{\s*:\|\:&\s*\}",
    r"curl\s+.*\|\s*(ba)?sh\b",
    r"wget\s+.*\|\s*(ba)?sh\b",
    r"\bnc\b.*-l\b",
)


def _native_bash(
    root: Path, command: str, workdir: str = ".", timeout: int = BASH_DEFAULT_TIMEOUT
) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return "Error: empty command."
    for pat in BASH_DENY:
        if re.search(pat, cmd):
            return f"Error: refused dangerous command (matched `{pat}`)."
    try:
        cwd = _resolve(workdir or ".", root)
    except ValueError as e:
        return f"Error: {e}"
    if not cwd.exists():
        return f"Error: workdir not found: {workdir!r}"
    if cwd.is_file():
        cwd = cwd.parent
    try:
        timeout_s = max(1, min(int(timeout or BASH_DEFAULT_TIMEOUT), 300))
    except (TypeError, ValueError):
        timeout_s = BASH_DEFAULT_TIMEOUT
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout_s}s: {cmd[:200]!r}"
    except FileNotFoundError:
        return "Error: `bash` not found on this system."
    except Exception as e:
        return f"Error executing bash: {e}"
    out = (proc.stdout or "") + (proc.stderr and f"\n[stderr]\n{proc.stderr}" or "")
    out = out.strip() or "(no output)"
    if len(out) > BASH_MAX_OUTPUT:
        out = out[:BASH_MAX_OUTPUT] + f"\n…[truncated {len(out) - BASH_MAX_OUTPUT} chars]"
    rel = cwd.relative_to(root.resolve())
    return f"$ {cmd}\n[cwd: {rel} | exit {proc.returncode}]\n{out}"


# ------------------------------------------------- langchain prebuilt layer
# langchain-community>=0.3 removed FileManagementToolkit; individual tools
# (ReadFileTool, WriteFileTool, ...) are the stable import path. We support both.
_LANGCHAIN_OK = False

try:  # pragma: no cover - depends on installed version
    from langchain_community.tools.file_management import (  # type: ignore
        FileSearchTool,
        ListDirectoryTool,
        ReadFileTool,
        WriteFileTool,
    )

    _LANGCHAIN_OK = True
except Exception:
    try:
        from langchain_community.tools.file_management.toolkit import (  # type: ignore
            FileManagementToolkit,
        )

        _LANGCHAIN_OK = True
    except Exception:
        _LANGCHAIN_OK = False


def _build_langchain_tools(root: Path) -> dict[str, Any]:
    """Instantiate prebuilt file tools sandboxed to root. Returns {} on failure."""
    try:
        try:
            from langchain_community.tools.file_management import (  # type: ignore
                FileSearchTool,
                ListDirectoryTool,
                ReadFileTool,
                WriteFileTool,
            )

            tools = [
                ReadFileTool(root_dir=str(root.resolve())),
                WriteFileTool(root_dir=str(root.resolve())),
                ListDirectoryTool(root_dir=str(root.resolve())),
                FileSearchTool(root_dir=str(root.resolve())),
            ]
        except Exception:
            from langchain_community.tools.file_management.toolkit import (  # type: ignore
                FileManagementToolkit,
            )

            toolkit = FileManagementToolkit(root_dir=str(root.resolve()))
            tools = toolkit.get_tools()
        # NOTE: some versions ship names with stray whitespace -> strip().
        return {t.name.strip().lower(): t for t in tools}
    except Exception:
        return {}


class ToolRegistry:
    """Holds TOOL_MAP + OpenAI schemas + invoke() with normalization.

    Native tools live in TOOL_MAP. GitHub MCP tools (`github_*`) are served
    by optional `mcp_tools.MCPBridge` instances passed as `mcp_bridges`.
    """

    def __init__(
        self,
        workspace_root: Path,
        memory_append: Callable[[str, str], str],
        confirm_fn: ConfirmFn | None = None,
        auto_confirm: bool = False,
        mcp_bridges: list[Any] | None = None,
        bash_timeout: int = BASH_DEFAULT_TIMEOUT,
    ):
        self.root = workspace_root.resolve()
        self._memory_append = memory_append
        # confirm_fn(tool_name, preview) -> True allows, False cancels.
        # None = no gating (backward compat for tests/offline mock).
        # auto_confirm=True bypasses gating (CI / --yes / AUTO_CONFIRM=1).
        self._confirm_fn = confirm_fn
        self.auto_confirm = auto_confirm
        self._mcp_bridges: list[Any] = list(mcp_bridges or [])
        self.bash_timeout = bash_timeout
        self._lc = _build_langchain_tools(self.root) if _LANGCHAIN_OK else {}
        self.backend = "langchain" if self._lc else "native"
        self.TOOL_MAP: dict[str, Callable[..., str]] = self._build_map()

    def _lc_invoke(self, name: str, args: dict) -> str | None:
        """Try a prebuilt langchain tool; return None if not present/fails to import."""
        tool = self._lc.get(name)
        if tool is None:
            return None
        try:
            out = tool.invoke(args)
            return out if isinstance(out, str) else str(out)
        except Exception as e:
            return f"Error (langchain {name}): {e}"

    def _needs_confirm(self, key: str) -> bool:
        if self._confirm_fn is None or self.auto_confirm:
            return False
        if key in MUTATING_TOOLS:
            return True
        # Full GitHub MCP: only write-like tools gate; reads stay frictionless.
        if key.startswith(("github_", "mcp_")):
            try:
                from mcp_tools import is_mcp_write_tool
            except Exception:
                return True  # fail closed when heuristic unavailable
            return bool(is_mcp_write_tool(key))
        return False

    @staticmethod
    def _preview(key: str, args: dict) -> str:
        if key == "add_memory":
            return (
                f"reason={args.get('reason_to_add')!r}\n"
                f"memory={str(args.get('memory_to_add'))[:500]!r}"
            )
        if key == "write_file":
            return f"path={args.get('path')!r} ({len(str(args.get('content', '')))} chars)"
        if key == "edit_file":
            return f"path={args.get('path')!r}\n-{str(args.get('old_string'))[:300]!r}"
        if key == "run_bash":
            return (
                f"command={str(args.get('command'))[:500]!r}\n"
                f"workdir={args.get('workdir', '.')!r} timeout={args.get('timeout', 30)!r}"
            )
        if key.startswith(("github_", "mcp_")):
            return f"{key}({str(args)[:500]})"
        return str(args)[:500]

    def _build_map(self):
        root = self.root

        def read_file(path: str) -> str:
            out = self._lc_invoke("read_file", {"file_path": path})
            if out is not None:
                return out
            return _native_read(root, path)

        def write_file(path: str, content: str = "") -> str:
            # langchain WriteFileTool arg names differ by version; try both.
            for args in ({"file_path": path, "text": content}, {"file_path": path, "content": content}):
                tool = self._lc.get("write_file")
                if tool is not None:
                    try:
                        out = tool.invoke(args)
                        return out if isinstance(out, str) else str(out)
                    except Exception:
                        continue
            return _native_write(root, path, content)

        def list_dir(path: str = ".") -> str:
            for key, args in (
                ("list_directory_tool", {"dir_path": path}),
                ("list_directory", {"dir_path": path}),
                ("list_directory_tool", {"directory_path": path}),
            ):
                out = self._lc_invoke(key, args)
                if out is not None:
                    return out
            # also try any lc tool whose name contains 'list'
            for name in self._lc:
                if "list" in name:
                    for args in ({"dir_path": path}, {"directory": path}, {"path": path}):
                        try:
                            out = self._lc[name].invoke(args)
                            return out if isinstance(out, str) else str(out)
                        except Exception:
                            continue
            return _native_list(root, path)

        def search_files(pattern: str, path: str = ".") -> str:
            # Prebuilt FileSearchTool is filename-oriented; content grep is native.
            # Pass dir_path so non-root searches still use the prebuilt tool.
            lc = self._lc.get("file_search")
            if lc is not None:
                try:
                    out = lc.invoke({"pattern": pattern, "dir_path": path or "."})
                    if isinstance(out, str) and out.strip():
                        return out
                except Exception:
                    pass
            return _native_search(root, pattern, path)

        def edit_file(path: str, old_string: str, new_string: str) -> str:
            return _native_edit(root, path, old_string, new_string)

        def add_memory(reason_to_add: str, memory_to_add: str) -> str:
            return self._memory_append(reason_to_add, memory_to_add)

        def run_bash(command: str, workdir: str = ".", timeout: int | None = None) -> str:
            return _native_bash(
                root, command, workdir or ".",
                timeout if timeout is not None else self.bash_timeout,
            )

        return {
            "read_file": read_file,
            "write_file": write_file,
            "edit_file": edit_file,
            "list_dir": list_dir,
            "search_files": search_files,
            "run_bash": run_bash,
            "add_memory": add_memory,
        }

    # -- OpenAI function schemas (stable, version-independent) -------------
    @property
    def openai_tools(self) -> list[dict]:
        tools = [
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file (sandboxed to workspace). Use this to retrieve long-term memory (MEMORY.md) or project files.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "Create/overwrite a file with content. Parent dirs auto-created. Requires user confirmation before executing.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {
                "name": "edit_file",
                "description": "Exact-string replace in a file. Fails if old_string missing or ambiguous. Requires user confirmation before executing.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "old_string": {"type": "string"},
                    "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
            {"type": "function", "function": {
                "name": "list_dir",
                "description": "List directory contents.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
            {"type": "function", "function": {
                "name": "search_files",
                "description": "Search filenames (glob) and file contents (regex). Returns up to 50 hits.",
                "parameters": {"type": "object", "properties": {
                    "pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {
                "name": "run_bash",
                "description": "Run a bash command jailed to the workspace (cwd=workdir). Non-interactive, timeout-capped. Requires user confirmation before executing.",
                "parameters": {"type": "object", "properties": {
                    "command": {"type": "string", "description": "Bash command to run."},
                    "workdir": {"type": "string", "description": "Directory relative to workspace root."},
                    "timeout": {"type": "integer", "description": "Seconds, 1-300."}},
                    "required": ["command"]}}},
            {"type": "function", "function": {
                "name": "add_memory",
                "description": "Save durable user/project facts to MEMORY.md long-term memory. Requires user confirmation before executing.",
                "parameters": {"type": "object", "properties": {
                    "reason_to_add": {"type": "string"}, "memory_to_add": {"type": "string"}},
                    "required": ["reason_to_add", "memory_to_add"]}}},
        ]
        # Full GitHub MCP tools (lazy: first access spawns npx once to list).
        for bridge in self._mcp_bridges:
            try:
                tools.extend(bridge.list_openai_tools())
            except Exception:
                continue
        return tools

    # -- execution ----------------------------------------------------------
    def _mcp_invoke(self, key: str, args: dict) -> str | None:
        """Route `github_*`/`mcp_*` to the owning bridge; None if no bridge owns it."""
        for bridge in self._mcp_bridges:
            try:
                names = {t["function"]["name"] for t in (bridge._schemas or [])}
            except Exception:
                names = set()
            if key in names:
                try:
                    return str(bridge.call_tool_sync(key, args))
                except Exception as e:
                    return f"Error executing {key}: {e}"
        # Lazy case: schemas not fetched yet (first github_* call lists first).
        for bridge in self._mcp_bridges:
            try:
                schemas = bridge.list_openai_tools()
            except Exception:
                continue
            if any(t["function"]["name"] == key for t in schemas):
                try:
                    return str(bridge.call_tool_sync(key, args))
                except Exception as e:
                    return f"Error executing {key}: {e}"
        return None

    def invoke(self, name: str, args: dict | None = None) -> str:
        key = (name or "").strip().lower()
        aliases = {"list_directory": "list_dir", "list_directory_tool": "list_dir",
                   "read": "read_file", "write": "write_file", "edit": "edit_file",
                   "grep": "search_files", "search": "search_files", "file_search": "search_files",
                   "bash": "run_bash", "shell": "run_bash", "exec": "run_bash",
                   "run_shell": "run_bash"}
        key = aliases.get(key, key)
        # MCP namespace before native lookup (native names never start with github_/mcp_).
        if key.startswith(("github_", "mcp_")):
            if self._needs_confirm(key):
                try:
                    allowed = self._confirm_fn(key, self._preview(key, args or {}))  # type: ignore[misc]
                except Exception as e:
                    return f"Cancelled: confirmation prompt failed for `{key}`: {e}"
                if not allowed:
                    return (
                        f"Cancelled by user: `{key}` not executed. "
                        "Explain what you wanted to do and ask how to proceed."
                    )
            hit = self._mcp_invoke(key, dict(args or {}))
            if hit is not None:
                return hit
            avail: list[str] = []
            for b in self._mcp_bridges:
                try:
                    avail += [t["function"]["name"] for t in b.list_openai_tools()]
                except Exception:
                    continue
            hint = f" Available github tools: {sorted(avail)}" if avail else (
                " GitHub MCP is not configured (set GITHUB_PERSONAL_ACCESS_TOKEN/GITHUB_TOKEN + `pip install mcp`).")
            return f"Error: unknown tool {name!r}.{hint}"
        func = self.TOOL_MAP.get(key)
        if func is None:
            return f"Error: unknown tool {name!r}. Available: {sorted(self.TOOL_MAP)}"
        if self._needs_confirm(key):
            try:
                allowed = self._confirm_fn(key, self._preview(key, args or {}))  # type: ignore[misc]
            except Exception as e:
                return f"Cancelled: confirmation prompt failed for `{key}`: {e}"
            if not allowed:
                return (
                    f"Cancelled by user: `{key}` not executed. "
                    "Explain what you wanted to do and ask how to proceed."
                )
        try:
            return str(func(**(args or {})))
        except TypeError as e:
            return f"Error: bad args for {key}: {e}"
        except Exception as e:
            return f"Error executing {key}: {e}"

    def describe(self) -> str:
        lines = [f"- {n} (backend: {self.backend})" for n in sorted(self.TOOL_MAP)]
        if self._lc:
            lines.append(f"prebuilt langchain tools detected: {sorted(self._lc)}")
        for bridge in self._mcp_bridges:
            try:
                lines.append(f"- {bridge.status}")
            except Exception as e:
                lines.append(f"- mcp bridge error: {e}")
        return "\n".join(lines)
