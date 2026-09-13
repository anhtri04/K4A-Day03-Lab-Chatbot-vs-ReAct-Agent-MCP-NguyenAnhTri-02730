"""CLI entrypoint: REPL + slash commands.

Slash detection uses `text.strip()` (per spec):
  /new [title]  new session
  /resume [id]  list sessions or jump to one
  /memory       show MEMORY.md long-term memory
  /tools        list available tools
  /sessions     list recent sessions
  /clear        clear screen
  /help         this help
  /quit         save + quit (/exit, /q aliases)

Run:  python main.py   (from cli-agent/ or repo root)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from memory import LongTermMemory, ShortTermMemory
from tools import ToolRegistry
from agent import CodingAgent
# -- optional rich / prompt_toolkit with graceful fallback -------------------
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
except Exception:
    Console = None  # type: ignore
    Markdown = None  # type: ignore
    Panel = None  # type: ignore

try:
    from prompt_toolkit import PromptSession  # type: ignore
    from prompt_toolkit.history import FileHistory  # type: ignore
    from prompt_toolkit.formatted_text import HTML  # type: ignore
    from prompt_toolkit.styles import Style  # type: ignore
except Exception:
    PromptSession = None  # type: ignore
    FileHistory = None  # type: ignore
    HTML = None  # type: ignore
    Style = None  # type: ignore


class CLI:
    def __init__(self):
        self.config = Config.from_env()
        self.short_term = ShortTermMemory(
            self.config.db_path, self.config.short_term_window, model=self.config.model
        )
        self.long_term = LongTermMemory(self.config.memory_path)
        # Full GitHub MCP: optional bridge (no token/mcp pkg -> disabled, no crash).
        try:
            from mcp_tools import build_github_bridge

            _github = build_github_bridge(
                github_token=self.config.github_token,
                enabled=self.config.mcp_github_enabled,
                timeout=self.config.mcp_tool_timeout,
            )
            _bridges = [_github] if _github is not None else []
        except Exception:
            _bridges = []
        self.tools = ToolRegistry(
            self.config.workspace_root,
            self.long_term.append,
            confirm_fn=self.cli_confirm,
            auto_confirm=self.config.auto_confirm,
            mcp_bridges=_bridges,
            bash_timeout=self.config.bash_timeout,
        )
        self.agent = CodingAgent(self.config, self.short_term, self.long_term, self.tools)
        self.session_id = self.short_term.create_session("default")
        self.console = Console() if Console else None
        hist = str(self.config.db_path.parent / ".prompt_history")
        self.prompt = PromptSession(history=FileHistory(hist)) if PromptSession else None

    # -- output helpers -------------------------------------------------------
    def out(self, text: str, md: bool = False) -> None:
        if self.console:
            self.console.print(Markdown(text) if (md and Markdown) else text)
        else:
            print(text)

    def cli_confirm(self, tool_name: str, preview: str) -> bool:
        """Human-in-the-loop gate for mutating tools. Returns True on y/yes."""
        if self.console:
            self.console.print(f"\n[yellow]◈ {tool_name} needs approval[/yellow]\n[dim]{preview}[/dim]")
        else:
            print(f"\n[confirm] {tool_name} needs approval:\n{preview}")
        try:
            if self.prompt:
                ans = self.prompt.prompt("Allow? [y/n]: ")
            else:
                ans = input("Allow? [y/n]: ")
        except (EOFError, KeyboardInterrupt):
            self.out("Cancelled.")
            return False
        allowed = ans.strip().lower() in ("y", "yes")
        if not allowed:
            self.out(f"Cancelled `{tool_name}`.")
        return allowed

    def banner(self) -> None:
        mode = f"{self.config.model} @ {self.config.base_url}" if self.agent.online \
            else "OFFLINE mock (set OPENAI_API_KEY in .env)"
        body = (f"model: {mode}\nsession: {self.session_id}  "
                f"[backend: {self.tools.backend}]\nType /help for commands.")
        if self.console and Panel:
            self.console.print(Panel(body, title="cli-agent"))
        else:
            print(f"=== cli-agent ===\n{body}")

    def status_line(self) -> str:
        """Rich line attached above the input box: session + token total."""
        try:
            s = self.short_term.get_session(self.session_id)
        except Exception:
            s = None
        if not s:
            return f"╰─ {self.session_id[:6]} | 0 tok | {self.config.model}"
        title = s.get("title") or "untitled"
        total = s.get("total_tokens", 0)
        n = s.get("n_msgs", 0)
        return f"╰─ {title}@{str(s['id'])[:6]} | {total} tok ({n} msgs) | {self.config.model}"

    # -- input -----------------------------------------------------------------
    def read_line(self) -> str | None:
        # Status line glued above the recognizable input box.
        if self.console:
            self.console.print(f"[dim]{self.status_line()}[/dim]", highlight=False)
            self.console.print("[dim]╭─ you ─────────────────[/dim]", highlight=False)
        try:
            if self.prompt:
                if HTML:
                    return self.prompt.prompt(HTML("<b><ansicyan>│ › </ansicyan></b> "))
                return self.prompt.prompt("│ › ")
            return input("│ you › ")
        except (EOFError, KeyboardInterrupt):
            return None

    def echo_user(self, text: str) -> None:
        """Highlight the user's message in history as a cyan panel."""
        if self.console and Panel:
            self.console.print(Panel(text, title="you", border_style="cyan", padding=(0, 1)))
        else:
            print(f"[you] {text}")

    # -- slash router -----------------------------------------------------------
    def handle_slash(self, raw: str) -> bool | None:
        """Return True if handled (continue), None if should quit."""
        text = raw.strip()
        if not text.startswith("/"):
            return False
        parts = text.split(None, 1)
        cmd = parts[0].strip().lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            self.out("Bye. Session saved to agent.db")
            return None
        if cmd == "/new":
            self.session_id = self.short_term.create_session(arg or "untitled")
            self.out(f"New session: {self.session_id}")
        elif cmd == "/resume":
            sessions = self.short_term.list_sessions()
            if not sessions:
                self.out("No saved sessions.")
            elif arg and self.short_term.session_exists(arg):
                self.session_id = arg
                self.out(f"Resumed {arg}")
            else:
                lines = ["Recent sessions (use `/resume <id>`):"]
                lines += [f"- {s['id']} | {s['title']} | {s['n_msgs']} msgs | {s['created_at']}"
                          for s in sessions]
                self.out("\n".join(lines))
        elif cmd == "/memory":
            self.out(self.long_term.read() or "(memory empty)", md=True)
        elif cmd == "/tools":
            self.out(self.tools.describe())
        elif cmd == "/sessions":
            sessions = self.short_term.list_sessions()
            self.out("\n".join(f"- {s['id']} | {s['title']} | {s['n_msgs']}" for s in sessions)
                     or "(none)")
        elif cmd == "/clear":
            os.system("clear" if os.name != "nt" else "cls")
        elif cmd in ("/help", "/h"):
            self.out("**Commands:** `/new [title]` `/resume [id]` `/memory` `/tools` "
                     "`/sessions` `/clear` `/quit`", md=True)
        else:
            self.out(f"Unknown command {cmd}. Try /help.")
        return True

    # -- main loop ---------------------------------------------------------------
    def run_streaming(self, text: str) -> None:
        """Consume agent.stream() events with live rich rendering."""
        if not self.console:
            result = self.agent.run(self.session_id, text)
            print(result["answer"])
            return
        for ev in self.agent.stream(self.session_id, text):
            t = ev["type"]
            if t == "message_start":
                continue  # silent iteration marker; tools/answer carry the signal
            elif t == "content_delta":
                self.console.print(ev["text"], end="", highlight=False)
            elif t == "tool_start":
                self.console.print(f"\n[yellow]◈ {ev['name']}(…)[/yellow]")
            elif t == "tool_end":
                obs = str(ev.get("observation", ""))[:160].replace("\n", " ")
                self.console.print(f"[green]✓ {ev['name']}[/green] [dim]{obs}[/dim]")
            elif t == "done":
                self.console.print()  # end the raw stream line
                self.out(ev["answer"], md=True)

    def loop(self) -> None:
        self.banner()
        while True:
            raw = self.read_line()
            if raw is None:
                self.out("\nBye.")
                break
            if not raw.strip():
                continue
            routed = self.handle_slash(raw)
            if routed is None:
                break
            if routed is True:
                continue
            text = raw.strip()
            self.echo_user(text)
            self.run_streaming(text)


def main() -> None:
    CLI().loop()


if __name__ == "__main__":
    main()
