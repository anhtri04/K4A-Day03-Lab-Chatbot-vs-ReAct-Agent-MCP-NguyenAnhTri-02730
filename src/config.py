"""Config loader. Supports OpenAI + any OpenAI-compatible base_url (e.g. DeepSeek)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # minimal fallback if dotenv not installed

    def load_dotenv(*a, **kw):  # type: ignore
        return False


load_dotenv()

THIS_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    workspace_root: Path = THIS_DIR.parent  # sandbox root for file tools (repo root by default)
    db_path: Path = THIS_DIR / "agent.db"
    memory_path: Path = THIS_DIR / "MEMORY.md"
    max_iterations: int = 8
    short_term_window: int = 20
    auto_confirm: bool = False  # --yes / AUTO_CONFIRM=1 skips y/n gate (CI)
    # -- bash tool ------------------------------------------------------
    bash_timeout: int = 30  # default per-command timeout (BASH_TIMEOUT, 1-300)
    # -- full GitHub MCP (upstream server via npx stdio, see mcp_tools.py) --
    github_token: str = ""  # GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_TOKEN
    mcp_github_enabled: bool = True  # MCP_GITHUB_ENABLED=0 to disable
    mcp_tool_timeout: int = 60  # per MCP list/call, covers cold npx boot

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @classmethod
    def from_env(cls) -> "Config":
        def _path(key: str, default: Path) -> Path:
            raw = os.getenv(key, "").strip()
            if not raw:
                return default
            p = Path(raw).expanduser()
            return p if p.is_absolute() else (Path.cwd() / p).resolve()

        workspace = _path("WORKSPACE_ROOT", Path.cwd().resolve())
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1",
            model=os.getenv("MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            workspace_root=workspace,
            db_path=_path("DB_PATH", THIS_DIR / "agent.db"),
            memory_path=_path("MEMORY_PATH", THIS_DIR / "MEMORY.md"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "8") or 8),
            short_term_window=20,
            auto_confirm=os.getenv("AUTO_CONFIRM", "").strip().lower() in ("1", "true", "yes", "y")
            or "--yes" in sys.argv,
            bash_timeout=max(1, min(int(os.getenv("BASH_TIMEOUT", "30") or 30), 300)),
            github_token=(
                os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
                or os.getenv("GITHUB_TOKEN", "").strip()
            ),
            mcp_github_enabled=os.getenv("MCP_GITHUB_ENABLED", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            mcp_tool_timeout=int(os.getenv("MCP_TOOL_TIMEOUT", "60") or 60),
        )
