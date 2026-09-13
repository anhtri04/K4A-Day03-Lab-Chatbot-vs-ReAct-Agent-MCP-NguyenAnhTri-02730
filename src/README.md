# CLI Coding Agent

ReAct-style coding agent (template extension of Lab #3, `starter-code/`).

## Features
- **FS tools**: `langchain-community` `FileManagementToolkit` + native `edit_file`,
  `search_files`, `add_memory`; sandboxed to `WORKSPACE_ROOT`. Native fallback if langchain is absent.
- **Short-term memory**: `sqlite3` (`agent.db`), sliding window of 20 msgs/session (WAL).
- **Long-term memory**: `MEMORY.md` — read via `read_file`, write via `add_memory`.
- **LLM**: `openai` SDK with `OPENAI_BASE_URL` (OpenAI, DeepSeek, OpenRouter, Ollama...).
- **CLI**: `rich` output + `prompt_toolkit` history; slash commands.

## Setup
```bash
cp .env.example .env   # set OPENAI_API_KEY, OPENAI_BASE_URL, MODEL
pip install -r requirements.txt
python main.py
```

## Slash commands
`/new [title]` `/resume [id]` `/memory` `/tools` `/sessions` `/clear` `/help` `/quit`

## Layout
`config.py` config · `memory.py` memory · `tools.py` tools · `agent.py` tool loop · `main.py` REPL
