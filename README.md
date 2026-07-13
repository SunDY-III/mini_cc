# Mini CC

A minimal coding agent built from scratch in Python (~5000 lines), inspired by Claude Code.

## Features

- **Agent Loop**: Automatic tool calling, result processing, iterative task completion
- **13 Built-in Tools**: read/write/edit files (mtime-protected), search, shell, web fetch, skills, sub-agents, plan mode
- **Dual Backend**: Supports Anthropic and OpenAI-compatible APIs
- **Streaming**: Real-time output with streaming tool execution
- **Parallel Execution**: Read-only tools auto-concurrent (2-3x speedup)
- **4-Layer Context Compression**: Budget truncation → stale snip → microcompact → auto-compact
- **Permission System**: 5 modes + declarative rules + dangerous command detection
- **Memory System**: 4-type file-based memory with semantic recall
- **Skill System**: `.claude/skills/` directory loading, inline and fork modes
- **Multi-Agent**: Sub-agent fork-return pattern with custom agent types
- **MCP Integration**: JSON-RPC over stdio for external tool servers
- **Budget Control**: Cost limits and turn limits

## Installation

```bash
pip install -e .
```

## Configuration

Set API credentials via environment variables:

```bash
# Anthropic format (recommended)
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Or OpenAI-compatible format
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

See `.env.example` for reference.

## Usage

```bash
# Interactive REPL mode
mini-cc

# One-shot mode
mini-cc "fix the bug in src/app.ts"

# With options
mini-cc --yolo "run all tests and fix failures"
mini-cc --plan "how would you refactor this?"
mini-cc --model gpt-4o "hello"
mini-cc --max-cost 0.50 --max-turns 20 "implement feature X"
mini-cc --resume
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/plan` | Toggle plan mode |
| `/cost` | Show token usage and cost |
| `/compact` | Manually compact conversation |
| `/goal <condition>` | Pursue a goal until met |
| `/loop [interval] <prompt>` | Re-run on interval or self-paced |
| `/memory` | List saved memories |
| `/skills` | List available skills |
| `/<skill-name>` | Invoke a skill |

## Permission Modes

| Flag | Mode | Description |
|------|------|-------------|
| *(default)* | default | Confirm dangerous actions |
| `--yolo` | bypassPermissions | Auto-approve everything |
| `--plan` | plan | Read-only planning |
| `--accept-edits` | acceptEdits | Auto-approve file edits |
| `--dont-ask` | dontAsk | Auto-deny confirmations |
| `--auto` | auto | LLM classifier judges actions |

## License

MIT
