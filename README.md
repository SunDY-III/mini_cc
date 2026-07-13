# Mini CC

一个用 Python 从零构建的极简编码代理（约 5000 行），灵感来自 Claude Code。

## 特性

- **Agent 循环**：自动工具调用、结果处理、迭代式任务完成
- **13 个内置工具**：读/写/编辑文件（mtime 保护）、搜索、Shell、网页抓取、技能系统、子代理、计划模式
- **双后端**：支持 Anthropic 和 OpenAI 兼容 API
- **流式输出**：支持流式工具执行的实时输出
- **并行执行**：只读工具自动并发执行（2-3 倍加速）
- **4 层上下文压缩**：预算截断 → 陈旧裁剪 → 微压缩 → 自动压缩
- **权限系统**：5 种模式 + 声明式规则 + 危险命令检测
- **记忆系统**：4 种类型的文件记忆，支持语义召回
- **技能系统**：`.claude/skills/` 目录加载，支持内联和分支模式
- **多代理**：子代理 fork-return 模式，支持自定义代理类型
- **MCP 集成**：通过 stdio 的 JSON-RPC 外部工具服务器
- **预算控制**：成本上限和轮次上限

## 安装

```bash
pip install -e .
```

## 配置

通过环境变量设置 API 凭证：

```bash
# Anthropic 格式（推荐）
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# 或 OpenAI 兼容格式
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

参考 `.env.example` 获取更多信息。

## 使用

```bash
# 交互式 REPL 模式
mini-cc

# 单次执行模式
mini-cc "修复 src/app.ts 中的 bug"

# 附加选项
mini-cc --yolo "运行所有测试并修复失败项"
mini-cc --plan "如何重构这个模块？"
mini-cc --model gpt-4o "你好"
mini-cc --max-cost 0.50 --max-turns 20 "实现功能 X"
mini-cc --resume
```

### REPL 命令

| 命令 | 描述 |
|---------|------|
| `/clear` | 清除对话历史 |
| `/plan` | 切换计划模式 |
| `/cost` | 显示 Token 用量和费用 |
| `/compact` | 手动压缩对话 |
| `/goal <条件>` | 持续执行直到条件满足 |
| `/loop [间隔] <提示>` | 按间隔或自定节奏重复执行 |
| `/memory` | 列出已保存的记忆 |
| `/skills` | 列出可用的技能 |
| `/<技能名>` | 调用技能 |

## 权限模式

| 标志 | 模式 | 描述 |
|------|------|------|
| *(默认)* | default | 确认危险操作 |
| `--yolo` | bypassPermissions | 自动批准所有操作 |
| `--plan` | plan | 只读计划模式 |
| `--accept-edits` | acceptEdits | 自动批准文件编辑 |
| `--dont-ask` | dontAsk | 自动拒绝所有确认 |
| `--auto` | auto | LLM 分类器判断操作 |

## 许可

MIT
