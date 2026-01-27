# 🍍 Made an MCP so I can manage Msty just by asking Claude

I'm not a developer, but I use Msty daily and got frustrated digging through settings and menus. Figured there had to be a better way.

So I teamed up with Claude to build this MCP. Now I just ask for what I want:

**"What models do I have?"** → Shows all my models across MLX, LLaMA.cpp, etc.

**"Which one's fastest for coding?"** → Tells me and explains why

**"Export my chats from last week"** → Done. Markdown, JSON, whatever.

**"How's my Msty looking?"** → Health check on everything

[GIF DEMO HERE]

Honestly the best part is I don't need to remember where anything is anymore. Just ask.

**What it does (42 tools):**
- Lists & benchmarks your models
- Exports/searches conversations
- Health checks & diagnostics
- Smart model recommendations
- Backup your configs

**To install:**
```bash
git clone https://github.com/DBSS/msty-admin-mcp.git
cd msty-admin-mcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then add to Claude Desktop config and restart.

**GitHub**: https://github.com/DBSS/msty-admin-mcp

Works on macOS with Msty 2.4.0+. Happy to help if anyone has questions! 🍍

---
*Built with Claude, based on original work by Pineapple*
