# Camaron

Camaron is a local-first knowledge chatbot.  It ingests text files from a
`data/` directory, keeps a lightweight capsule memory, and answers questions
about those documents.  It also understands workspace-file queries and has a
basic coding lane.

## Quick start

### Terminal (CLI) mode

```bash
python camaron.py
```

### GUI mode (requires PyQt6)

```bash
python camaron.py --gui
```

## Chat commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show memory status |
| `/sources` | List loaded source files |
| `/rescan` | Rescan `data/` and refresh memory |
| `/coding` | Show coding-lane status |
| `/workspace` | Show workspace-lane status |
| `/quit` | Exit |

## How it works

1. Place `.txt`, `.md`, `.json`, or `.csv` files in a `data/` folder next to
   `camaron.py`.
2. Start Camaron.  It ingests those files into a capsule memory on first run.
3. Ask questions in natural language.  Camaron retrieves the most relevant
   capsules and composes a grounded response.

Conversation history is saved to `brain_state/doku_dialogue.jsonl`.
