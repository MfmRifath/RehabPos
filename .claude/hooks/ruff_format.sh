#!/usr/bin/env bash
# PostToolUse hook: format an edited .py file with ruff. Never blocks the edit.
f=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null)
if [[ "$f" == *.py && -f "$f" ]]; then
  cd "${CLAUDE_PROJECT_DIR:-.}" && uv run --quiet ruff format "$f" >/dev/null 2>&1
fi
exit 0
