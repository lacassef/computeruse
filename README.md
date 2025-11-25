# macos_cua_agent

Prototype implementation of the Computer Use Agent described in `project.md`. The agent runs a simple See-Think-Decide-Act-Verify loop on macOS Monterey, prioritizing safety and Retina-aware coordinate handling. By default it operates in a dry-run mode (no HID injection and no OpenRouter calls) so it can be inspected safely.
The cognitive core calls Claude Opus 4.5 via OpenRouter using a custom `computer` tool (OpenAI-style function calling), not Anthropic's official `computer_20xx` beta.

## Quickstart
- Python 3.11+ on macOS Monterey; grant Screen Recording and Accessibility permissions to your terminal once you enable HID control.
- Create a virtualenv and install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Add a `.env` with your OpenRouter key and any overrides you need (see below).
- Run the loop: `python -m macos_cua_agent.main` (first run uses placeholder frames and a stubbed cognitive core).
  - The CLI will prompt you for a task description. The agent runs until it halts (noop/limits) or you hit `Ctrl+C`, then you can enter another prompt or press Enter to quit.

## Configuration
Environment variables (place in `.env`):
- Computer control uses Claude Opus 4.5 via OpenRouter: set `USE_OPENROUTER=true`, `OPENROUTER_API_KEY`, and optionally `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`) and `OPENROUTER_MODEL` (default `anthropic/claude-opus-4.5`).
- `ENABLE_HID`=true to send real mouse/keyboard events via `pyautogui`.
- `ENABLE_SEMANTIC`=true to route actions through the semantic driver stub.
- `MAX_STEPS`, `MAX_FAILURES`, `MAX_WALL_CLOCK_SECONDS`, `VERIFY_DELAY_MS`, `LOG_LEVEL`, `ENCODE_FORMAT`.

Policy configuration lives in `macos_cua_agent/policies/safety_rules.yaml`. Extend block/allow/HITL lists to reflect your risk posture.

## Safety and Permissions
- HID control stays disabled until `ENABLE_HID=true`; actions are logged instead.
- OpenRouter calls stay disabled until `USE_OPENROUTER=true` and `OPENROUTER_API_KEY` is set.
- macOS TCC permissions (Screen Recording, Accessibility) must be granted to your terminal when you turn on HID control or screenshots.

## Testing
- Run regression tests: `pytest macos_cua_agent/tests`
- `test_coordinates.py` checks Retina conversion helpers. Benchmark tests are skipped until a macOS interactive harness is available.
