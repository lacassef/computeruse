# macos_cua_agent

Prototype implementation of the Computer Use Agent described in `project.md`. The agent runs a simple See-Think-Decide-Act-Verify loop on macOS Monterey, prioritizing safety and Retina-aware coordinate handling. By default it operates in a dry-run mode (no HID injection and no Anthropic calls) so it can be inspected safely.

## Quickstart
- Python 3.11+ on macOS Monterey; grant Screen Recording and Accessibility permissions to your terminal once you enable HID control.
- Create a virtualenv and install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Add a `.env` with any overrides you need (see below).
- Run the loop: `python -m macos_cua_agent.main` (first run uses placeholder frames and a stubbed cognitive core).

## Configuration
Environment variables (place in `.env`):
- `USE_ANTHROPIC`=true to call Anthropic; requires `ANTHROPIC_API_KEY`.
- `ANTHROPIC_MODEL` (default `claude-3-5-sonnet-20240620`), `ANTHROPIC_TOOL_TYPE` (default `computer_20241022`), `ANTHROPIC_BETA_HEADER` (default `computer-use-2025-01-24`).
- `ENABLE_HID`=true to send real mouse/keyboard events via `pyautogui`.
- `ENABLE_SEMANTIC`=true to route actions through the semantic driver stub.
- `MAX_STEPS`, `MAX_FAILURES`, `MAX_WALL_CLOCK_SECONDS`, `VERIFY_DELAY_MS`, `LOG_LEVEL`, `ENCODE_FORMAT`.

Policy configuration lives in `macos_cua_agent/policies/safety_rules.yaml`. Extend block/allow/HITL lists to reflect your risk posture.

## Safety and Permissions
- HID control stays disabled until `ENABLE_HID=true`; actions are logged instead.
- Anthropic calls stay disabled until `USE_ANTHROPIC=true` and `ANTHROPIC_API_KEY` is set.
- macOS TCC permissions (Screen Recording, Accessibility) must be granted to your terminal when you turn on HID control or screenshots.

## Testing
- Run regression tests: `pytest macos_cua_agent/tests`
- `test_coordinates.py` checks Retina conversion helpers. Benchmark tests are skipped until a macOS interactive harness is available.

