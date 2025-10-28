# OpenRouter Zero-Cost Model Demo
- Lists zero-cost OpenRouter models via the OpenAI Python SDK and the OpenRouter-compatible API.
- Prompts you to pick a model number, chat back and forth, and optionally stream the live reply using the OpenRouter-compatible Responses API. Reasoning defaults are pre-set in code.

## Setup
- Ensure Python 3.11+ (OpenRouter recommends 3.11; pyproject uses 3.14).
- Install dependencies using your preferred tool (e.g. `uv sync` or `pip install openai`).
- Save your OpenRouter API key into `api_key.txt` beside `main.py`.

## Usage
- Run `python main.py` or `uv run main.py`.
- Choose a listed zero-cost model by number.
- Enter your message, decide on streaming (`Y`) or non-streaming (`n`), and keep chatting until you type `exit`.
- Read the model’s responses in the console (streamed line-by-line or printed all at once).

## Notes
- Pricing data comes from `client.models.list()`; the script filters where both prompt and completion pricing are `0`.
- Streaming uses `stream=True` plus `print(piece, end="", flush=True)` for live output.
- Every turn reuses the growing `messages` list so the model keeps conversational context until you type `exit`.
- Reasoning effort is forced to `high` in code so reasoning-capable models use their full chain-of-thought capacity; other sampling parameters rely on OpenRouter defaults.
- Messaging is sent via `client.responses.create` / `client.responses.stream` (streaming) per the OpenRouter/OpenAI Responses API guidelines.
- References: [OpenRouter docs](https://openrouter.ai/docs), [models API](https://openrouter.ai/api/v1/models), [OpenAI SDK docs](https://platform.openai.com/docs/api-reference/models).