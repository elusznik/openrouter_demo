# OpenRouter Zero-Cost Model Demo
- Lists zero-cost OpenRouter models via the OpenAI Python SDK and the OpenRouter-compatible API.
- Prompts you to pick a model number, then chat back and forth using the OpenRouter-compatible Responses API. Reasoning defaults are pre-set in code.

## Setup
- Ensure Python 3.14+ (as specified in `pyproject.toml`).
- Install dependencies using your preferred tool (e.g., `uv sync` or `pip install -r requirements.txt`).
- Set your OpenRouter API key as an environment variable named `OPENROUTER_API_KEY`. Alternatively, save it in a file named `api_key.txt` in the same directory as `main.py`.

## Usage
- Run `python main.py` or `uv run main.py`.
- Choose a listed zero-cost model by number.
- Enter your message and keep chatting until you type `exit`.
- Read the model’s responses in the console once the assistant finishes its turn.

## Notes
- Pricing data comes from `client.models.list()`; the script filters where both prompt and completion pricing are `0`.
- Every turn reuses the growing `messages` list so the model keeps conversational context until you type `exit`.
- Reasoning effort is forced to `high` in code so reasoning-capable models use their full chain-of-thought capacity; other sampling parameters rely on OpenRouter defaults.
- Messaging is sent via `client.responses.create` per the OpenRouter/OpenAI Responses API guidelines.
- References: [OpenRouter docs](https://openrouter.ai/docs), [models API](https://openrouter.ai/api/v1/models), [OpenAI SDK docs](https://platform.openai.com/docs/api-reference/models).