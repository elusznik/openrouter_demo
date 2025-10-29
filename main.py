import json
from typing import Any, Dict, List

from openai import OpenAI


def to_dict(model: Any) -> Dict[str, Any]:
    """Return the model data as a plain dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return model
    return getattr(model, "__dict__", {})


def extract_text_from_response(response: Any) -> str:
    """Gather plain text content from a Responses API result."""
    if not response:
        return ""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: List[str] = []
    output_items = getattr(response, "output", None) or []

    for item in output_items:
        content_list = getattr(item, "content", None) or []

        for part in content_list:
            text_value = getattr(part, "text", None)
            if text_value:
                parts.append(text_value)
                continue

            if isinstance(part, dict) and part.get("type") == "output_text":
                part_text = part.get("text")
                if part_text:
                    parts.append(part_text)

    return "".join(parts).strip()


def response_to_json(response: Any) -> str:
    """Return a JSON string for any response-like object."""
    return json.dumps(to_dict(response), indent=2, sort_keys=True)


def extract_reasoning_text(response: Any) -> List[str]:
    """Collect reasoning text segments from a Responses API result."""
    if not response:
        return []

    data = to_dict(response)
    output_items = data.get("output", [])
    reasoning_chunks: List[str] = []

    for item in output_items:
        if item.get("type") != "reasoning":
            continue

        for part in item.get("content", []) or []:
            text = part.get("text")
            if text:
                reasoning_chunks.append(text)

    return reasoning_chunks


def extract_reasoning_summary(response: Any) -> List[str]:
    """Collect reasoning summary text segments from a Responses API result."""
    if not response:
        return []

    data = to_dict(response)
    summary_chunks: List[str] = []

    # Primary location: reasoning output items include a "summary" list.
    for item in data.get("output", []) or []:
        if item.get("type") != "reasoning":
            continue

        for entry in item.get("summary", []) or []:
            text = entry.get("text") if isinstance(entry, dict) else None
            if text:
                summary_chunks.append(text)

    # Some providers may also populate a top-level reasoning.summary field.
    for entry in data.get("reasoning", {}).get("summary", []) or []:
        text = entry.get("text") if isinstance(entry, dict) else None
        if text:
            summary_chunks.append(text)

    return summary_chunks


def main() -> None:
    with open("api_key.txt", "r", encoding="utf-8") as file:
        api_key = file.read().strip()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    try:
        response = client.models.list()
    except Exception as error:  # noqa: BLE001 - provide user-friendly output
        print(f"Could not load model list: {error}")
        return

    free_models: List[Dict[str, Any]] = []

    for item in response.data:
        data = to_dict(item)
        pricing = data.get("pricing", {})

        try:
            prompt_cost = float(pricing.get("prompt", 0))
            completion_cost = float(pricing.get("completion", 0))
        except (TypeError, ValueError):
            continue

        if prompt_cost == 0 and completion_cost == 0:
            free_models.append(data)

    if not free_models:
        print("No zero-cost models available right now.")
        return

    print("Zero-cost models on OpenRouter:")
    for index, model in enumerate(free_models, start=1):
        model_id = model.get("id", "unknown")
        name = model.get("name") or model.get("display_name") or model_id
        print(f"{index}. {name} ({model_id})")

    choice = input("\nPick a model number (or press Enter to exit): ").strip()

    if not choice:
        print("No model selected.")
        return

    try:
        choice_index = int(choice) - 1
    except ValueError:
        print("Please enter a number from the list.")
        return

    if choice_index < 0 or choice_index >= len(free_models):
        print("That number is outside the list.")
        return

    selected_model = free_models[choice_index]
    selected_id = selected_model.get("id")

    print(f"\nSelected {selected_id}\n")

    messages: List[Dict[str, str]] = []
    print("Type your message (or 'exit' to quit).")

    while True:
        question = input("You: ").strip()

        if not question:
            print("Please enter some text or 'exit'.")
            continue

        if question.lower() in {"exit", "quit"}:
            print("Conversation ended.")
            break

        messages.append({"role": "user", "content": question})

        try:
            response = client.responses.create(
                model=selected_id,
                input=messages,
                reasoning={"effort": "high", "generate_summary": True},
            )
        except Exception as error:  # noqa: BLE001 - provide user-friendly output
            print(f"Could not get a reply from {selected_id}: {error}")
            messages.pop()  # Remove the user message on failure.
            continue

        reasoning_chunks = extract_reasoning_text(response)

        if reasoning_chunks:
            # print("\nReasoning:\n")
            print("<think>")
            for chunk in reasoning_chunks:
                print(chunk)
            print("</think>")
            print()

        summary_chunks = extract_reasoning_summary(response)

        if summary_chunks:
            # print("Reasoning summary:\n")
            for chunk in summary_chunks:
                print(chunk)
                print()

        reply_text = extract_text_from_response(response)

        if not reply_text:
            print("The model did not include any text in the reply.")
            messages.pop()
            continue

        # print("\nModel reply:\n")
        print(reply_text)

        reasoning_text_for_history = ""
        summary_text_for_history = ""

        if reasoning_chunks:
            inner = "\n".join(reasoning_chunks).strip()
            if inner:
                reasoning_text_for_history = inner

        if summary_chunks:
            inner_summary = "\n".join(summary_chunks).strip()
            if inner_summary:
                summary_text_for_history = inner_summary

        show_json = input("Show raw JSON response? [y/N]: ").strip().lower()
        if show_json == "y":
            print("\nRaw response JSON:\n")
            print(response_to_json(response))
            print()

        if reasoning_text_for_history:
            messages.append(
                {
                    "role": "assistant",
                    "content": reasoning_text_for_history,
                }
            )
        if summary_text_for_history:
            messages.append(
                {
                    "role": "assistant",
                    "content": summary_text_for_history,
                }
            )
        messages.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    main()
