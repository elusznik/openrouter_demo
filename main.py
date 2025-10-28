import json

from openai import OpenAI


def to_dict(model):
    """Return the model data as a plain dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return model
    return getattr(model, "__dict__", {})


def extract_text_from_response(response):
    """Gather plain text content from a Responses API result."""
    if not response:
        return ""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts = []
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


def response_to_json(response):
    """Return a JSON string for any response-like object."""
    return json.dumps(to_dict(response), indent=2, sort_keys=True)


def extract_reasoning_text(response):
    """Collect reasoning text segments from a Responses API result."""
    if not response:
        return []

    data = to_dict(response)
    output_items = data.get("output", [])
    reasoning_chunks = []

    for item in output_items:
        if item.get("type") != "reasoning":
            continue

        for part in item.get("content", []) or []:
            text = part.get("text")
            if text:
                reasoning_chunks.append(text)

    return reasoning_chunks


def main():
    with open("api_key.txt", "r", encoding="utf-8") as file:
        api_key = file.read().strip()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    try:
        response = client.models.list()
    except Exception as error:
        print(f"Could not load model list: {error}")
        return

    free_models = []

    for item in response.data:
        data = to_dict(item)
        pricing = data.get("pricing", {})

        # Some entries return strings like "0"; convert them to float.
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

    stream_answer = input("Stream the response as it arrives? [Y/n]: ").strip().lower()
    use_stream = stream_answer != "n"

    messages = []
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

        latest_response = None

        if use_stream:
            try:
                print("\nModel reply (streaming):\n")
                reply_parts = []

                with client.responses.stream(
                    model=selected_id,
                    input=messages,
                    reasoning={"effort": "high","exclude": False},
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            delta = getattr(event, "delta", None)
                            piece = ""

                            if isinstance(delta, str):
                                piece = delta
                            else:
                                text_attr = getattr(delta, "text", None)
                                if text_attr:
                                    piece = text_attr
                                elif isinstance(delta, dict):
                                    piece = delta.get("text") or delta.get("value", "")

                            if piece:
                                print(piece, end="", flush=True)
                                reply_parts.append(piece)

                    final_response = stream.get_final_response()

                reply = "".join(reply_parts).strip()
                if not reply:
                    reply = extract_text_from_response(final_response)

                print("\n")
            except Exception as error:
                print(f"Could not start streaming from {selected_id}: {error}")
                messages.pop()  # Remove the user message so history stays clean.
                continue

            if not reply:
                print("The model did not include any text in the reply.")
                messages.pop()
                continue

            latest_response = final_response
        else:
            try:
                response = client.responses.create(
                    model=selected_id,
                    input=messages,
                    reasoning={"effort": "high"},
                )
            except Exception as error:
                print(f"Could not get a reply from {selected_id}: {error}")
                messages.pop()  # Remove the user message on failure.
                continue

            reply = extract_text_from_response(response)

            if not reply:
                print("The model did not include any text in the reply.")
                messages.pop()
                continue

            print("\nModel reply:\n")
            print(reply)

            latest_response = response

        if reply:
            messages.append({"role": "assistant", "content": reply})

        if latest_response:
            reasoning_chunks = extract_reasoning_text(latest_response)
            if reasoning_chunks:
                print("Reasoning trace:\n")
                for chunk in reasoning_chunks:
                    print(chunk)
                    print()

            show_json = input("Show raw JSON response? [y/N]: ").strip().lower()
            if show_json == "y":
                print("\nRaw response JSON:\n")
                print(response_to_json(latest_response))
                print()


if __name__ == "__main__":
    main()