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

        if use_stream:
            try:
                stream = client.chat.completions.create(
                    model=selected_id,
                    messages=messages,
                    stream=True,
                )
            except Exception as error:
                print(f"Could not start streaming from {selected_id}: {error}")
                messages.pop()  # Remove the user message so history stays clean.
                continue

            print("\nModel reply (streaming):\n")
            reply_parts = []

            for chunk in stream:
                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)

                if isinstance(delta, dict):
                    piece = delta.get("content") or ""
                else:
                    piece = getattr(delta, "content", "") or ""

                if piece:
                    print(piece, end="", flush=True)
                    reply_parts.append(piece)

            reply = "".join(reply_parts).strip()
            print("\n")
        else:
            try:
                completion = client.chat.completions.create(
                    model=selected_id,
                    messages=messages,
                )
            except Exception as error:
                print(f"Could not get a reply from {selected_id}: {error}")
                messages.pop()  # Remove the user message on failure.
                continue

            try:
                first_choice = completion.choices[0]
            except (AttributeError, IndexError):
                print("The model returned an empty response.")
                messages.pop()
                continue

            message = getattr(first_choice, "message", None)

            if isinstance(message, dict):
                reply = message.get("content", "")
            else:
                reply = getattr(message, "content", "")

            if not reply:
                print("The model did not include any text in the reply.")
                messages.pop()
                continue

            print("\nModel reply:\n")
            print(reply)

        if reply:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()