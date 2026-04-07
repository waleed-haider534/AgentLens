import ollama


def get_local_models() -> list:
    try:
        response = ollama.list()
        result = []

        for model in response.models:
            name = model.model
            size_bytes = model.size or 0
            size_gb = round(size_bytes / (1024 ** 3), 2)
            details = model.details

            # Skip cloud models (they have no real size)
            if size_bytes < 1000000:
                continue

            result.append({
                "name": name,
                "size": f"{size_gb} GB",
                "family": details.family or "Unknown",
                "parameters": details.parameter_size or "Unknown",
                "quantization": details.quantization_level or "Unknown",
            })

        return result

    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return []


def test_local_model(model_name: str, test_prompt: str) -> str:
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": test_prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error testing model: {e}"


def is_ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False