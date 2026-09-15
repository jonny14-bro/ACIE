from rich.console import Console
import os

def get_model_name(path):
    return os.path.basename(path).replace(".gguf", "")

console = Console()

def _has_repetition(text: str, window: int = 120, min_repeats: int = 3) -> bool:
    """
    Detect if `text` ends in a repeated phrase loop.
    Checks whether any phrase of `window` chars appears >= min_repeats times
    in the last 2000 chars of text.
    """
    tail = text[-2000:]
    if len(tail) < window * min_repeats:
        return False
    phrase = tail[-window:]
    return tail.count(phrase) >= min_repeats

def _stream_with_repeat_guard(
    model, prompt, max_tokens, temperature, model_type, model_path
) -> str:
    """
    Like stream_response but stops early if repetition is detected mid-stream.
    Uses build_prompt via a single user message so the model type's chat format
    is respected (prevents the model from echoing back instruction bullets).
    """
    from engine import get_model_name

    label = get_model_name(model_path) if model_path else "Assistant"

    if model_type == "qwen":
        stop_tokens = ["<|im_end|>", "<|im_start|>"]
    elif model_type == "deepseek":
        stop_tokens = ["<|user|>", "<|system|>"]
    elif model_type in ("mistral", "llama"):
        stop_tokens = ["### Instruction:", "### System:"]
    else:
        stop_tokens = []

    full_response = ""
    console.print(f"\n[bold green]{label}:[/bold green] ", end="")

    for chunk in model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        echo=False,
        stop=stop_tokens,
    ):
        token = chunk["choices"][0]["text"]
        token = token.replace("<think>", "").replace("</think>", "")
        full_response += token
        print(token, end="", flush=True)

        # Kill runaway repetition mid-stream
        if len(full_response) > 400 and _has_repetition(full_response):
            console.print("\n[bold red]⚠ Repetition detected — stopping generation[/bold red]")
            break

    print("\n")
    return full_response

def build_prompt(history, model_type="qwen"):
    if model_type == "qwen":
        prompt = ""
        for msg in history:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    elif model_type == "deepseek":
        prompt = ""
        for msg in history:
            prompt += f"<|{msg['role']}|>\n{msg['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    elif model_type in ["mistral", "llama"]:
        prompt = ""
        for msg in history:
            if msg["role"] == "system":
                prompt += f"### System:\n{msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"### Instruction:\n{msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"### Response:\n{msg['content']}\n\n"
        prompt += "### Response:\n"
        return prompt

    else:
        # generic fallback
        prompt = ""
        for msg in history:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n"
        prompt += "ASSISTANT:\n"
        return prompt


def stream_response(model, prompt, max_tokens=512, temperature=0.7,
                    model_type="qwen", model_path=None):
    full_response = ""
    label = get_model_name(model_path) if model_path else "Assistant"
    console.print(f"\n[bold green]{label}:[/bold green] ", end="")

    # FIX: Qwen needs both stop tokens to prevent runaway generation.
    # <|im_end|> stops at role boundary, <|im_start|> stops if model
    # tries to generate a new role token (e.g. starts a fake user turn).
    if model_type == "qwen":
        stop_tokens = ["<|im_end|>", "<|im_start|>"]
    elif model_type == "deepseek":
        stop_tokens = ["<|user|>", "<|system|>"]
    elif model_type in ("mistral", "llama"):
        stop_tokens = ["### Instruction:", "### System:"]
    else:
        stop_tokens = []

    for chunk in model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        echo=False,
        stop=stop_tokens,
    ):
        token = chunk["choices"][0]["text"]
        # Strip reasoning tags from thinking models (DeepSeek-R1, Qwen-thinking)
        # Note: this strips the TAGS only — full think-block filtering needs buffering
        token = token.replace("<think>", "").replace("</think>", "")
        full_response += token
        print(token, end="", flush=True)
    print("\n")
    return full_response


def run_single(model, prompt, max_tokens=512, temperature=0.7,
               model_type="qwen", model_path=None):
    history = [{"role": "user", "content": prompt}]
    formatted = build_prompt(history, model_type=model_type)
    return stream_response(model, formatted, max_tokens, temperature, model_type, model_path)


def run_chat(model, system_prompt=None, max_tokens=512, temperature=0.7,
             model_type="qwen", model_path=None, memory=None):
    """
    Plain chat loop (no ACIE routing).
    Used when --no-acie flag is passed, or for backward compatibility.
    """
    history = [{
        "role":    "system",
        "content": system_prompt or "You are a helpful AI assistant running locally.",
    }]

    model_name = get_model_name(model_path)
    console.print(f"[bold magenta]{model_name} ready. Type 'exit' to quit.[/bold magenta]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if not user_input:
            continue

        temp_history = list(history)

        if memory:
            ctx = memory.build_context(user_input)
            if ctx.strip():
                temp_history.append({"role": "system", "content": ctx})

        temp_history.append({"role": "user", "content": user_input})
        prompt   = build_prompt(temp_history, model_type=model_type)
        response = stream_response(model, prompt, max_tokens, temperature, model_type, model_path)

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})

        if memory:
            memory.add_message("user",      user_input)
            memory.add_message("assistant", response)

            combined = f"User: {user_input}\nAssistant: {response}"
            if len(combined) > 20:
                memory.add_global_memory(combined)

            memory.save_session()
            memory.save_global_memory()
