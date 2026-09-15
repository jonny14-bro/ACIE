"""
PyInfer — ACIE Edition
Run from your project directory:

    python pyinfer.py
    python pyinfer.py --model path/to/model.gguf
    python pyinfer.py --model fast.gguf --refiner big.gguf
    python pyinfer.py --setup          (re-run hardware setup)
    python pyinfer.py --no-acie        (plain chat, no routing)
    python pyinfer.py --prompt "..."   (single-shot)
"""

import os, sys, json, subprocess, psutil, argparse
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel
from rich.prompt  import Prompt, Confirm

from loader import load_model
from memory import MemorySystem
from acie   import run_acie_chat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import run_chat, run_single

console = Console()
PROFILE = Path(__file__).parent / "profile.json"


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect() -> dict:
    ram_gb   = round(psutil.virtual_memory().total / 1024**3, 1)
    vram_gb  = 0.0
    gpu_name = "No GPU"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", stderr=subprocess.DEVNULL,
        )
        parts    = out.strip().split(",")
        gpu_name = parts[0].strip()
        vram_gb  = round(int(parts[1].strip()) / 1024, 1)
    except Exception:
        pass
    free_gb = 0.0
    try:
        free_gb = round(psutil.disk_usage(Path.home()).free / 1024**3, 1)
    except Exception:
        pass
    return {"gpu": gpu_name, "vram": vram_gb, "ram": ram_gb, "disk_free": free_gb}


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

def _optimise(hw: dict) -> dict:
    vram = hw["vram"]
    ram  = hw["ram"]

    layers = {
        "9b-q8":  [(12,40),(8,16),(6,10),(4,6),(0,0)],
        "9b-q4":  [(12,80),(8,36),(6,24),(4,18),(0,0)],
        "7b-q4":  [(8,40),(6,32),(4,24),(2,12),(0,0)],
        "14b-q4": [(12,48),(8,20),(6,12),(4,8),(0,0)],
    }

    def pick_layers(key):
        for thr, n in layers[key]:
            if vram >= thr:
                return n
        return 0

    if   vram >= 8 or ram >= 24: ctx, batch = 4096, 512
    elif vram >= 4 or ram >= 12: ctx, batch = 2048, 256
    else:                        ctx, batch = 1024, 128

    if   vram >= 12 or (vram >= 8 and ram >= 16): rec_q, rec_m = "Q8_0",   "9B"
    elif vram >= 6  or (vram >= 4 and ram >= 12): rec_q, rec_m = "Q4_K_M", "9B"
    elif vram >= 4  or ram >= 12:                 rec_q, rec_m = "Q4_K_M", "7B"
    elif ram  >= 8:                               rec_q, rec_m = "Q4_K_S", "7B"
    else:                                         rec_q, rec_m = "Q2_K",   "7B"

    if   vram >= 8:  tier = "High Performance (GPU)"
    elif vram >= 4:  tier = "Mid-range (GPU + CPU offload)"
    elif vram >  0:  tier = "Low VRAM (minimal GPU assist)"
    elif ram  >= 16: tier = "CPU-only (adequate RAM)"
    else:            tier = "CPU-only (limited RAM)"

    cpu_cores = max(4, (psutil.cpu_count(logical=False) or 4) - 1)

    return {
        "tier_label":          tier,
        "gpu_layers_table":    {k: [(t, n) for t, n in layers[k]] for k in layers},
        "default_context":     ctx,
        "default_batch":       batch,
        "cpu_threads":         cpu_cores,
        "default_max_tokens":  512 if ram >= 8 else 256,
        "default_temperature": 0.7,
        "use_mmap":            True,
        "use_mlock":           False,  # mlock is unsupported/unstable on Windows; disable always
        "recommended_quant":   rec_q,
        "recommended_model_size": rec_m,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup() -> dict:
    console.print(Panel.fit(
        "[bold cyan]PyInfer[/bold cyan] · hardware setup\n"
        "[dim]Detecting your hardware — confirm or correct below[/dim]",
        border_style="cyan",
    ))

    hw = _detect()
    console.print(f"\n  Auto-detected:")
    console.print(f"    GPU  : [green]{hw['gpu']}[/green]")
    console.print(f"    VRAM : [green]{hw['vram']} GB[/green]")
    console.print(f"    RAM  : [green]{hw['ram']} GB[/green]")
    console.print(f"    Disk : [green]{hw['disk_free']} GB free[/green]\n")

    if not Confirm.ask("  Looks right?", default=True):
        gpu_input = Prompt.ask("  GPU name (or 'none')", default=hw["gpu"])
        if gpu_input.lower() == "none":
            hw["gpu"]  = "No GPU"
            hw["vram"] = 0.0
        else:
            hw["gpu"]  = gpu_input
            hw["vram"] = float(Prompt.ask("  VRAM GB", default=str(hw["vram"])))
        hw["ram"] = float(Prompt.ask("  RAM GB", default=str(hw["ram"])))

    default_model_dir = str(Path.home() / "models")
    model_dir = Prompt.ask("\n  Where do you store GGUF models?", default=default_model_dir)

    # Remap keys to profile.json schema
    hw_profile = {
        "gpu_name": hw["gpu"],
        "vram_gb":  hw["vram"],
        "ram_gb":   hw["ram"],
    }
    opt = _optimise(hw)

    profile = {"hardware": hw_profile, "model_dir": model_dir, "optimizations": opt}
    PROFILE.write_text(json.dumps(profile, indent=2))

    console.print(f"\n  [bold green]✓[/bold green]  Profile saved → [dim]{PROFILE}[/dim]")
    console.print(f"  Tier  : [bold]{opt['tier_label']}[/bold]")
    console.print(f"  Rec.  : [cyan]{opt['recommended_model_size']} {opt['recommended_quant']}[/cyan]\n")
    return profile


def load_profile() -> dict:
    return json.loads(PROFILE.read_text())


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_model_type(path: str) -> str:
    n = path.lower()
    if "deepseek" in n: return "deepseek"
    if "qwen"     in n: return "qwen"
    if "mistral"  in n: return "mistral"
    if "llama"    in n: return "llama"
    return "generic"


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(prog="pyinfer",
                                 description="PyInfer ACIE — Local LLM Inference Engine")
    ap.add_argument("model",            nargs="?",  help="Path to GGUF file")
    ap.add_argument("--refiner",  "-r",             help="Path to larger GGUF for confidence fallback")
    ap.add_argument("--setup",    "-S", action="store_true", help="Re-run hardware setup")
    ap.add_argument("--no-acie",        action="store_true", help="Use plain chat (no routing)")
    ap.add_argument("--prompt",   "-p", help="Single-shot prompt")
    ap.add_argument("--system",   "-s", help="System prompt override")
    ap.add_argument("--gpu-layers",     type=int,   default=None)
    ap.add_argument("--ctx",            type=int,   default=None)
    ap.add_argument("--max-tokens",     type=int,   default=None)
    ap.add_argument("--temperature",    type=float, default=None)
    ap.add_argument("--no-memory",      action="store_true", help="Disable memory system")
    args = ap.parse_args()

    console.print(Panel.fit(
        "[bold cyan]PyInfer[/bold cyan] — ACIE Edition",
        border_style="cyan",
    ))

    if args.setup:
        setup()
        return

    if not PROFILE.exists():
        console.print("[dim]No profile.json found — running setup…[/dim]\n")
        profile = setup()
    else:
        profile = load_profile()
        opt = profile["optimizations"]
        hw  = profile["hardware"]
        console.print(f"[dim]Profile loaded · {opt['tier_label']}[/dim]")
        console.print(f"[dim]GPU {hw['gpu_name']}  ·  "
                      f"{hw['vram_gb']}GB VRAM  ·  {hw['ram_gb']}GB RAM[/dim]\n")

    # ── Resolve model path ────────────────────────────────────────────────────
    model_path = args.model
    if not model_path:
        model_dir  = profile.get("model_dir", str(Path.home() / "models"))
        model_path = Prompt.ask("  Model path", default=model_dir)

    if not os.path.isfile(model_path):
        console.print(f"[red]✗  File not found:[/red] {model_path}")
        sys.exit(1)

    # ── Load primary model ────────────────────────────────────────────────────
    model = load_model(model_path, profile, args.gpu_layers, args.ctx)
    mtype = detect_model_type(model_path)
    opt   = profile["optimizations"]
    ctx   = args.ctx or opt["default_context"]
    max_tokens  = args.max_tokens or min(3072, ctx - 300)
    temperature = args.temperature or opt["default_temperature"]

    # ── Load refiner model (optional) ────────────────────────────────────────
    refiner_model = None
    if args.refiner:
        if not os.path.isfile(args.refiner):
            console.print(f"[yellow]⚠  Refiner not found, skipping: {args.refiner}[/yellow]")
        else:
            console.print(f"[dim]Loading refiner: {args.refiner}[/dim]")
            refiner_model = load_model(args.refiner, profile, args.gpu_layers, args.ctx)

    # ── Memory ────────────────────────────────────────────────────────────────
    memory = None if args.no_memory else MemorySystem()

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.prompt:
        run_single(model, args.prompt, max_tokens, temperature, mtype, model_path)
    elif args.no_acie:
        run_chat(model, args.system, max_tokens, temperature, mtype, model_path, memory)
    else:
        run_acie_chat(
            model         = model,
            model_path    = model_path,
            model_type    = mtype,
            max_tokens    = max_tokens,
            temperature   = temperature,
            system_prompt = args.system,
            memory        = memory,
            refiner_model = refiner_model,
            refiner_path  = args.refiner,
        )


if __name__ == "__main__":
    main()
