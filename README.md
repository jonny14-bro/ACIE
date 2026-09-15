# ACIE

# ACIE — Adaptive Coding Intelligence Engine


## 🛠️ Tech Stack

<div align="center">

### Core Language & Runtime

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

### AI & Machine Learning

<img src="https://img.shields.io/badge/llama.cpp-Local%20LLM-000000?style=for-the-badge" alt="llama.cpp"/>
<img src="https://img.shields.io/badge/Sentence--Transformers-Embeddings-FF6F00?style=for-the-badge" alt="Sentence Transformers"/>
<img src="https://img.shields.io/badge/FAISS-Vector%20Search-0468FF?style=for-the-badge" alt="FAISS"/>
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>

### System & Execution

<img src="https://img.shields.io/badge/Local%20Inference-Offline%20First-2E7D32?style=for-the-badge" alt="Local Inference"/>
<img src="https://img.shields.io/badge/Modular%20Architecture-Engineered-6A1B9A?style=for-the-badge" alt="Modular Architecture"/>

### Developer Tools

<img src="https://img.shields.io/badge/Git-Version%20Control-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git"/>
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
<img src="https://img.shields.io/badge/Linux-Supported-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux"/>

</div>

> A local-first, adaptive coding intelligence engine that routes requests, estimates task difficulty, retrieves relevant memory, generates multi-step code, detects semantic drift during generation, validates the assembled result, and performs targeted self-correction.

ACIE is designed around the idea that a coding assistant should do more than produce a single completion. Instead of treating every request identically, it builds a pipeline around **intent, difficulty, memory, generation control, validation, and recovery**.

---

## Overview

**ACIE (Adaptive Coding Intelligence Engine)** is a Python-based local AI coding system built around GGUF-compatible local language models through `llama-cpp-python`.

The main pipeline combines several cooperating components:

```text
User Request
     │
     ▼
┌───────────────┐
│ Input Router  │  Detect task mode, language, signals
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    PRIMA      │  Estimate difficulty + allocate resources
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Memory     │  Retrieve relevant previous knowledge
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Prompt Engine │  Construct context-aware prompt
└───────┬───────┘
        │
        ├──────────────────────┐
        ▼                      │
┌───────────────┐              │
│ Task Planner  │              │
└───────┬───────┘              │
        ▼                      │
┌───────────────┐              │
│ Step Executor │              │
└───────┬───────┘              │
        │                      │
        ▼                      │
┌───────────────┐              │
│    DELTA      │◄─────────────┘
│ Drift Monitor │
└───────┬───────┘
        │
        ▼
┌────────────────┐
│ Goal Checking  │
│ + Confidence   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Self-Correction│
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 3-Layer        │
│ Validator      │
└───────┬────────┘
        │
        ▼
   Final Result
```

The system is intentionally modular: routing, resource estimation, memory, generation, drift detection, execution, and validation are implemented as separate Python modules.

---

## Key Features

### 🧭 Intelligent task routing

ACIE analyzes the input and assigns a task mode such as:

- `generate`
- `debug`
- `explain`
- `refactor`
- `general`

The router also derives a memory domain and constructs a mode-specific system prompt.

### ⚙️ PRIMA adaptive resource allocation

**PRIMA — Predictive Resource & Intent Modulation Algorithm** performs pre-generation difficulty fingerprinting.

It combines multiple signals, including:

- syntactic density
- ambiguity
- domain novelty
- input length
- error complexity

These signals are combined into a difficulty score and used to derive a coupled resource vector containing:

- token budget
- temperature
- memory retrieval size

The goal is to avoid treating a simple question and a complex coding task as identical workloads.

### 🧠 Multi-domain memory

ACIE uses a persistent memory system backed by:

- Sentence Transformers embeddings
- FAISS indexes
- domain-specific storage
- quality-aware memory saving
- relevance-ranked retrieval

The memory system supports domains such as:

```text
coding
debug
general
coding_python
```

Code can also be stored and retrieved through the `CodeRetriever` abstraction.

Runtime memory is stored locally and is intentionally excluded from Git through `.gitignore`.

### 📋 Multi-step project planning

For larger coding tasks, ACIE can turn the request into a multi-step plan.

The project state tracks:

- modules
- pending tasks
- completed tasks
- execution order

The executor then generates modules step-by-step rather than treating an entire project as one undifferentiated completion.

### 🧩 Modular step execution

`executor.py` executes generated coding plans.

It includes safeguards such as:

- code extraction
- definition extraction
- semantic duplicate detection
- adaptive per-step generation limits
- repetition detection
- retry handling
- final module assembly

The current implementation retains the historical function name `execute_plan_v3` for compatibility, while the module itself is now named `executor.py`.

### 📡 DELTA semantic drift monitoring

**DELTA — Divergence-Estimated Layer Token Allocation** monitors generated output while it is streaming.

It uses embedding similarity and an EMA-based semantic distance signal to estimate whether generation is drifting away from the original intent.

When the configured drift conditions are reached, DELTA can:

1. stop the current generation,
2. construct a targeted correction prompt,
3. continue generation from a better semantic direction.

> **Important:** DELTA is a heuristic drift-detection mechanism. Semantic distance is not a formal correctness proof and should not be interpreted as one.

### 🎯 Goal-aware completion checking

ACIE extracts goals from the user's request and checks whether the generated response appears to satisfy them.

This gives the pipeline another signal besides raw generation completion.

### 📊 Confidence estimation

The router contains a post-generation confidence checker that classifies a response as:

```text
high
medium
low
```

The confidence signal can participate in the broader correction/refinement flow.

### 🔄 Self-correction

If the result is incomplete or does not satisfy the extracted goals, ACIE can invoke a targeted self-correction stage instead of immediately returning the first generation.

This creates a feedback loop:

```text
Generate
   ↓
Inspect
   ↓
Incomplete / weak?
   ↓ yes
Correct
   ↓
Inspect again
```

### 🛡️ Three-layer validation

`validator.py` performs post-generation validation at three levels.

#### Layer 1 — Integrity

Checks structural/code-level issues such as:

- missing imports
- undefined names across modules
- skipped plan steps
- broken initialization signatures
- other basic integrity problems

Static patches may be applied where appropriate.

#### Layer 2 — Integration

Checks whether generated modules actually fit together.

Examples include:

- method-name drift
- class wiring mismatches
- broken module-to-module calls
- inconsistent data flow

The system can request targeted regeneration for a broken module.

#### Layer 3 — Completeness

Audits the assembled project for issues such as:

- missing entry points
- unfinished/stubbed implementations
- skipped work
- incomplete generated functionality

The validator can use targeted gap filling where supported.

---

# Architecture

ACIE is organized as a pipeline rather than a single monolithic model call.

## 1. PyInfer

`pyinfer.py` is the primary command-line entry point.

It is responsible for:

- command-line argument handling
- hardware detection
- optimization/profile setup
- model type detection
- model selection
- starting ACIE or plain chat

The setup process can generate a local `profile.json` containing machine-specific runtime information.

That file is intentionally **not** part of the public repository.

The repository provides:

```text
config/profile.example.json
```

as a sanitized configuration example.

---

## 2. Router

`router.py` analyzes the user request.

Conceptually:

```text
Input
  │
  ├── language/signals
  ├── task characteristics
  └── intent clues
        │
        ▼
   RouteDecision
        │
        ├── mode
        ├── language
        └── memory domain
```

Supported task modes are represented by the `TaskMode` type.

The router also provides mode-specific system prompt construction and a response confidence heuristic.

---

## 3. PRIMA

`prima.py` calculates a difficulty fingerprint before generation.

```text
Input
 │
 ├── syntactic density
 ├── ambiguity
 ├── domain novelty
 ├── length hint
 └── error complexity
 │
 ▼
Difficulty Score
 │
 ▼
Resource Vector
 ├── tokens
 ├── temperature
 └── memory_k
```

This is intended to make resource allocation adaptive to the request.

---

## 4. Memory

`memory.py` implements the persistent retrieval layer.

The high-level flow is:

```text
Text
 │
 ▼
Sentence Transformer
 │
 ▼
Embedding
 │
 ▼
FAISS domain index
 │
 ▼
Ranked retrieval
 │
 ▼
Prompt context
```

Memory entries are quality-aware. Low-quality exchanges can be excluded from persistence while medium/high-quality exchanges can be retained.

---

## 5. Prompt and generation engine

`engine.py` contains core generation utilities, including:

- model-name extraction
- repetition detection
- prompt construction
- streaming helpers

It provides shared generation behavior used by ACIE and the executor.

---

## 6. Planner and executor

The planning/execution flow uses `builder.py`, `acie.py`, and `executor.py`.

A larger coding request can be transformed into a sequence such as:

```text
Step 1 → Generate module
Step 2 → Generate module
Step 3 → Generate module
Step 4 → Assemble project
```

The project state records module contents and task completion.

The executor contains retry and deduplication safeguards intended to reduce repeated or malformed modules.

---

## 7. DELTA

`delta.py` operates during streamed generation.

Conceptually:

```text
Prompt ───────────────► Local LLM
                           │
                           ▼
                      Token chunks
                           │
                           ▼
                    Chunk accumulator
                           │
                           ▼
                    Embedding signal
                           │
                           ▼
                    EMA drift estimate
                           │
                 ┌─────────┴─────────┐
                 │                   │
              Stable              Drift
                 │                   │
                 ▼                   ▼
             Continue          Correction prompt
```

The implementation uses cosine similarity/distance and an EMA-style signal.

This should be understood as a practical heuristic for detecting semantic divergence, not a mathematical guarantee of output correctness.

---

## 8. Goal and confidence loop

After generation, ACIE can evaluate:

```text
Response
   │
   ├── Goal extraction/checking
   ├── Completeness check
   └── Confidence estimate
          │
          ▼
   Self-correction if needed
```

This adds post-generation reasoning to the pipeline.

---

## 9. Validator

Finally, the generated project is passed through the validator.

```text
Generated Project
       │
       ▼
Integrity
       │
       ▼
Integration
       │
       ▼
Completeness
       │
       ▼
Validated / Repaired Result
```

---

# Repository Structure

```text
ACIE/
├── acie.py
├── builder.py
├── config/
│   └── profile.example.json
├── delta.py
├── docs/
├── engine.py
├── executor.py
├── loader.py
├── memory.py
├── prima.py
├── pyinfer.py
├── requirements.txt
├── retriever.py
├── router.py
├── tests/
├── validator.py
├── .gitignore
└── README.md
```

## Module Responsibilities

| File | Responsibility |
|---|---|
| `pyinfer.py` | CLI entry point, setup, hardware/profile handling, model selection |
| `acie.py` | Main ACIE orchestration pipeline |
| `router.py` | Input analysis, task routing, memory-domain selection, confidence |
| `prima.py` | Difficulty estimation and adaptive resource allocation |
| `memory.py` | Persistent multi-domain semantic memory |
| `retriever.py` | Code-specific memory retrieval helper |
| `engine.py` | Prompt/generation utilities and repetition protection |
| `builder.py` | Project/task state management |
| `executor.py` | Multi-step code generation and module assembly |
| `delta.py` | Streaming semantic drift estimation and correction |
| `validator.py` | Three-layer post-generation validation and repair |
| `loader.py` | Profile-aware GGUF model loading |
| `config/profile.example.json` | Sanitized example runtime profile |
| `requirements.txt` | Python dependencies |

---

# Requirements

ACIE currently depends on:

```text
sentence-transformers>=2.7.0
huggingface_hub>=0.23.0
faiss-cpu
numpy
rich
psutil
llama-cpp-python
```

You also need a compatible local **GGUF language model**.

ACIE does not ship model weights in this repository.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ACIE.git
cd ACIE
```

Replace `<your-username>` with the GitHub account that owns the repository.

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### `llama-cpp-python`

`llama-cpp-python` can require platform-specific build/runtime configuration, especially when GPU acceleration is desired.

If the default installation does not work on your machine, consult the package's installation instructions and select the build appropriate for your CPU/GPU environment.

---

# Model Setup

ACIE expects a local GGUF model.

Example:

```text
models/
└── your-model.gguf
```

Model files are intentionally excluded from Git by the repository's `.gitignore`.

You can provide the model directly:

```bash
python pyinfer.py models/your-model.gguf
```

The setup flow can also be used to generate the local hardware/runtime profile:

```bash
python pyinfer.py --setup
```

The generated local profile is:

```text
profile.json
```

It is intentionally ignored by Git because it can contain machine-specific hardware and path information.

---

# Usage

## Start ACIE

```bash
python pyinfer.py path/to/model.gguf
```

## Single-shot prompt

```bash
python pyinfer.py --model path/to/model.gguf --prompt "Explain binary search in Python"
```

The CLI also accepts the model as the positional argument:

```bash
python pyinfer.py path/to/model.gguf --prompt "Explain binary search in Python"
```

## Use a larger refiner model

```bash
python pyinfer.py \
  --model path/to/fast-model.gguf \
  --refiner path/to/larger-model.gguf
```

The refiner option is intended for confidence/refinement fallback scenarios.

## Plain chat without ACIE routing

```bash
python pyinfer.py path/to/model.gguf --no-acie
```

This bypasses the ACIE routing pipeline for a simpler chat mode.

## Override the system prompt

```bash
python pyinfer.py \
  path/to/model.gguf \
  --system "You are an expert Python engineer."
```

## Disable memory

```bash
python pyinfer.py \
  path/to/model.gguf \
  --no-memory
```

## Runtime controls

The CLI exposes controls for:

```text
--gpu-layers
--ctx
--max-tokens
--temperature
```

For example:

```bash
python pyinfer.py \
  path/to/model.gguf \
  --ctx 4096 \
  --max-tokens 2048 \
  --temperature 0.7
```

---

# Configuration

The repository contains a sanitized example:

```text
config/profile.example.json
```

Example structure:

```json
{
  "hardware": {
    "gpu_name": "Example GPU",
    "vram_gb": 8.0,
    "ram_gb": 16.0
  },
  "model_dir": "~/models",
  "optimizations": {
    "tier_label": "Generic GPU + CPU",
    "gpu_layers_table": {},
    "default_context": 4096,
    "default_batch": 256,
    "cpu_threads": 8,
    "default_max_tokens": 2048,
    "default_temperature": 0.7,
    "use_mmap": true,
    "use_mlock": false,
    "recommended_quant": "Q4_K_M",
    "recommended_model_size": "7B"
  }
}
```

The example intentionally uses generic hardware information.

Do **not** commit a machine-generated `profile.json` containing personal machine paths, hardware information, or local model locations.

---

# Runtime Data

ACIE's memory system can create local runtime data under:

```text
memory_store/
```

This directory is excluded from Git.

The repository should therefore contain source/configuration code, not a developer's private conversation or memory database.

---

# Development

Compile-check the repository with:

```bash
python -m compileall -q .
```

A successful command produces no output and returns exit code `0`.

For development, keep generated files out of source control:

```bash
git status --short
```

and verify that runtime/config/model artifacts remain ignored.

---

# Validation Philosophy

ACIE intentionally separates several concepts that are often mixed together in local AI applications:

### Generation

Can the model produce an answer/code?

### Drift detection

Is generation moving away from the apparent intent?

### Goal checking

Does the response appear to address the requested objectives?

### Confidence

Does the response exhibit signals associated with a stronger or weaker result?

### Validation

Does the assembled code have structural and integration problems?

### Self-correction

Can identified weaknesses be targeted instead of blindly regenerating everything?

This gives ACIE a layered architecture:

```text
          GENERATION
              │
              ▼
       DRIFT DETECTION
              │
              ▼
        GOAL CHECKING
              │
              ▼
        CONFIDENCE
              │
              ▼
       SELF-CORRECTION
              │
              ▼
       CODE VALIDATION
```

No individual layer guarantees correctness. The design instead combines multiple independent signals to improve robustness.

---

# Design Principles

## Local-first

The core architecture is designed around locally hosted GGUF models rather than requiring a hosted inference API.

## Modular

Major capabilities are separated into focused modules so that routing, memory, generation, validation, and resource allocation can evolve independently.

## Adaptive

PRIMA adjusts generation resources based on task characteristics rather than applying one fixed configuration to every request.

## Feedback-driven

ACIE does not have to accept the first generation as final. Goal checks, confidence estimation, DELTA, and validation provide opportunities for recovery.

## Runtime-safe repository

Machine-specific profiles, model weights, memory stores, caches, logs, and Python bytecode should not be committed to the public repository.

---

# Limitations

ACIE is an experimental/local AI engineering project and should not be interpreted as a formal program verifier or guaranteed autonomous software engineer.

Important limitations include:

- Local model quality strongly affects the final result.
- Hardware determines feasible model size and inference performance.
- Routing is heuristic rather than learned classification.
- Confidence scoring is heuristic.
- DELTA detects estimated semantic drift; it does not prove correctness.
- Generated code can still contain logical, security, or architectural errors.
- Static validation cannot replace execution, integration testing, or human review.
- FAISS/Sentence Transformer retrieval quality depends on embedding quality and stored memory.
- `llama-cpp-python` installation can vary by platform and acceleration configuration.

For production software, generated code should still be reviewed and tested.

---

# Security Considerations

When publishing an ACIE installation:

### Do not commit:

```text
profile.json
memory_store/
*.gguf
*.bin
*.safetensors
models/
.env
API keys
tokens
passwords
private credentials
```

The repository's `.gitignore` already excludes the major runtime/model categories.

Before the first public push, it is recommended to inspect the staged diff:

```bash
git diff --cached
```

and verify that only intended source/configuration/documentation files are included.

---

# Project Status

ACIE is an evolving prototype focused on experimenting with adaptive local coding intelligence.

The architecture currently combines:

- task routing
- adaptive resource allocation
- semantic memory
- multi-step planning
- modular execution
- repetition protection
- semantic drift estimation
- goal-aware checking
- confidence estimation
- self-correction
- three-layer code validation

The project is intended to provide a foundation for further experimentation rather than claim production-level autonomous coding reliability.

---

# Roadmap Ideas

Potential future improvements include:

- stronger learned task routing
- more robust language detection
- expanded automated test generation
- runtime execution/sandbox validation of generated projects
- richer dependency analysis
- more reliable cross-module symbol resolution
- benchmark suites for routing and validation
- configurable memory retention policies
- better drift calibration
- structured generation schemas
- GPU/backend-specific installation profiles
- automated CI testing
- expanded unit and integration test coverage
- project-level evaluation benchmarks

---

# Contributing

Contributions are welcome.

A useful contribution should generally:

1. Keep modules focused.
2. Avoid committing model weights or machine-specific runtime data.
3. Add or update tests for changed behavior.
4. Document non-obvious architectural changes.
5. Preserve compatibility where practical.
6. Run at least:

```bash
python -m compileall -q .
```

before submitting changes.

For larger architectural changes, explain the motivation, trade-offs, and expected effect on the ACIE pipeline.

---

# License

This repository is intended to be distributed under the license included in the repository.

If no license file has yet been added, choose and add an appropriate open-source license before publishing the repository publicly.

---

# Acknowledgements

ACIE builds on the local Python AI ecosystem, including:

- `llama-cpp-python` for local GGUF model inference
- `sentence-transformers` for semantic embeddings
- `FAISS` for vector retrieval
- `NumPy` for numerical operations
- `Rich` for terminal presentation
- `psutil` for hardware/resource information

---

## Final Pipeline

At a high level, ACIE can be summarized as:

```text
                 ┌─────────────────────┐
                 │     User Request    │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │       Router        │
                 │ intent / mode / lang│
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │       PRIMA         │
                 │ difficulty / budget │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │       Memory        │
                 │ semantic retrieval  │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Prompt Construction │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Planner / Generator │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │      Executor       │
                 │ modular generation  │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │       DELTA         │
                 │ drift estimation    │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Goals + Confidence  │
                 └──────────┬──────────┘
                            ▼
                    ┌───────┴───────┐
                    │               │
                 Good?            Weak?
                    │               │
                    │               ▼
                    │        Self-Correction
                    │               │
                    └───────┬───────┘
                            ▼
                 ┌─────────────────────┐
                 │ 3-Layer Validator  │
                 │ integrity / wiring │
                 │ / completeness     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │    Final Result     │
                 └─────────────────────┘
```

**ACIE's core idea:** make local coding intelligence adaptive by combining *routing, resource modulation, memory, controlled generation, semantic monitoring, validation, and correction* into one pipeline.
