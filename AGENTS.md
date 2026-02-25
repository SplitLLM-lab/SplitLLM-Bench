# AGENTS.md

Project: SplitLLM
Purpose: Split-based edge–cloud collaboration for LLM inference

This repository contains research-oriented code for splitting large language
models into edge and cloud components and running collaborative inference.

This is NOT a framework or production SDK.
This is a research codebase prioritizing clarity, reproducibility, and simplicity.

---

## Core Philosophy

Prefer:

- simple Python scripts
- explicit logic
- minimal abstraction
- reproducible experiments
- readable research code

Avoid:

- unnecessary object-oriented design
- enterprise-style architecture
- plugin systems
- dependency injection
- "manager", "controller", or "service" layers
- over-generalization
- premature optimization

When in doubt, choose readability over abstraction.

---

## Architecture Rules

This repository follows a **research-code structure**, not a library structure.

Only introduce abstractions when they clearly reduce duplication or complexity.

Guidelines:

- Prefer functions over classes
- Only introduce classes when stateful behavior is required
- Do not abstract code used only once
- Avoid deep module hierarchies
- Avoid metaprogramming
- Avoid design patterns unless explicitly requested

Bad example:
Creating reusable architecture for a single experiment.

Good example:
Writing a clear script that runs the experiment end-to-end.

---

## Project Structure 

Keep files small and focused.
Avoid creating new directories unless necessary.

File placement rules:

- Put code in the corresponding folder by domain.
- `model/` for inference runtime/model interface code.
- `split/` for checkpoint split tools.
- `tests/` for experiment/reference scripts.
- `bench/` for benchmark scripts and shared benchmark helpers.
- `custom/` for user-defined experimental codecs/modules used by benchmarks.
- `docs/` for documentation and design notes.
- Keep `model/` internally separated by concern (e.g. codec in `model/codec.py`, runtime in `model/runtime.py`, model API in `model/model.py`).
- Avoid putting codec implementation details directly into runtime files when a small dedicated module is enough.
- Keep a no-transport-overhead codec baseline as the default for local runtime experiments; only switch to wire/compressed codec when explicitly needed.
- Avoid adding new top-level `.py` files unless explicitly required.
- Do not modify the project root `README.md` unless explicitly requested by the user.
- Prefer writing module-level docs in folder READMEs (e.g. `split/README.md`, `model/README.md`).

---

## Coding Style

Python version:
Python 3.10+

Style rules:

- Use type hints when helpful
- Prefer dataclasses over classes with many methods
- Prefer pathlib over os.path
- Prefer standard library
- Avoid decorators unless trivial
- Avoid metaclasses
- Avoid global state unless necessary

Keep control flow explicit and linear.

Readable code is more important than clever code.

---

## Logging & Debugging

Use lightweight, explicit logs for easier debugging.

Preferred log prefixes:

- `[info]` operation start, key parameters, selected path/device/config
- `[ok]` successful completion, output paths, key counts/shapes/time
- `[warn]` recoverable anomalies, fallback paths, ignored keys/items
- `[error]` failures before raising/exit, with the failing step context
- `[progress]` periodic status for long loops (dataset/model/key scan)

Logging rules:

- Most meaningful operations should print at least one `[info]` and one `[ok]` or `[error]`.
- Keep logs one-line and concrete; include numbers and file paths when possible.
- Prefer aggregated progress logs over per-item spam.
- Do not introduce heavy logging frameworks unless explicitly requested.
- Simple `print(...)` with consistent prefixes is preferred in this repository.

---

## Research Code Guidelines

This repository supports:

- model splitting experiments
- checkpoint transformation
- edge–cloud inference experiments
- latency / bandwidth experiments
- reproducibility scripts

---

## Benchmark Guidelines

Benchmark code should stay measurement-focused and reproducible.

Rules:

- Put benchmark entry scripts in `bench/` (example: `bench/ppl.py`).
- Put reusable benchmark helpers in `bench/utils.py` and reuse them across benchmarks.
- Benchmark CLI should focus on benchmark settings (dataset, split, samples, max_length, batch_size, seed, output path).
- Do not add benchmark-only transport simulation switches unless explicitly requested.
- The tested path should be determined by selected model/runtime and codec, not by synthetic benchmark toggles.
- For line-by-line JSONL generation workloads, default to `bench.utils.generate_jsonl_with_model` as the standard interface.
- For quality-evaluation tasks that first require generation (for example MCQ/free-form answer sets), prefer a two-stage pipeline: generate into JSONL first, then score from JSONL.
- JSONL generation jobs should write one output row per completed sample, flush incrementally, and reorder final output to match input order by index.
- Keep codec selection explicit via `--codec` and support custom module specs (e.g. `custom.my_codec`, `pkg.mod:codec_obj`, `pkg.mod.MyCodec`).
- Keep result JSON schema stable across benchmarks: `benchmark`, `model`, `codec`, `dataset`, `runtime`, `eval` (plus benchmark-specific fields when needed).
- Keep logs lightweight with `[info]`, `[progress]`, `[result]`, `[ok]`, `[error]`.
- Run benchmarks from repo root using module mode when possible (example: `python3 -m bench.ppl ...`).
- Benchmark mode policy (`local` vs `remote`) should follow metric sensitivity, not convenience:
- If a benchmark is system/deployment sensitive (e.g., end-to-end latency, throughput, bandwidth, transport overhead), implement both `local` and `remote` modes.
- If a benchmark is quality-only and remote/local are expected to be nearly identical (example: PPL/perplexity、MMLU), implement `local` mode only.
- For new benchmarks, default to implementing both modes unless there is a clear, documented reason that `remote` adds no meaningful signal.

---

## Model Splitting Logic

The repository already contains working experimental code under:

```

tests/

```

These scripts are considered **ground truth references**.

---

## Testing

Testing in this repository means:

- experiment scripts run successfully
- splitting logic produces correct checkpoints
- configs remain compatible with HuggingFace models

Do not introduce large testing frameworks unless requested.

---

## When Generating Code

The agent should behave like:

"A researcher writing clean experimental Python code."

NOT like:

"A library engineer designing reusable infrastructure."

Preferred output style:

- short modules
- direct functions
- minimal indirection
- clear tensor manipulation logic

---

## Anti-Overengineering Rules (Important)

The agent MUST NOT:

- convert scripts into class hierarchies
- introduce factory patterns
- introduce plugin architectures
- introduce registries
- introduce config frameworks
- split small files into many modules
- generalize single-use logic

Only refactor when explicitly requested.

---

## Reproducibility Priority

All experiments should be runnable with:

- minimal setup
- explicit scripts
- documented parameters

Avoid hidden behavior.

---

## Environment

Use this project environment by default:

- Create/activate virtual environment under repo root: `.venv`
- Preferred executables: `.venv/bin/python`, `.venv/bin/uv`
- For shell commands, run with: `source .venv/bin/activate && <command>`
- If a task involves running models / LLM inference, full end-to-end execution is optional for agent verification; syntax and basic runnability checks are enough, and the user can run full tests.

---

## Summary

This is a research prototype repository for split LLM inference.

Key values:

- simplicity
- clarity
- reproducibility
- minimal abstraction
