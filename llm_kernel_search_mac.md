# TensorWrap Kernel Optimization System (macOS) — Implementation Guide

This document outlines a minimal agentic search system for auto-generating and evaluating GPU kernels using LLMs. It is specifically tailored to be developed and tested on macOS with no hardware dependencies beyond CPU (GPU support optional via future remote plugin).

---

## 0. Goal

Build a prototype that:

- Accepts a baseline CUDA/C++ kernel.
- Uses LLMs to generate optimization ideas and code variants.
- Compiles and benchmarks candidate kernels.
- Selects the best kernel after 3 beam-search rounds.
- Runs entirely on a Mac (CPU-only).

---

## 1. Environment Setup

### Install the following dependencies (using Homebrew or other means):
- `python3`
- `virtualenv`
- `cmake`
- `ninja`
- `pybind11`

### Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### `requirements.txt`
```
openai
google-generativeai
ray
duckdb
rich
pydantic
pybind11
pytest
ruff
```

---

## 2. Project Structure

```text
tensorwrap/
├── orchestrator.py
├── ideas.py
├── codegen.py
├── evaluator/
│   └── local_cpu.py
├── scorer.py
├── storage.py
├── schemas.py
├── templates/
│   ├── brainstorm.j2
│   └── implement.j2
├── problems/
│   └── matmul/
│       ├── ref_impl.py
│       └── spec.yaml
└── bin/
    └── tensorwrap
```

Create the scaffold:

```bash
mkdir -p tensorwrap/{templates,evaluator,problems/matmul} bin
touch tensorwrap/{__init__,orchestrator,ideas,codegen,scorer,storage,schemas}.py
chmod +x bin/tensorwrap
```

---

## 3. Module Responsibilities

- `schemas.py`: Pydantic models (`ProblemSpec`, `KernelCandidate`, `Score`).
- `storage.py`: DuckDB/SQLite wrapper, `kernels` table.
- `ideas.py`: Generates brainstorm ideas via LLM.
- `codegen.py`: Turns ideas into kernel code.
- `evaluator/local_cpu.py`: Compiles and benchmarks candidates. Includes timeout and np.allclose correctness check.
- `scorer.py`: Ranks candidates by lowest latency among correct ones.
- `orchestrator.py`: Main loop (3 rounds, beam=4).
- `templates/brainstorm.j2`: Prompt for idea generation.
- `templates/implement.j2`: Prompt for code synthesis.
- `problems/matmul/ref_impl.py`: NumPy matmul reference.
- `problems/matmul/spec.yaml`: Kernel problem config.
- `bin/tensorwrap`: CLI wrapper for fast testing.

---

## 4. Compilation Harness Example

```python
def _compile(code:str, func_name:str="kernel"):
    source = f"""
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    {code}
    PYBIND11_MODULE(candidate, m) {{
        m.def("{func_name}", &{func_name});
    }}
    """
    # TODO: add actual pybind11 compile command
    return source
```

Include timeout (e.g. `signal.alarm(2)`) and correctness check using `np.allclose(candidate_out, ref_out, atol=1e-2, rtol=1e-2)`.

---

## 5. Manual Test Run

```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

Then run this script:

```python
from tensorwrap.orchestrator import Orchestrator
from tensorwrap.schemas import ProblemSpec
prob = ProblemSpec(name="matmul", shape_a=[512,512], shape_b=[512,512], dtype="float32")
print(Orchestrator(prob).run())
```

---

## 6. Validation Checklist

1. `pytest` → green.
2. `python -m tensorwrap.orchestrator --dry-run` compiles 1 kernel and exits.
3. DB file `search.db` contains ≥ 1 row with `ok=True`.
4. DuckDB schema for `kernels` table:

```sql
CREATE TABLE kernels (
  id INTEGER PRIMARY KEY,
  problem TEXT,
  round INTEGER,
  code TEXT,
  idea TEXT,
  correct BOOLEAN,
  latency_ms DOUBLE
);
```

---

## 7. Example Prompt Templates

### `templates/brainstorm.j2`
```jinja
You are an expert CUDA kernel optimizer.
Here is the current kernel:
```
{{ code[:300] }}
```
Propose 4 ideas to make this kernel faster. Each idea should reference a specific GPU feature or memory-level optimization.
```

### `templates/implement.j2`
```jinja
You are an expert CUDA engineer. Here is a baseline kernel:
```
{{ baseline }}
```
Now rewrite this kernel by applying the following idea:
- {{ idea }}
Return only code inside triple backticks.
```

---

## 8. Sample Problem Config

### `problems/matmul/spec.yaml`
```yaml
name: matmul
shape_a: [512, 512]
shape_b: [512, 512]
dtype: float32
```


