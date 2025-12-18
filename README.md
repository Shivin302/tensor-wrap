# TensorWrap Kernel Optimization System

An agentic search system for auto-generating and evaluating optimized CPU and GPU kernels using LLMs. Inspired by https://scalingintelligence.stanford.edu/pubs/kernelbench/
This product starts off with a numpy reference implementation of an operation, and incrementally calls LLMs to generate ideas for optimizing in C++ or CUDA, another LLM call to implement the idea, and then running that kernel to get results for the next iteration for kernel optimization. We are able to replicate results of the Stanford paper. Included in the code are matmul and convolution kernels, and this applies to any operation that can be written in numpy. 

## Setup

### Prerequisites

Install the required dependencies using Homebrew:

```bash
brew install python3 cmake ninja pybind11
```

### Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### API Keys

Set your LLM provider API keys:

```bash
export OPENAI_API_KEY=your_openai_key
# OR
export GOOGLE_API_KEY=your_google_key
```

### Running a Search

To run a search using the Python API:

```python
from tensorwrap.orchestrator import Orchestrator
from tensorwrap.schemas import ProblemSpec

# Define problem specification
prob = ProblemSpec(name="matmul", shape_a=[512, 512], shape_b=[512, 512], dtype="float32")

# Create orchestrator and run search
best_kernel = Orchestrator(prob).run()
print(f"Best kernel found with latency: {best_kernel.latency_ms:.2f}ms")
```

### Command Line Interface

Use the provided CLI wrapper:

```bash
./bin/tensorwrap --problem matmul --rounds 3 --beam 4
```

Options:
- `--problem`: Problem name (currently only "matmul" is supported)
- `--rounds`: Number of optimization rounds (default: 3)
- `--beam`: Beam width (default: 4)
- `--dry-run`: Run for a single candidate and exit

### Testing

Run the test suite:

```bash
pytest
```

## System Components

- `schemas.py`: Pydantic models for problem specification and kernels
- `storage.py`: DuckDB wrapper for storing kernel candidates
- `ideas.py`: Generates optimization ideas via LLM
- `codegen.py`: Converts ideas into kernel code
- `evaluator/local_cpu.py`: Compiles and benchmarks candidates
- `scorer.py`: Ranks candidates by performance
- `orchestrator.py`: Main search loop

## Extending

### Adding New Problems

To add a new problem:

1. Create a directory in `problems/` for your problem
2. Add a `spec.yaml` file defining the problem parameters
3. Add a reference implementation in a `ref_impl.py` file
4. Update `Orchestrator._get_baseline_kernel()` to provide a baseline implementation
