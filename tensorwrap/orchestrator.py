import argparse
import os
import sys
from typing import List, Optional, Dict, Any
from rich import print as rprint

from .schemas import ProblemSpec, KernelCandidate
from .ideas import IdeaGenerator
from .codegen import CodeGenerator
from .evaluator.local_cpu import LocalCPUEvaluator
from .scorer import Scorer
from .storage import Storage

class Orchestrator:
    """Main orchestrator for the kernel optimization search process."""
    
    def __init__(
        self,
        problem_path: str,
        beam_width: int = 4,
        n_rounds: int = 3,
        mode: str = "local_cpu",
        dry_run: bool = False,
        db_path: str = "kernels.db"
    ):
        """Initialize the orchestrator.
        
        Args:
            problem_path: Path to the problem spec directory
            beam_width: Width of the beam search
            n_rounds: Number of optimization rounds
            mode: Evaluation mode (local_cpu only for now)
            dry_run: If True, use mock responses for testing without LLM API calls
            db_path: Path to kernels database
        """
        self.problem_path = problem_path
        self.beam_width = beam_width
        self.n_rounds = n_rounds
        self.mode = mode
        self.dry_run = dry_run
        
        # Initialize storage
        self.storage = Storage(db_path)
        
        # Initialize modules
        self.idea_generator = IdeaGenerator(mock_mode=dry_run)
        # Ensure problem_path is absolute
        if not os.path.isabs(problem_path):
            abs_problem_path = os.path.abspath(problem_path)
        else:
            abs_problem_path = problem_path
            
        self.code_generator = CodeGenerator(
            mock_mode=dry_run,
            problem_path=abs_problem_path,
            self_healing=not dry_run,  # Only use self-healing when not in dry-run mode
            max_iterations=3
        )
        # Use mock mode for evaluation only when in dry-run mode
        self.evaluator = LocalCPUEvaluator(problem_path, mock_mode=dry_run)
        
        # Load the problem specification
        self.problem_spec = self.evaluator.problem_spec
        self.scorer = Scorer()
        
    def run(self) -> Optional[KernelCandidate]:
        """Run the optimization search process.
        
        Returns:
            The best kernel candidate found, or None if no valid candidates
        """
        # Get baseline kernel code
        baseline_code = self._get_baseline_kernel()
        
        # Get the problem name from the path
        problem_name = self.problem_path.split('/')[-1]
        
        # Store baseline as round 0
        baseline = KernelCandidate(
            problem=problem_name,
            round=0,
            code=baseline_code,
            idea="Baseline implementation"
        )
        
        # Evaluate baseline
        is_correct, latency_ms = self.evaluator.evaluate(baseline)
        baseline.correct = is_correct
        baseline.latency_ms = latency_ms
        
        # Save baseline to storage
        baseline.id = self.storage.save_candidate(baseline)
        
        # Format the latency, handling None values
        latency_display = f"{baseline.latency_ms:.2f}ms" if baseline.latency_ms is not None else "N/A"
        rprint(f"[bold green]Baseline[/bold green]: correct={baseline.correct}, latency={latency_display}")
        
        # Start with baseline as best candidate
        best_candidate = baseline if baseline.correct else None
        
        # Keep track of candidates from the previous round
        prev_candidates = [baseline] if baseline.correct else []
        
        # Run optimization rounds
        for round_num in range(1, self.n_rounds + 1):
            rprint(f"\n[bold blue]Round {round_num}/{self.n_rounds}[/bold blue]")
            
            if not prev_candidates:
                rprint("No valid candidates from previous round, stopping search.")
                break
            
            # Get the best candidates from the previous round
            top_candidates = self.scorer.rank_candidates(prev_candidates, self.beam_width)
            
            # Generate and evaluate new candidates
            round_candidates = []
            
            for parent in top_candidates:
                # Generate ideas
                ideas = self.idea_generator.generate_ideas(parent.code)
                
                for idea in ideas:
                    # Generate code from idea - may return code string or dict with metadata
                    code_result = self.code_generator.generate_code(parent.code, idea, self.problem_spec)
                    
                    # Handle different return types from code generator
                    if isinstance(code_result, dict):
                        new_code = code_result["code"]
                        # If we already have latency information from self-healing, use it later
                        latency_ms_from_healing = code_result.get("latency_ms")
                    else:
                        new_code = code_result
                        latency_ms_from_healing = None
                    
                    # Create candidate
                    problem_name = os.path.basename(os.path.normpath(self.problem_path))
                    candidate = KernelCandidate(
                        problem=problem_name,
                        round=round_num,
                        code=new_code,
                        idea=idea
                    )
                    
                    try:
                        if latency_ms_from_healing is not None:
                            rprint(f"Using pre-validated candidate from self-healing for idea: {idea[:60]}...")
                            is_correct = True  # We know it's correct from self-healing
                            latency_ms = latency_ms_from_healing
                        else:
                            # Standard evaluation
                            rprint(f"Evaluating candidate based on idea: {idea[:60]}...")
                            is_correct, latency_ms = self.evaluator.evaluate(candidate)
                        
                        candidate.correct = is_correct
                        candidate.latency_ms = latency_ms
                    except Exception as e:
                        print(f"Error during candidate evaluation: {e}")
                        candidate.correct = False
                        candidate.latency_ms = None
                    
                    # Save candidate to storage
                    candidate.id = self.storage.save_candidate(candidate)
                    
                    # Format latency display safely, handling None values
                    latency_display = f"{candidate.latency_ms:.2f}ms" if candidate.latency_ms is not None else "N/A"
                    rprint(f"  correct={candidate.correct}, latency={latency_display}")
                    
                    round_candidates.append(candidate)
                    
                    # Update best candidate
                    if candidate.correct and (best_candidate is None or candidate.latency_ms < best_candidate.latency_ms):
                        best_candidate = candidate
                    
                    # Check for dry run
                    if self.dry_run:
                        rprint("\n[bold yellow]Dry run completed, exiting early.[/bold yellow]")
                        return best_candidate
            
            # Update previous candidates for next round
            prev_candidates = round_candidates
        
        # Return the best candidate
        if best_candidate is not None:
            rprint(f"\n[bold green]Best candidate[/bold green]: round={best_candidate.round}, " + 
                  f"latency={best_candidate.latency_ms:.2f}ms")
            rprint(f"Idea: {best_candidate.idea}")
        else:
            rprint("\n[bold red]No valid candidates found.[/bold red]")
            
        return best_candidate
    
    def _get_baseline_kernel(self) -> str:
        """Get the baseline kernel code for the problem.
        
        Returns:
            The baseline kernel code
        """
        # Get the problem name from the path
        problem_name = self.problem_path.split('/')[-1]
        
        # For matmul problem, use a simple C++ implementation
        if problem_name == "matmul":
            return """
            #include <cstddef>
            #include <cmath>
            #include <pybind11/numpy.h>
            namespace py = pybind11;
            
            // Define a simple matrix multiplication function that attempts to mimic NumPy's behavior
            py::array_t<float> kernel(py::array_t<float> a, py::array_t<float> b) {
                auto buf_a = a.request();
                auto buf_b = b.request();
                
                if (buf_a.ndim != 2 || buf_b.ndim != 2) {
                    throw std::runtime_error("Number of dimensions must be 2");
                }
                
                if (buf_a.shape[1] != buf_b.shape[0]) {
                    throw std::runtime_error("Incompatible shapes");
                }
                
                size_t M = buf_a.shape[0];
                size_t K = buf_a.shape[1];
                size_t N = buf_b.shape[1];
                
                // Create the result array with the correct shape
                py::array_t<float> result = py::array_t<float>({M, N});
                auto buf_result = result.request();
                
                float* ptr_a = static_cast<float*>(buf_a.ptr);
                float* ptr_b = static_cast<float*>(buf_b.ptr);
                float* ptr_result = static_cast<float*>(buf_result.ptr);
                
                // Initialize the result array to zeros
                for (size_t i = 0; i < M * N; ++i) {
                    ptr_result[i] = 0.0f;
                }
                
                // Use cache-friendly iteration order and accumulate in blocks
                for (size_t i = 0; i < M; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        float sum = 0.0f;
                        for (size_t k = 0; k < K; ++k) {
                            sum += ptr_a[i * K + k] * ptr_b[k * N + j];
                        }
                        ptr_result[i * N + j] = sum;
                    }
                }
                
                return result;
            }
            """
        else:
            raise ValueError(f"Unknown problem: {self.problem_spec.name}")

def main():
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(description="TensorWrap kernel optimization search")
    parser.add_argument("--problem", help="Path to problem directory")
    parser.add_argument("--rounds", type=int, default=3, help="Number of optimization rounds")
    parser.add_argument("--beam", type=int, default=2, help="Beam size (top-k candidates per round)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual compilation)")
    parser.add_argument("--mode", default="generate", help="Mode: generate or evaluate")
    parser.add_argument("--db", default="kernels.db", help="Path to kernels database")
    args = parser.parse_args()
    
    # Determine problem path
    problem_path = args.problem
    if not problem_path:
        print("Error: Please specify a problem path with --problem")
        sys.exit(1)
    
    # Create and run orchestrator
    orchestrator = Orchestrator(
        problem_path=problem_path,
        n_rounds=args.rounds,
        beam_width=args.beam,
        mode=args.mode,
        dry_run=args.dry_run,
        db_path=args.db
    )
    
    best_candidate = orchestrator.run()
    
    if best_candidate is not None:
        print(f"\nBest candidate found with latency: {best_candidate.latency_ms:.2f}ms")
    else:
        print("\nNo valid candidates found.")

if __name__ == "__main__":
    main()