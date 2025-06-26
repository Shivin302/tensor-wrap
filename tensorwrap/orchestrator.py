import argparse
import os
import sys
from typing import List, Optional, Dict, Any
from rich import print as rprint

from .schemas import ProblemSpec, KernelCandidate
from .ideas import IdeaGenerator
from .codegen import CodeGenerator, SelfHealingCodeGenerator
from .evaluator.evaluator_utils import Evaluator
from .evaluator.cpp_cpu import MockEvaluator, LocalCPUCompiler
from .evaluator.triton_gpu import TritonGPUEvaluator
from .evaluator.cuda_gpu import CudaGPUEvaluator
from .scorer import Scorer
from .storage import Storage

use_one_shot_kernels = True

class Orchestrator:
    """Main orchestrator for the kernel optimization search process."""
    
    def __init__(
        self,
        problem_path: str,
        beam_width: int = 4,
        n_rounds: int = 3,
        mode: str = "cpp_cpu",
        dry_run: bool = False,
        db_path: str = "kernels.db"
    ):
        """Initialize the orchestrator.
        
        Args:
            problem_path: Path to the problem spec directory
            beam_width: Width of the beam search
            n_rounds: Number of optimization rounds
            mode: Evaluation mode (cpp_cpu, triton_gpu, cuda_gpu)
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
            
        # Use mock mode for evaluation only when in dry-run mode
        if dry_run:
            self.evaluator = MockEvaluator(problem_path)
        else:
            if mode == "cpp_cpu":
                compiler = LocalCPUCompiler()
                self.evaluator = Evaluator(problem_path, compiler)
            elif mode == "triton_gpu":
                self.evaluator = TritonGPUEvaluator(problem_path)
            elif mode == "cuda_gpu":
                self.evaluator = CudaGPUEvaluator(problem_path)

        if dry_run or use_one_shot_kernels:
            self.code_generator = CodeGenerator(
                mock_mode=dry_run,
                problem_path=abs_problem_path,
                max_iterations=3
            )
        else:
            self.code_generator = SelfHealingCodeGenerator(
                problem_path=abs_problem_path,
                evaluator=self.evaluator,
                max_iterations=3
            )
        
        # Load the problem specification
        self.problem_spec = self.evaluator.problem_spec
        self.scorer = Scorer()
        

    def run(self) -> Optional[KernelCandidate]:
        """Run the optimization search process.
        
        Returns:
            The best kernel candidate found, or None if no valid candidates
        """
        reference_code = self._get_reference_kernel()
        
        reference = KernelCandidate(
            problem=self.problem_spec.name,
            round=0,
            code=reference_code,
            idea="reference implementation"
        )
        
        # Evaluate reference
        latency_ms = self.evaluator.run_reference()
        reference.latency_ms = latency_ms
        
        # Save reference to storage
        reference.id = self.storage.save_candidate(reference)
        
        # Format the latency, handling None values
        latency_display = f"{reference.latency_ms:.2f}ms" if reference.latency_ms is not None else "N/A"
        rprint(f"[bold green]reference[/bold green]: latency={latency_display}")
        
        self.best_candidate = reference
        reference_idea = "Write a C++ implementation of this python code"
        candidate = self.optimize_idea(reference, reference_idea, 0)
        # Update best candidate
        if candidate.correct and (self.best_candidate is None or candidate.latency_ms < self.best_candidate.latency_ms):
            self.best_candidate = candidate
        prev_candidates = [candidate]

        # Run optimization rounds
        for round_num in range(1, self.n_rounds + 1):
            rprint(f"\n[bold blue]Round {round_num}/{self.n_rounds}[/bold blue]")
            
            if not prev_candidates:
                rprint("No valid candidates from previous round, stopping search.")
                break
            
            # Get the best candidates from the previous round
            top_candidates = self.scorer.rank_candidates(prev_candidates, self.beam_width)
            # top_candidates = self.scorer.rank_candidates(prev_candidates, 1)
            
            # Generate and evaluate new candidates
            round_candidates = []
            
            for parent in top_candidates:
                parent_candidates = self.optimize_candidate(parent, round_num)
                round_candidates.extend(parent_candidates)


            print("-" * 120)
            
            # Update previous candidates for next round
            prev_candidates = round_candidates
        
        # Return the best candidate
        if self.best_candidate is not None:
            rprint(f"\n[bold green]Best candidate[/bold green]: round={self.best_candidate.round}, " + 
                  f"latency={self.best_candidate.latency_ms:.2f}ms")
            rprint(f"Idea: {self.best_candidate.idea}")
        else:
            rprint("\n[bold red]No valid candidates found.[/bold red]")
            
        return self.best_candidate
    

    def optimize_candidate(self, candidate: KernelCandidate, round_num: int) -> List[KernelCandidate]:
        """Optimize a candidate kernel."""
        ideas = self.idea_generator.generate_ideas(candidate.code, self.beam_width)
        
        round_candidates = []
        print("-" * 120)
        for idea in ideas:
            candidate = self.optimize_idea(candidate, idea, round_num)
            round_candidates.append(candidate)
            
            # Update best candidate
            if candidate.correct and (self.best_candidate is None or candidate.latency_ms < self.best_candidate.latency_ms):
                self.best_candidate = candidate

        return round_candidates


    def optimize_idea(self, candidate: KernelCandidate, idea: str, round_num: int) -> KernelCandidate:
        print("-" * 120)

        idea_title = idea.split("\n")[0]
        rprint(f"[bold blue]{idea_title}[/bold blue]")

        # Generate code from idea - may return code string or dict with metadata
        code_result = self.code_generator.generate_code(candidate.code, idea, self.problem_spec)
        
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
                rprint(f"Using pre-validated candidate from self-healing")
                is_correct = True  # We know it's correct from self-healing
                latency_ms = latency_ms_from_healing
            else:
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
        rprint(f"Results: correct={candidate.correct}, latency={latency_display}")
        print("-" * 120)
        return candidate
        
        


    def _get_reference_kernel(self) -> str:
        """Get the reference kernel code for the problem.
        
        Returns:
            The reference kernel code
        """
        reference_kernel_path = self.problem_path + "/ref_impl.py"
        
        with open(reference_kernel_path, "r") as f:
            return f.read()

def main():
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(description="TensorWrap kernel optimization search")
    parser.add_argument("--problem", type=str, default="matmul", help="directory name in tensorwrap/problems")
    parser.add_argument("--rounds", type=int, default=1, help="Number of optimization rounds")
    parser.add_argument("--beam", type=int, default=2, help="Beam size (top-k candidates per round)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual compilation)")
    parser.add_argument("--mode", default="cpp_cpu", help="Mode: cpp_cpu, triton_gpu, cuda_gpu")
    parser.add_argument("--db", default="kernels.db", help="Path to kernels database")
    args = parser.parse_args()
    
    # Determine problem path
    problem_path = "tensorwrap/problems/" + args.problem
    
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