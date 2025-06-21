"""
Self-healing kernel optimizer that uses LLM to iteratively debug and fix kernels.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import os

from .schemas import ProblemSpec, KernelCandidate
import re
import time

from .evaluator.local_cpu import LocalCPUEvaluator


@dataclass
class CompileResult:
    """Result of a compilation attempt."""
    success: bool
    error: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a validation attempt."""
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    correctness: Optional[bool] = None


class LLMSelfHealingOptimizer:
    """Uses LLM to iteratively debug and fix kernels until a viable candidate is found."""
    
    def __init__(self, llm_client, problem_path, max_iterations=3):
        """Initialize the optimizer.
        
        Args:
            llm_client: OpenAI client for LLM API calls
            problem_path: Path to problem specification directory
            max_iterations: Maximum number of optimization iterations
        """
        self.llm_client = llm_client
        self.problem_path = problem_path
        self.max_iterations = max_iterations
        self.evaluator = LocalCPUEvaluator(problem_path, mock_mode=False)
        self.problem_name = self._get_problem_name_from_path(problem_path)
        print(f"Using self-healing optimizer for problem: {self.problem_name}")
    
    def _get_problem_name_from_path(self, path):
        """Extract problem name from path.
        
        Args:
            path: Path to problem directory
            
        Returns:
            Problem name (last component of path)
        """
        return os.path.basename(os.path.normpath(path))
        
    def optimize_kernel(self, idea, baseline_code, problem_spec):
        """Generate a viable kernel candidate through LLM-based iterative debugging.
        
        Args:
            idea: The optimization idea
            baseline_code: The baseline kernel code
            problem_spec: Problem specification object
            
        Returns:
            Dictionary with optimized code and metadata, or None if optimization failed
        """
        print(f"\nStarting self-healing optimization for idea: {idea[:50]}...")
        
        context = {
            "idea": idea,
            "baseline_code": baseline_code,
            "problem_spec": problem_spec,
            "attempt_history": [],
            "current_iteration": 0
        }
        
        while context["current_iteration"] < self.max_iterations:
            # Increment iteration counter
            context["current_iteration"] += 1
            print(f"Iteration {context['current_iteration']}/{self.max_iterations}")
            
            # Generate candidate kernel or fix previous attempt
            if context["current_iteration"] == 1:
                candidate = self._generate_initial_candidate(context)
            else:
                candidate = self._fix_candidate(context)
                
            if not candidate:
                print("Failed to generate candidate code")
                break
                
            # Try to compile and validate
            compile_result, validation_result = self._test_candidate(candidate)
            
            # Record this attempt in history
            context["attempt_history"].append({
                "candidate": candidate,
                "compile_result": compile_result,
                "validation_result": validation_result
            })
            
            # Check if we've succeeded
            if compile_result.success and validation_result.success:
                print(f"Successfully optimized kernel in {context['current_iteration']} iterations!")
                return {
                    "code": candidate,
                    "idea": idea,
                    "iterations": context["current_iteration"],
                    "latency_ms": validation_result.latency_ms
                }
            else:
                if not compile_result.success:
                    print(f"Compilation failed: {compile_result.error}")
                elif not validation_result.success:
                    print(f"Validation failed: {validation_result.error}")
        
        # Failed to generate a viable candidate after max iterations
        print(f"Failed to optimize kernel after {self.max_iterations} iterations")
        return None
    
    def _generate_initial_candidate(self, context):
        """Generate initial kernel candidate based on optimization idea."""
        print("Generating initial optimized kernel...")
        
        prompt = f"""
        You are an expert kernel developer optimizing computational kernels for CPU (not GPU).
        
        BASELINE CODE:
        ```cpp
        {context["baseline_code"]}
        ```
        
        PROBLEM SPECIFICATION:
        {str(context["problem_spec"].__dict__)}
        
        OPTIMIZATION IDEA:
        {context["idea"]}
        
        Create an optimized version of this kernel that implements the optimization idea.
        The code must:
        1. Be compilable C++ with pybind11 for CPU (not CUDA/GPU)
        2. Maintain the same function signature: py::array_t<float> kernel(py::array_t<float> a, py::array_t<float> b)
        3. Produce numerically equivalent results
        4. Implement the optimization idea effectively but for CPU only
        5. Include helpful comments explaining key optimizations
        
        PYBIND11/GIL HANDLING:
        1. DO NOT use advanced GIL state management (PyGILState_* functions)
        2. DO NOT use Py_BEGIN_ALLOW_THREADS or Py_END_ALLOW_THREADS
        3. Let pybind11 handle all GIL management automatically
        
        FORBIDDEN APPROACHES:
        1. DO NOT use CUDA, cuBLAS, or any GPU-specific libraries or headers
        2. DO NOT use OpenMP or other parallel threading libraries
        3. DO NOT use advanced GIL-state manipulation functions
        
        Only output the complete optimized kernel code.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-2025-04-16",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content
            return self._extract_code(response_text)
        except Exception as e:
            print(f"Error generating initial candidate: {e}")
            return None
    
    def _fix_candidate(self, context):
        """Fix issues with the previous candidate."""
        print("Fixing previous candidate...")
        
        last_attempt = context["attempt_history"][-1]
        
        prompt = f"""
        You are an expert kernel developer debugging optimized computational kernels for CPU (not GPU).
        
        OPTIMIZATION IDEA:
        {context["idea"]}
        
        BASELINE CODE:
        ```cpp
        {context["baseline_code"]}
        ```
        
        PROBLEM SPECIFICATION:
        {str(context["problem_spec"].__dict__)}
        
        PREVIOUS ATTEMPT:
        ```cpp
        {last_attempt["candidate"]}
        ```
        
        ISSUES ENCOUNTERED:
        Compilation success: {last_attempt["compile_result"].success}
        {last_attempt["compile_result"].error if not last_attempt["compile_result"].success else ""}
        
        Validation success: {last_attempt["validation_result"].success}
        {last_attempt["validation_result"].error if not last_attempt["validation_result"].success else ""}
        
        TASK:
        Fix the issues in the previous attempt while preserving the optimization idea.
        
        IMPORTANT CONSTRAINTS:
        1. This must be pure C++ code for CPU (not CUDA/GPU)
        2. Do NOT use CUDA, cuBLAS, or any GPU-specific libraries or headers
        3. Fix any compilation errors and ensure correct numerical output
        4. The kernel function signature must be: py::array_t<float> kernel(py::array_t<float> a, py::array_t<float> b)
        5. Ensure the PYBIND11_MODULE and m.def("kernel", &kernel) declaration is included
        
        PYBIND11/GIL HANDLING - CRITICAL:
        1. DO NOT use advanced GIL state management (PyGILState_* functions)
        2. DO NOT use Py_BEGIN_ALLOW_THREADS or Py_END_ALLOW_THREADS
        3. Let pybind11 handle all GIL management automatically
        4. DO NOT use OpenMP or other parallel threading libraries
        5. These errors commonly cause segmentation faults and crashes
        
        CHAIN OF THOUGHT:
        1. Identify the root cause of the error
        2. Determine necessary changes while preserving optimizations
        3. Implement fixes systematically
        4. Verify changes address the specific issues
        
        Only output the complete fixed kernel code.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-2025-04-16",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content
            return self._extract_code(response_text)
        except Exception as e:
            print(f"Error fixing candidate: {e}")
            return None
    
    def _test_candidate(self, candidate_code):
        """Test if the candidate compiles and produces correct results."""
        # 1. Try compiling the code
        compile_result = self._compile_candidate(candidate_code)
        
        # 2. If compilation succeeds, validate correctness and performance
        validation_result = ValidationResult(
            success=False, 
            error="Not validated - compilation failed"
        )
        
        if compile_result.success:
            validation_result = self._validate_candidate(candidate_code)
            
        return compile_result, validation_result
    
    def _compile_candidate(self, candidate_code):
        """Compile the candidate code and return result."""
        try:
            # Use the evaluator's _compile method
            output_path = self.evaluator._compile(candidate_code)
            if output_path:
                return CompileResult(success=True, error=None, output_path=output_path)
            else:
                return CompileResult(success=False, error="Compilation failed")
        except Exception as e:
            print(f"Exception during compilation: {e}")
            return CompileResult(success=False, error=str(e))
    
    def _validate_candidate(self, candidate_code):
        """Validate candidate correctness and performance."""
        try:
            # Create a temporary candidate to validate using our stored problem_name
            candidate = KernelCandidate(
                problem=self.problem_name,
                round=0,
                code=candidate_code,
                idea="Optimization candidate"
            )
            
            print(f"Validating candidate with problem: {self.problem_name}")
            
            # Check if code contains GIL management issues
            gil_keywords = ["PyGILState_", "Py_BEGIN_ALLOW_THREADS", "Py_END_ALLOW_THREADS"]
            uses_advanced_gil = any(keyword in candidate_code for keyword in gil_keywords)
            if uses_advanced_gil:
                print("WARNING: Detected advanced GIL handling in candidate. This may cause issues with pybind11.")
                return ValidationResult(success=False, error="Advanced GIL handling detected. This requires careful implementation with pybind11.", latency_ms=None)
            
            # Use the evaluator to validate
            is_correct, latency_ms = self.evaluator.evaluate(candidate)
            
            if not is_correct:
                return ValidationResult(success=False, error="Output is not correct", latency_ms=None)
            
            return ValidationResult(success=True, error=None, latency_ms=latency_ms)
        
        except Exception as e:
            print(f"Exception during validation: {e}")
            return ValidationResult(success=False, error=str(e), latency_ms=None)
    
    def _extract_code(self, llm_response):
        """Extract code from LLM response."""
        # Simple extraction of code between triple backticks
        code_matches = re.findall(r"```(?:cpp)?\n(.*?)```", llm_response, re.DOTALL)
        if code_matches:
            return code_matches[0].strip()
        return llm_response.strip()
