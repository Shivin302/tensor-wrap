#!/usr/bin/env python
"""
████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ██╗    ██╗██████╗  █████╗ ██████╗ 
╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗
   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██║ █╗ ██║██████╔╝███████║██████╔╝
   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║███╗██║██╔══██╗██╔══██║██╔═══╝ 
   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██║     
   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     
                                                                                        
"""
from rich.console import Console
from rich.text import Text
console = Console()

def get_tensorwrap_logo():
    """Returns a styled ASCII art logo for TensorWrap using Rich formatting."""
    # The original Unicode ASCII art logo
    logo_text = """
████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ██╗    ██╗██████╗  █████╗ ██████╗ 
╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗
   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██║ █╗ ██║██████╔╝███████║██████╔╝
   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║███╗██║██╔══██╗██╔══██║██╔═══╝ 
   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██║     
   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     
"""
    
    # Create a panel with a strong border to display the logo
    panel = Panel(
        Text(logo_text, style="bold cyan"),
        box=box.HEAVY,  # Use a heavier border for emphasis
        border_style="bright_blue",
        padding=(0, 6),  # Add padding for better appearance
        expand=True     # Don't expand to fill width
    )
    
    # Center the panel with the logo
    centered_logo = Align.center(panel)
    
    return centered_logo

import argparse
import time
import random
import os
import shutil
import importlib.util
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich import print as rprint
from rich import box
from rich.align import Align
from rich.live import Live

# Initialize rich console
console = Console()

# Formatting utility functions
def get_terminal_width():
    """Get the current terminal width."""
    return shutil.get_terminal_size().columns


def print_separator(char="-", width=80):
    """Prints a separator line of given width."""
    console.print(char * min(width, 80), style="dim cyan")


def print_centered(text):
    """Print text centered in the terminal."""
    term_width = get_terminal_width()
    console.print("{:^{width}}".format(text, width=term_width))


def print_wrapped(messages, width=80, style=""):
    """Print messages with proper line wrapping and Rich styling.
    
    Args:
        messages (str or list): A single string or list of strings to be printed with wrapping
        width (int): Maximum line width
        style (str): Rich style string to apply to the text
    """
    if isinstance(messages, str):
        messages = [messages]
        
    for message in messages:
        # Simple wrapping implementation
        current_line = ""
        for word in message.split():
            if len(current_line) + len(word) + 1 <= width:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                console.print(current_line, style=style)
                current_line = word
        if current_line:
            console.print(current_line, style=style)

# Compilation steps configuration with relative complexity (determines execution time)
COMPILATION_STEPS = {
    "llm": {
        "standard": [
            {"name": "Loading transformer LLM", "complexity": 1.0},
            # Commented out but keeping structure intact
            # {"name": "Setting up tokenization pipeline", "complexity": 1.2},
            {"name": "Compiling model to LLM.pt", "complexity": 1.5}
        ],
        "optimized": [
            {"name": "Loading basic transformer LLM", "complexity": 1.0},
            {"name": "Compiling basic model to LLM.pt", "complexity": 1.5},
            {"name": "Analyzing attention mechanism bottlenecks", "complexity": 1.5},
            {"name": "Profiling layer-wise runtime performance", "complexity": 2.0},
            {"name": "Identifying compute-intensive operations", "complexity": 1.2},
            {"name": "Measuring memory access patterns", "complexity": 1.8},
            {"name": "Optimizing memory layout for cache efficiency", "complexity": 2.0},
            {"name": "Generating hardware-specific optimizations", "complexity": 2.5},
            {"name": "Implementing fused attention kernels", "complexity": 3.0},
            # Commented out but keeping structure intact
            # {"name": "Applying weight matrix sparsification", "complexity": 1.5},
            {"name": "Compiling model to optimized_LLM.pt", "complexity": 2.2}
        ]
    },
    "vision": {
        "standard": [
            {"name": "Loading vision model", "complexity": 1.0},
            # Commented out but keeping structure intact
            # {"name": "Setting up image processing pipeline", "complexity": 1.3},
            # {"name": "Configuring inference parameters", "complexity": 0.8},
            {"name": "Compiling model to Vision.pt", "complexity": 1.6}
        ],
        "optimized": [
            {"name": "Loading basic vision model", "complexity": 1.0},
            {"name": "Compiling basic model to Vision.pt", "complexity": 1.5},
            {"name": "Analyzing model architecture", "complexity": 1.0},
            {"name": "Profiling layer-by-layer execution times", "complexity": 2.2},
            {"name": "Identifying compute-intensive convolutional layers", "complexity": 1.5},
            {"name": "Generating hardware-specific optimizations", "complexity": 2.8},
            {"name": "Optimizing convolutional layers with custom CUDA kernels", "complexity": 3.0},
            {"name": "Applying memory layout optimizations", "complexity": 1.8},
            {"name": "Fusing batch normalization with convolutions", "complexity": 2.0},
            {"name": "Loading vision model with custom kernels", "complexity": 1.2},
            {"name": "Compiling model to optimized_Vision.pt", "complexity": 1.7}
        ]
    }
}

# Verification steps configuration
VERIFICATION_STEPS = {
    "common": [
        {"name": "Model structure validated", "wait": 0.5},
        {"name": "Tensor operations compatible with TensorWrap", "wait": 0.5}
    ],
    "vision": [
        {"name": "Image processing layers identified", "wait": 0.5}
    ],
    "llm": [
        {"name": "Attention mechanisms validated", "wait": 0.5}
    ]
}

# Inference steps configuration
INFERENCE_STEPS = {
    "common": [
        {"name": "Model weights loaded", "wait": 0.5},
        {"name": "Runtime optimizations applied", "wait": 0.5}
    ],
    "vision": [
        {"name": "Image preprocessing pipeline initialized", "wait": 0.5}
    ],
    "llm": [
        {"name": "Tokenization pipeline configured", "wait": 0.5}
    ]
}

# Inference latency configuration (in milliseconds)
LATENCY_CONFIG = {
    "llm": {
        "standard": 261,
        "optimized": 73
    },
    "vision": {
        "standard": 120,
        "optimized": 45
    }
}

def detect_model_type(model_file):
    """
    Detects model type by inspecting the model file.
    
    Args:
        model_file (str): Path to the model Python file.
        
    Returns:
        str: Detected model type ('llm' or 'vision').
    """
    try:
        # Load the module to inspect it
        module_name = os.path.basename(model_file).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Create model and check its type
        model = module.create_model()
        if hasattr(model, 'model_type'):
            return model.model_type
            
        # Fallback detection based on filename
        if 'vision' in module_name.lower():
            return 'vision'
        else:
            return 'llm'
    except Exception as e:
        # Fallback to detecting from filename
        if 'vision' in model_file.lower():
            return 'vision'
        return 'llm'

def simulate_compilation(optimize=False, model_file=None):
    """
    Simulates the compilation process for the model.
    
    Args:
        optimize (bool): Whether to use optimized compilation.
        model_file (str): Path to the model file being processed.
    """
    # Detect model type from file
    model_type = detect_model_type(model_file)
    if model_type == "vision":
        model_name = "optimized_Vision.pt" if optimize else "Vision.pt"
    else:  # llm
        model_name = "optimized_LLM.pt" if optimize else "LLM.pt"


    # Determine header message based on model type and optimization
    if model_type == "vision":
        if optimize:
            header = "Compiling and optimizing Vision model with TensorWrap"
        else:
            header = "Compiling basic Vision model with TensorWrap"
    else:  # llm
        if optimize:
            header = "Compiling and optimizing LLM with TensorWrap"
        else:
            header = "Compiling basic LLM with TensorWrap"
            
    # Display colorful header
    console.print(Panel(header, style="bold blue"))
    
    # Verify model file
    console.print(f"\nVerifying model file: {model_file}")
    
    # Run common verification steps
    for step in VERIFICATION_STEPS["common"]:
        time.sleep(step["wait"])
        console.print(f"✓ {step['name']}")

    # Run model-specific verification steps
    for step in VERIFICATION_STEPS[model_type]:
        time.sleep(step["wait"])
        console.print(f"✓ {step['name']}")
    console.print("\n")
    
    # Get the appropriate steps based on model type and optimization mode
    compilation_mode = "optimized" if optimize else "standard"
    steps = COMPILATION_STEPS[model_type][compilation_mode]
    
    # Set the correct model name in the final step
    for step in steps:
        if "Compiling model to" in step["name"]:
            # Just set the full name directly rather than using replacement
            step["name"] = f"Compiling model to {model_name}"
    
    # Base time unit for step simulation
    base_time_unit = 0.5
    
    # Process each step with dynamic timing based on complexity
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        console=console
    ) as progress:
        for step in steps:
            # Scale the number of iterations and time by complexity
            complexity = step["complexity"]
            total_steps = max(3, min(10, int(complexity * 5)))
            step_time = base_time_unit * (complexity / 6) * 0.75
            
            # Create a task for this step
            task = progress.add_task(f"[cyan]{step['name']}...", total=total_steps)
            
            # Simulate processing
            for _ in range(total_steps):
                time.sleep(step_time)
                progress.update(task, advance=1)
    
    # Print completion message with rich formatting
    console.print()
    
    # Create compilation complete panel
    completion_panel = Panel(
        Text.from_markup(f"[bold green]Compilation complete![/]\n[cyan]{model_name} ready for inference[/]", justify="center"),
        border_style="green",
        box=box.DOUBLE_EDGE,
        width=60  # Fixed width that looks good
    )
    
    # Use Rich's Align component to properly center the panel
    centered_panel = Align.center(completion_panel)
    console.print(centered_panel)
    console.print()
    
    return model_name

def simulate_inference(model_name, model_type):
    """Simulates running inference on the compiled model.
    
    Args:
        model_name (str): Name of the model file to run inference on.
        model_type (str): Type of model ('llm' or 'vision').
    """
    # Determine header message based on model type
    if model_type == "vision":
        header = f"Running profiling on Vision model: {model_name}"
    else:  # llm
        header = f"Running profiling on LLM model: {model_name}"
    
    # Display colorful header in a panel to match compilation style
    console.print()
    console.print(Panel(header, style="bold blue"))
    console.print()  # Add an empty line before the progress starts
    console.print(f"Loading {model_name}...", style="cyan")  # Use consistent styling
    
    # Setup the steps with consistent formatting
    init_steps = []
    # Add common steps
    for step in INFERENCE_STEPS["common"]:
        init_steps.append({
            "name": step["name"],
            "wait": step["wait"],
            "complexity": 1.0
        })
    
    # Add model-specific steps
    for step in INFERENCE_STEPS[model_type]:
        init_steps.append({
            "name": step["name"],
            "wait": step["wait"],
            "complexity": 1.5
        })
        
    # Get appropriate latency from configuration
    is_optimized = "optimized" in model_name.lower()
    latency_type = "optimized" if is_optimized else "standard"
    latency = LATENCY_CONFIG[model_type][latency_type]
    
    # Add final inference step
    init_steps.append({
        "name": "Running model inference",
        "wait": latency / 100,  # Scale based on latency
        "complexity": 3.0
    })
    
    # Process steps with consistent progress display matching compilation
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        console=console
    ) as progress:
        for step in init_steps:
            # Scale the number of iterations and time by complexity
            complexity = step.get("complexity", 1.0)
            total_steps = max(3, min(10, int(complexity * 5)))
            step_time = step["wait"] / total_steps
            
            # Create a task for this step
            task = progress.add_task(f"[cyan]{step['name']}...", total=total_steps)
            
            # Simulate processing
            for _ in range(total_steps):
                time.sleep(step_time)
                progress.update(task, advance=1)
    
    # Build inference results content
    console.print()  # Add spacing
    
    # Create centered panel with fixed width to ensure proper centering
    term_width = get_terminal_width()
    panel_width = min(70, term_width - 4)  # Allow some margin
    
    # Basic result content
    result_content = f"[bold green]Inference complete![/]\n[cyan]Latency: {latency} milliseconds[/]"
    
    # Add optimization statistics if using optimized model
    if is_optimized:
        base_latency = LATENCY_CONFIG[model_type]["standard"]
        speedup = round(base_latency/latency, 1)
        
        # Add more detailed statistics
        result_content += f"\n\n[yellow]✨ Optimization Results ✨[/]\n"
        result_content += f"[green]Speed improvement: [bold]{speedup}x[/] faster[/]\n"
        result_content += f"[green]Latency reduction: [bold]{base_latency - latency}ms[/][/]\n"
        result_content += f"[dim]({base_latency}ms → {latency}ms)[/]"
    
    # Create styled panel
    profile_title = "Vision Model Profile" if model_type == "vision" else "LLM Model Profile"
    results_panel = Panel(
        Text.from_markup(result_content, justify="center"),
        width=panel_width,
        title=f"[bold cyan]{profile_title}[/]",
        border_style="blue",
        box=box.DOUBLE
    )
    
    # Use Rich's Align component to properly center the panel
    centered_panel = Align.center(results_panel)
    console.print(centered_panel)
    console.print()  # Add final spacing

def main():
    """
    Main entry point for the TensorWrap CLI.
    """
    # Print TensorWrap logo and header
    console.print(get_tensorwrap_logo())
    
    # Skip the intro panel here since our logo already includes it
    console.print()
    
    parser = argparse.ArgumentParser(
        description="TensorWrap: Compile and profile ML models"
    )
    parser.add_argument(
        "model_file", 
        help="Path to the model Python file"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Use optimized compilation process"
    )
    args = parser.parse_args()
    
    # Print model file information in a styled way
    console.print(f"Processing model file: [cyan]{args.model_file}[/]", style="bold", justify="center")
    
    # Run the appropriate workflow
    model_name = simulate_compilation(args.optimize, args.model_file)
    model_type = detect_model_type(args.model_file)
    simulate_inference(model_name, model_type)

if __name__ == "__main__":
    main()