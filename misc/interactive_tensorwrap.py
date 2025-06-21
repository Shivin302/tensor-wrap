#!/usr/bin/env python
"""
████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ██╗    ██╗██████╗  █████╗ ██████╗ 
╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗
   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██║ █╗ ██║██████╔╝███████║██████╔╝
   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║███╗██║██╔══██╗██╔══██║██╔═══╝ 
   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██║     
   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     
                                                                                        
Make your models faster, cheaper, and easier to scale.

Interactive UI for TensorWrap
"""

import os
import time
import random
import importlib.util
import glob
from tqdm import tqdm
import sys
import shutil
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint
from rich import box
from rich.align import Align
from rich.live import Live

# Import the original TensorWrap code for shared functionality
import tensorwrap

class InteractiveTensorWrap:
    """Interactive UI for TensorWrap that allows users to navigate and run models through a CLI interface."""
    
    def __init__(self):
        """Initialize the interactive TensorWrap UI."""
        # Get terminal width with fallback for environments without terminal
        try:
            self.terminal_width = shutil.get_terminal_size().columns
        except:
            self.terminal_width = 80  # Default fallback width
            
        self.models = []
        self.current_menu = "main"
        self.current_model = None
        self.prompt_symbol = "› "
        
        # Initialize code preview settings
        self.preview_page = 0
        self.lines_per_page = 15  # Default lines per page
        
        # Initialize rich console
        self.console = Console()
        
    def clear_screen(self):
        """Clear the terminal screen."""
        self.console.clear()
        
    def print_header(self):
        """Print the TensorWrap header."""
        logo = tensorwrap.get_tensorwrap_logo()
        self.console.print(logo)
        
        # Create panel with appropriate settings
        intro_panel = Panel(
            Text("Make your models faster, cheaper, and easier to scale.", justify="center"),
            title="[bold cyan]TensorWrap[/] - [yellow]AI Model Optimization Framework[/]",
            border_style="blue",
            width=70  # Fixed width that looks good
        )
        
        # Use Rich's Align component to properly center the panel
        centered_panel = Align.center(intro_panel)
        self.console.print(centered_panel)
        self.console.print()
        
    def scan_models(self, silent=False):
        """Scan the current directory for model files.
        
        Args:
            silent (bool): If True, don't print any status messages
        
        Returns:
            list: List of found model files
        """
        self.models = []
        py_files = glob.glob("*.py")
        
        if not silent:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Scanning for model files..."),
                console=self.console
            ) as progress:
                task = progress.add_task("Scanning", total=len(py_files))
                
                for file in py_files:
                    # Skip the main scripts
                    if file in ["tensorwrap.py", "interactive_tensorwrap.py", "__init__.py"]:
                        progress.advance(task)
                        continue
                        
                    # Check if it's potentially a model file
                    if "model" in file.lower() or self._is_model_file(file):
                        self.models.append(file)
                    
                    progress.advance(task)
        else:
            # Silent mode - no progress display
            for file in py_files:
                # Skip the main scripts
                if file in ["tensorwrap.py", "interactive_tensorwrap.py", "__init__.py"]:
                    continue
                    
                # Check if it's potentially a model file
                if "model" in file.lower() or self._is_model_file(file):
                    self.models.append(file)
                
        return self.models
        
    def _is_model_file(self, filename):
        """Check if a file is likely a model file by examining its content."""
        try:
            with open(filename, 'r') as f:
                content = f.read().lower()
                return 'create_model' in content or 'model' in content
        except:
            return False

    def print_menu(self):
        """Print the current menu based on the navigation state."""
        self.clear_screen()
        self.print_header()
        
        if self.current_menu == "main":
            self.console.print("[bold blue]MAIN MENU[/]\n")
            table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
            table.add_column("Option", style="cyan")
            table.add_column("Description", style="yellow")
            
            table.add_row("1.", "Select and run a model")
            table.add_row("2.", "Scan for models")
            # Add empty row for minimal vertical spacing
            table.add_row("", "")
            table.add_row("0.", "Exit")
            
            self.console.print(table)
            self.console.print("\nType a number and press [green]Enter[/] to continue...", style="dim")
            
        elif self.current_menu == "model_selection":
            self.console.print("[bold blue]MODEL SELECTION[/]\n")
            
            # Create a table regardless of whether models are found
            table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
            
            if not self.models:
                self.console.print("[bold red]No models found.[/] Please scan for models first.")
                # Simple table with just the back option
                table.add_column("Option", style="cyan")
                table.add_column("Description", style="yellow")
                # Add empty row for minimal vertical spacing
                table.add_row("", "")
                table.add_row("0.", "Back to main menu")
            else:
                table.add_column("Option", style="cyan")
                table.add_column("Model", style="yellow")
                table.add_column("Type", style="green")
                
                for i, model in enumerate(self.models, 1):
                    model_type = tensorwrap.detect_model_type(model)
                    table.add_row(f"{i}.", model, f"({model_type} model)")
                
                # Add empty row for minimal vertical spacing
                table.add_row("", "", "")
                table.add_row("0.", "Back to main menu", "")
            
            self.console.print(table)
            self.console.print("\nType a number and press [green]Enter[/] to select a model...", style="dim")
            
        elif self.current_menu == "run_options":
            model_type = tensorwrap.detect_model_type(self.current_model)
            self.console.print(f"[bold blue]RUN OPTIONS[/] for [yellow]{self.current_model}[/] ([green]{model_type} model[/])\n")
            
            table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
            table.add_column("Option", style="cyan")
            table.add_column("Description", style="yellow")
            
            table.add_row("1.", "Run standard compilation")
            table.add_row("2.", "Run optimized compilation")
            table.add_row("3.", "Preview model code")
            # Add empty row for minimal vertical spacing
            table.add_row("", "")
            table.add_row("0.", "Back to model selection")
            
            self.console.print(table)
            self.console.print("\nType a number and press [green]Enter[/] to continue...", style="dim")
            
        elif self.current_menu == "code_preview":
            self._display_code_preview()
    
    def _display_code_preview(self):
        """Display the code of the currently selected model file with line numbers and syntax highlighting."""
        try:
            # Safe file reading with error handling
            try:
                with open(self.current_model, 'r') as f:
                    code_content = f.read()
                    code_lines = code_content.splitlines()
            except Exception as e:
                self.console.print(f"\n[bold red]Couldn't read the file:[/] {e}")
                self.console.print("Press Enter to continue...", style="dim")
                input()
                return
                
            # Pagination setup
            total_lines = len(code_lines)
            if total_lines == 0:
                self.console.print("\n[bold yellow]The file is empty![/]")
                self.console.print("Press Enter to continue...", style="dim")
                input()
                return
                
            # Calculate page parameters - show more code at once
            self.lines_per_page = min(30, max(10, self.terminal_width // 3))  # Show more lines
            start_line = self.preview_page * self.lines_per_page
            end_line = min(start_line + self.lines_per_page, total_lines)
            
            # Create a sliced view of the code for this page
            displayed_code = "\n".join(code_lines[start_line:end_line])
            
            # Create a rich syntax object with line numbers
            syntax = Syntax(displayed_code, "python", line_numbers=True, start_line=start_line+1)
            
            # Display the code with a header panel
            self.console.print()
            self.console.print(Panel(
                syntax,
                title=f"[bold cyan]{self.current_model}[/] [dim](Lines {start_line+1}-{end_line} of {total_lines})[/]",
                border_style="blue",
                expand=False
            ))
            
            # Navigation options
            has_prev = self.preview_page > 0
            has_next = end_line < total_lines
            
            # Build navigation bar with proper spacing
            nav_table = Table.grid(padding=(0, 1))
            nav_table.add_column(style="green")
            
            # Create separate navigation options for pagination and exit
            page_options = []
            if has_prev:
                page_options.append("[cyan]p[/]-[yellow]Prev[/]")
            if has_next:
                page_options.append("[cyan]n[/]-[yellow]Next[/]")
                
            # Add pagination options if available
            if page_options:
                nav_table.add_row(f"PAGE NAVIGATION: {' | '.join(page_options)}")
                # Add an empty row for spacing
                nav_table.add_row("")
                
            # Add the return option with consistent styling (now includes Enter key)
            nav_table.add_row("EXIT: [cyan]0[/] or [cyan]Enter[/]-[yellow]Return to run options[/]")
            self.console.print(nav_table)
            
        except Exception as e:
            self.console.print(f"\n[bold red]Error displaying code preview:[/] {e}")
            self.console.print("Press Enter to continue...", style="dim")
            input()
            self.current_menu = "run_options"
    
    def handle_input(self, user_input):
        """Handle user input based on the current menu."""
        if self.current_menu == "main":
            if user_input == "1":
                if not self.models:
                    self.scan_models()
                self.current_menu = "model_selection"
            elif user_input == "2":
                models = self.scan_models()
                # Show centered results message
                result_text = Text(f"  Found {len(models)} model(s).", justify="left")
                self.console.print("\n[green]" + result_text.plain + "[/]", justify="left")
                time.sleep(1.5)
                self.current_menu = "main"
            elif user_input == "0":
                self.console.print("\n[yellow]Exiting TensorWrap Interactive...[/]")
                return False
                
        elif self.current_menu == "model_selection":
            try:
                choice = int(user_input)
                if choice == 0:
                    # Back to main menu
                    self.current_menu = "main"
                elif 1 <= choice <= len(self.models):
                    # Select model and show run options
                    self.current_model = self.models[choice - 1]
                    self.current_menu = "run_options"
            except ValueError:
                # Invalid input, just stay on the same menu
                pass
                
        elif self.current_menu == "run_options":
            if user_input == "1":
                self.run_model(optimize=False)
                input("\nPress Enter to continue...")
                self.current_menu = "run_options"
            elif user_input == "2":
                self.run_model(optimize=True)
                input("\nPress Enter to continue...")
                self.current_menu = "run_options"
            elif user_input == "0":
                self.current_menu = "model_selection"
            elif user_input == "3":
                self.preview_page = 0  # Initialize at first page
                self.current_menu = "code_preview"
                
        elif self.current_menu == "code_preview":
            # Handle code preview navigation
            if user_input.lower() == "p" or user_input == "previous":
                if self.preview_page > 0:
                    self.preview_page -= 1
            elif user_input.lower() == "n" or user_input == "next":
                # Check if there are more pages
                try:
                    with open(self.current_model, 'r') as f:
                        total_lines = len(f.readlines())
                    if (self.preview_page + 1) * self.lines_per_page < total_lines:
                        self.preview_page += 1
                    else:
                        self.console.print("\n[yellow]You've reached the end of the file.[/]")
                        time.sleep(1)
                except Exception as e:
                    # If file can't be read, go back to run options
                    self.console.print(f"\n[bold red]Can't access the file:[/] {e}")
                    time.sleep(1.5)
                    self.current_menu = "run_options"
            elif user_input == "0" or user_input.lower() == "q" or user_input == "back" or user_input == "return" or user_input == "":
                self.current_menu = "run_options"
                
        return True
    
    def run_model(self, optimize=False):
        """Run the selected model with or without optimization."""
        # Clear screen and only show the header (no options)
        self.clear_screen()
        self.print_header()
        model_name = tensorwrap.simulate_compilation(optimize, self.current_model)
        model_type = tensorwrap.detect_model_type(self.current_model)
        tensorwrap.simulate_inference(model_name, model_type)
    
    def run(self):
        """Main loop for the interactive UI."""
        # Initialize without any prints
        self.models = self.scan_models(silent=True)
        
        running = True
        while running:
            self.clear_screen()
            self.print_menu()
            user_input = input(f"\n{self.prompt_symbol}")
            running = self.handle_input(user_input)
        
        # Display centered goodbye message
        self.console.print()
        
        goodbye_panel = Panel(
            Text("[bold green]Thank you for using TensorWrap![/]", justify="center"),
            width=60,  # Fixed width that looks good
            border_style="cyan",
            box=box.DOUBLE
        )
        
        # Use Rich's Align component to properly center the panel
        centered_panel = Align.center(goodbye_panel)
        self.console.print(centered_panel)


def main():
    """Entry point for the interactive TensorWrap UI."""
    app = InteractiveTensorWrap()
    app.run()


if __name__ == "__main__":
    main()
