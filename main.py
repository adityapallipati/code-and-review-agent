import os
import logging
import argparse
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from rich.tree import Tree
from rich import print as rprint
from src.agents.task_manager import TaskManager, TaskRequest
from dotenv import load_dotenv
import datetime
import json

# Load environment variables
load_dotenv()

# Set up logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()]
)

console = Console()
logger = logging.getLogger(__name__)

class CodeGenerationCLI:
    """Command Line Interface for the Code Generation System"""
    
    def __init__(self):
        """Initialize the CLI with TaskManager and console."""
        try:
            self.task_manager = TaskManager()
            self.console = console
            self.history = []
        except Exception as e:
            console.print(f"[red]Error initializing system: {str(e)}[/red]")
            raise

    def display_welcome_message(self):
        """Display welcome message and system information."""
        welcome_md = """
        # AI Code Generation and Review System

        This system will:
        1. Generate code based on your description
        2. Perform automated code review and security checks
        3. Allow you to review and approve the code
        4. Execute the code safely if approved

        The system checks for:
        • Code quality and best practices
        • Security vulnerabilities
        • Performance optimizations
        • Potential bugs
        • Input validation
        
        Use Ctrl+C at any time to exit.
        """
        self.console.print(Panel(Markdown(welcome_md), title="Welcome", border_style="blue"))

    def get_user_requirements(self) -> dict:
        """Get custom requirements from user."""
        self.console.print("\n[bold blue]Configure Code Generation Requirements[/bold blue]")
        
        # Set default requirements
        requirements = {
            "review_requirements": {
                "security_check": True,
                "performance_check": True,
                "quality_check": True,
                "require_tests": True
            },
            "execution_requirements": {
                "allow_file_operations": False,
                "allow_network_access": False,
                "max_runtime_seconds": 30
            }
        }
        
        try:
            # Test requirements
            if Confirm.ask("Would you like to customize the requirements?", default=False):
                requirements["review_requirements"]["require_tests"] = Confirm.ask(
                    "Require unit tests?",
                    default=True
                )
                
                requirements["review_requirements"]["performance_check"] = Confirm.ask(
                    "Perform performance analysis?",
                    default=True
                )
                
                requirements["review_requirements"]["security_check"] = Confirm.ask(
                    "Perform security checks?",
                    default=True
                )
                
                requirements["execution_requirements"]["allow_file_operations"] = Confirm.ask(
                    "Allow file system access?",
                    default=False
                )
                
                requirements["execution_requirements"]["allow_network_access"] = Confirm.ask(
                    "Allow network access?",
                    default=False
                )
                
                requirements["execution_requirements"]["max_runtime_seconds"] = int(
                    Prompt.ask(
                        "Maximum execution time (seconds)",
                        default="30"
                    )
                )
            else:
                self.console.print("[green]Using default requirements[/green]")
            
            return requirements
            
        except Exception as e:
            self.console.print(f"[red]Error configuring requirements: {str(e)}[/red]")
            self.console.print("[yellow]Using default requirements[/yellow]")
            return requirements

    def display_code(self, code: str):
        """Display generated code with syntax highlighting."""
        if not code:
            self.console.print("[yellow]No code generated yet.[/yellow]")
            return
            
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Generated Code", border_style="green"))

    def display_review_results(self, review_feedback: Optional[dict]):
        """Display code review results in a structured format."""
        if not review_feedback:
            self.console.print("[yellow]No review feedback available.[/yellow]")
            return

        # Create review tree
        review_tree = Tree("📋 Code Review Results")
        
        # Security section
        security = review_feedback.get("security", {})
        security_node = review_tree.add("🔒 Security Analysis")
        security_status = "✅" if not security.get("has_issues") else "⚠️"
        security_node.add(f"Status: {security_status} ({security.get('risk_level', 'unknown')} risk)")
        if security.get("issues"):
            issues_node = security_node.add("Issues Found:")
            for issue in security["issues"]:
                issues_node.add(f"⚠️ {issue}")
        if security.get("recommendations"):
            rec_node = security_node.add("Recommendations:")
            for rec in security["recommendations"]:
                rec_node.add(f"💡 {rec}")

        # Performance section
        performance = review_feedback.get("performance", {})
        perf_node = review_tree.add("⚡ Performance Analysis")
        perf_status = "✅" if not performance.get("has_issues") else "⚠️"
        perf_node.add(f"Status: {perf_status}")
        if performance.get("issues"):
            issues_node = perf_node.add("Issues Found:")
            for issue in performance["issues"]:
                issues_node.add(f"⚠️ {issue}")
        if performance.get("recommendations"):
            rec_node = perf_node.add("Recommendations:")
            for rec in performance["recommendations"]:
                rec_node.add(f"💡 {rec}")

        # Code Quality section
        quality = review_feedback.get("quality", {})
        quality_node = review_tree.add("📊 Code Quality")
        quality_status = "✅" if not quality.get("has_issues") else "⚠️"
        quality_node.add(f"Status: {quality_status}")
        if quality.get("metrics"):
            metrics_node = quality_node.add("Metrics:")
            for metric, value in quality["metrics"].items():
                metrics_node.add(f"{metric}: {value}")
        if quality.get("issues"):
            issues_node = quality_node.add("Issues Found:")
            for issue in quality["issues"]:
                issues_node.add(f"⚠️ {issue}")

        # Overall Status
        status_node = review_tree.add("📝 Overall Status")
        approved = "✅ Approved" if review_feedback.get("approved") else "❌ Not Approved"
        status_node.add(f"Approval Status: {approved}")
        if review_feedback.get("requires_human_review"):
            status_node.add("⚠️ Requires Human Review")
        
        # Display the tree
        self.console.print(review_tree)

    def display_execution_results(self, execution_result: Optional[dict]):
        """Display code execution results."""
        if not execution_result:
            self.console.print("[yellow]No execution results available.[/yellow]")
            return

        # Create execution results panel
        if execution_result.get("success"):
            panel_style = "green"
            title = "✅ Execution Successful"
            content = execution_result.get("output", "No output")
        else:
            panel_style = "red"
            title = "❌ Execution Failed"
            content = execution_result.get("error", "Unknown error")

        runtime = execution_result.get("runtime_seconds", 0)
        footer = f"Runtime: {runtime:.2f} seconds"

        self.console.print(Panel(
            content,
            title=title,
            subtitle=footer,
            border_style=panel_style
        ))

    def run(self):
        """Run the main CLI interaction loop."""
        try:
            self.display_welcome_message()
            
            while True:
                try:
                    # Get code description from user
                    self.console.print("\n[bold blue]What code would you like to generate?[/bold blue]")
                    self.console.print("(Describe the functionality you want, include any specific requirements)")
                    prompt = self.console.input("\n> ")
                    
                    if not prompt.strip():
                        continue
                    
                    logger.info(f"Received prompt: {prompt}")
                    
                    # Get custom requirements
                    requirements = self.get_user_requirements()
                    logger.info(f"Requirements configured: {requirements}")
                    
                    # Create task request
                    request = TaskRequest(
                        prompt=prompt,
                        review_requirements=requirements["review_requirements"],
                        execution_requirements=requirements["execution_requirements"]
                    )
                    
                    # Execute workflow with progress indication
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console,
                        transient=True  # This makes the progress bar disappear after completion
                    ) as progress:
                        task_id = progress.add_task(
                            "Generating and reviewing code...",
                            total=None
                        )
                        
                        logger.info("Executing workflow...")
                        result = self.task_manager.execute_workflow(request)
                        logger.info(f"Workflow result: {result}")
                        
                        progress.update(task_id, completed=True)
                    
                    # Handle errors
                    if "error" in result:
                        self.console.print(f"\n[red]Error:[/red] {result['error']}")
                        if Confirm.ask("Would you like to try again?", default=True):
                            continue
                        break
                    
                    # Display results
                    self.console.print("\n[bold green]Generation Complete![/bold green]")
                    self.display_code(result["generated_code"])
                    
                    self.console.print("\n[bold blue]Review Results:[/bold blue]")
                    self.display_review_results(result["review_feedback"])
                    
                    if result.get("review_feedback", {}).get("requires_human_review"):
                        if not Confirm.ask("\nWould you like to proceed with this code?", default=False):
                            continue
                    
                    if Confirm.ask("\nWould you like to execute this code?", default=False):
                        self.display_execution_results(result["execution_result"])
                    
                    if not Confirm.ask("\nWould you like to generate more code?", default=True):
                        break
                    
                except Exception as e:
                    logger.exception("Error during code generation")
                    self.console.print(f"\n[red]Error during code generation: {str(e)}[/red]")
                    if not Confirm.ask("Would you like to try again?", default=True):
                        break
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Exiting...[/yellow]")
        except Exception as e:
            self.console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")
            logger.exception("Unexpected error in CLI")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Code Generation and Review System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save interaction history"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory for saving generated code and history"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if needed
    if args.save_history:
        os.makedirs(args.output, exist_ok=True)
    
    # Create and run CLI
    try:
        cli = CodeGenerationCLI()
        cli.run()
        
        # Save history if requested
        if args.save_history and cli.history:
            history_file = os.path.join(
                args.output,
                f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(history_file, 'w') as f:
                json.dump(cli.history, f, indent=2)
                console.print(f"\n[green]History saved to: {history_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        if args.debug:
            raise

if __name__ == "__main__":
    main()