import os
import yaml
import logging
import tempfile
import subprocess
from typing import TypedDict, Sequence, Any, Dict, Callable
from typing_extensions import TypeAlias
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import StateGraph
import datetime

# Local imports
from src.agents.code_generator import CodeGenerator
from src.agents.code_reviewer import CodeReviewer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
NodeType: TypeAlias = str

class TaskRequest(BaseModel):
    """Input model for task execution"""
    prompt: str = Field(..., description="Description of the code to generate")
    max_attempts: int = Field(default=3, description="Maximum generation attempts")
    timeout_seconds: int = Field(default=300, description="Timeout for entire workflow")
    review_requirements: dict = Field(
        default_factory=lambda: {
            "security_check": True,
            "performance_check": True,
            "quality_check": True,
            "require_tests": True
        }
    )
    execution_requirements: dict = Field(
        default_factory=lambda: {
            "allow_file_operations": False,
            "allow_network_access": False,
            "max_runtime_seconds": 30
        }
    )

class WorkflowState(TypedDict):
    """Complete workflow state definition"""
    messages: Sequence[dict]
    current_code: str | None
    next_step: str
    attempts: int
    status: dict
    errors: list[str]
    review_feedback: dict | None
    execution_result: dict | None
    human_feedback: dict | None
    metadata: dict
    route: str | None

class TaskManager:
    """Manages the code generation, review, and execution workflow"""
    
    def __init__(self):
        """Initialize the task manager."""
        self.config = self._load_config()
        self.workflow_graph = self._build_workflow_graph()
        logger.info("TaskManager initialized with workflow graph")

    def _load_config(self) -> dict:
        """Load configuration from yaml file."""
        try:
            with open("src/config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
                logger.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _build_workflow_graph(self) -> StateGraph:
        """Build the workflow graph with nodes and conditional edges."""
        # Create the graph
        graph = StateGraph(WorkflowState)
        
        # Define the next step function
        def get_next_step(state: WorkflowState) -> dict:
            """Determine the next step based on current state."""
            current_step = state.get("next_step", "generate")
            logger.info(f"Current step: {current_step}")

            next_step = current_step  # Default to same step
            
            if current_step == "generate":
                if state.get("errors"):
                    next_step = "error"
                elif state["status"].get("generation") is True:
                    next_step = "review"
                # else stay in generate

            elif current_step == "review":
                review_feedback = state.get("review_feedback", {})
                if state.get("errors"):
                    next_step = "error"
                elif review_feedback.get("requires_human_review"):
                    next_step = "await_human_input"
                elif review_feedback.get("approved"):
                    next_step = "prepare_execution"
                else:
                    next_step = "generate"

            elif current_step == "await_human_input":
                if state.get("errors"):
                    next_step = "error"
                elif state["human_feedback"].get("approved"):
                    next_step = "prepare_execution"
                else:
                    next_step = "generate"

            elif current_step == "prepare_execution":
                if state.get("errors"):
                    next_step = "error"
                elif state["status"].get("execution_prepared"):
                    next_step = "execute"
                else:
                    next_step = "error"

            elif current_step == "execute":
                if state.get("errors"):
                    next_step = "error"
                elif state["execution_result"].get("success"):
                    next_step = "complete"
                else:
                    next_step = "error"

            logger.info(f"Moving from {current_step} to {next_step}")
            return {"next": next_step}

        # Add nodes
        nodes = {
            "generate": self._generation_node,
            "review": self._review_node,
            "await_human_input": self._human_input_node,
            "prepare_execution": self._preparation_node,
            "execute": self._execution_node,
            "complete": self._complete_workflow,
            "error": self._handle_error,
        }

        # Add all nodes to graph
        for name, func in nodes.items():
            graph.add_node(name, func)

        # Add edge for each node to router
        graph.add_node("router", get_next_step)
        
        # Connect each node to router
        for name in nodes.keys():
            if name not in ["complete", "error"]:
                graph.add_edge(name, "router")
        
        # Connect router to all nodes
        for name in nodes.keys():
            graph.add_edge("router", name)
        
        # Set entry point
        graph.set_entry_point("generate")
        
        logger.info("Workflow graph built successfully")
        return graph.compile()
    
    def _generation_node(self, state: WorkflowState) -> WorkflowState:
        """Handle code generation."""
        logger.info("Starting code generation...")
        generator = CodeGenerator()
        try:
            state = generator.generate(state)
            logger.info(f"Generation complete. Success: {state['status'].get('generation', False)}")
            state["next_step"] = "generate"  # Will be updated by router if successful
            return state
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            state["errors"].append(str(e))
            state["next_step"] = "error"
            return state

    def _review_node(self, state: WorkflowState) -> WorkflowState:
        """Handle code review."""
        logger.info("Starting code review...")
        reviewer = CodeReviewer()
        try:
            state = reviewer.review(state)
            logger.info(f"Review complete. Feedback: {state.get('review_feedback', {}).get('approved', False)}")
            state["next_step"] = "review"  # Will be updated by router based on review results
            return state
        except Exception as e:
            logger.error(f"Error in code review: {e}")
            state["errors"].append(str(e))
            state["next_step"] = "error"
            return state

    def _human_input_node(self, state: WorkflowState) -> WorkflowState:
        """Handle human input/approval step."""
        logger.info("Awaiting human input...")
        state["human_feedback"] = {
            "approved": True,
            "feedback": "Automated approval for demonstration",
            "timestamp": str(datetime.datetime.now())
        }
        logger.info("Human input received")
        state["next_step"] = "await_human_input"  # Will be updated by router
        return state

    def _preparation_node(self, state: WorkflowState) -> WorkflowState:
        """Prepare code for execution."""
        logger.info("Preparing for execution...")
        state["status"]["execution_prepared"] = True
        state["next_step"] = "prepare_execution"  # Will be updated by router
        return state

    def _execution_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the approved code."""
        logger.info("Starting code execution...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as tmp:
                tmp.write(state["current_code"])
                tmp.flush()
                result = subprocess.run(
                    ['python', tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                execution_result = {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else None,
                    "runtime_seconds": 0.1
                }
        except Exception as e:
            execution_result = {
                "success": False,
                "output": None,
                "error": str(e),
                "runtime_seconds": 0
            }
            
        state["execution_result"] = execution_result
        state["next_step"] = "execute"  # Will be updated by router
        logger.info(f"Execution complete. Success: {execution_result['success']}")
        return state

    def route_after_generation(self, state: WorkflowState) -> str:
        """Route after code generation."""
        logger.info("Routing after generation...")
        logger.info(f"Current state: {state['status']}")
        
        if state["errors"]:
            logger.info("Routing to error due to errors")
            return "error"
            
        if state["attempts"] >= self.config["code_generator"].get("max_attempts", 3):
            logger.info("Routing to error due to max attempts")
            return "error"
            
        if state["messages"][-1].get("metadata", {}).get("requires_review", False):
            logger.info("Routing to human review")
            return "await_human_input"
            
        if state["status"].get("generation") is True:
            logger.info("Routing to review")
            return "review"
            
        logger.info("Routing back to generate")
        return "generate"

    def route_after_review(self, state: WorkflowState) -> str:
        """Route after code review."""
        logger.info("Routing after review...")
        if state["errors"]:
            return "error"
        review_feedback = state.get("review_feedback", {})
        if review_feedback.get("requires_human_review"):
            return "await_human_input"
        if not review_feedback.get("approved"):
            return "generate"
        return "prepare_execution"

    def route_after_human_input(self, state: WorkflowState) -> str:
        """Route after human input."""
        logger.info("Routing after human input...")
        if state["errors"]:
            return "error"
        if not state["human_feedback"].get("approved", False):
            return "generate"
        return "prepare_execution"

    def route_after_preparation(self, state: WorkflowState) -> str:
        """Route after preparation."""
        logger.info("Routing after preparation...")
        if state["errors"]:
            return "error"
        if state["status"].get("execution_prepared"):
            return "execute"
        return "error"

    def route_after_execution(self, state: WorkflowState) -> str:
        """Route after execution."""
        logger.info("Routing after execution...")
        if state["errors"]:
            return "error"
        if state["execution_result"].get("success"):
            return "complete"
        return "error"

    def _complete_workflow(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow completion."""
        logger.info("Completing workflow...")
        state["metadata"]["end_time"] = str(datetime.datetime.now())
        state["next_step"] = "complete"
        return state

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors."""
        logger.error(f"Handling errors: {state['errors']}")
        state["metadata"]["end_time"] = str(datetime.datetime.now())
        state["next_step"] = "error"
        return state

    @traceable
    def execute_workflow(self, request: TaskRequest) -> dict:
        """Execute the full workflow based on the request."""
        try:
            logger.info(f"Starting workflow with prompt: {request.prompt}")
            
            # Initialize workflow state
            initial_state = WorkflowState(
                messages=[{"role": "human", "content": request.prompt}],
                current_code=None,
                next_step="generate",
                attempts=0,
                status={},
                errors=[],
                review_feedback=None,
                execution_result=None,
                human_feedback=None,
                metadata={
                    "start_time": str(datetime.datetime.now()),
                    "request_config": request.dict(),
                    "workflow_id": os.urandom(16).hex()
                }
            )
            
            logger.info("Starting workflow execution...")
            logger.info(f"Initial state: {initial_state}")
            
            try:
                final_state = self.workflow_graph.invoke(initial_state)
                logger.info("Workflow completed successfully")
                logger.info(f"Final state: {final_state}")
            except Exception as e:
                logger.error(f"Error during graph execution: {e}")
                raise
            
            # Prepare response
            response = {
                "status": "success" if not final_state["errors"] else "error",
                "generated_code": final_state["current_code"],
                "review_feedback": final_state["review_feedback"],
                "execution_result": final_state["execution_result"],
                "errors": final_state["errors"],
                "metadata": final_state["metadata"]
            }
            
            logger.info(f"Workflow response prepared: {response['status']}")
            return response
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "error": str(e),
                "status": "error"
            }