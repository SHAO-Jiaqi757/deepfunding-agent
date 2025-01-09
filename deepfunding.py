import logging
import os
from typing import Annotated, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from IPython.display import Image
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langsmith import Client
from pydantic import BaseModel, Field

from tests.test_utils import MOCK_REPO_DATA  # Import the mock data

logger = logging.getLogger(__name__)

from langgraph.types import Command
from tenacity import retry, stop_after_attempt, wait_exponential

from tools.data_collection import (
    analyze_code_quality,
    analyze_issues_prs,
    fetch_repo_metrics,
    get_activity_metrics,
    get_contributor_metrics,
    get_dependencies,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Add after load_dotenv()
client = Client(api_key=langchain_api_key)


# Define state schema for repository comparison
class ComparisonState(TypedDict):
    """State for comparing two repositories"""

    messages: Annotated[List[BaseMessage], add_messages]  # Chat history with reducer
    repo_a: Dict  # First repository data
    repo_b: Dict  # Second repository data
    analysis: Dict  # Analysis results
    phase: Literal["collect", "analyze", "validate", "complete"]


def create_metrics_node():
    """Creates node for collecting repository metrics"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)

    metrics_agent = create_react_agent(
        model,
        [
            fetch_repo_metrics,
            analyze_code_quality,
            get_dependencies,
            get_contributor_metrics,
            analyze_issues_prs,
            get_activity_metrics,
        ],
        state_modifier=SystemMessage(
            content="You are a repository analysis specialist. Collect and analyze repository metrics comprehensively."
        ),
    )

    def metrics_node(state: ComparisonState) -> Command[Literal["analyzer"]]:
        """Collect metrics for both repositories"""
        logger.info(
            f"Processing repos: {state['repo_a']['url']} and {state['repo_b']['url']}"
        )

        # First repository
        repo_a_data = {
            "messages": state["messages"]
            + [
                SystemMessage(content=f"Analyze repository at {state['repo_a']['url']}")
            ],
            "url": state["repo_a"]["url"],
            "command": "analyze_repository",
        }
        repo_a_result = metrics_agent.invoke(repo_a_data)

        # Second repository
        repo_b_data = {
            "messages": state["messages"]
            + [
                SystemMessage(content=f"Analyze repository at {state['repo_b']['url']}")
            ],
            "url": state["repo_b"]["url"],
            "command": "analyze_repository",
        }
        repo_b_result = metrics_agent.invoke(repo_b_data)

        logger.info("Metrics collection complete")
        return Command(
            update={
                "repo_a": {
                    **state["repo_a"],
                    "metrics": HumanMessage(
                        content=repo_a_result["messages"][-1].content,
                        name="metrics_collector",
                    ),
                },
                "repo_b": {
                    **state["repo_b"],
                    "metrics": HumanMessage(
                        content=repo_b_result["messages"][-1].content,
                        name="metrics_collector",
                    ),
                },
                "phase": "analyze",
            },
            goto="analyzer",
        )

    return metrics_node


def create_analysis_node():
    """Creates node for comparative analysis"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)

    analysis_agent = create_react_agent(
        model,
        tools=[],
        state_modifier=SystemMessage(
            content="You are a comparison specialist. Compare repositories and determine their relative strengths. \
            Note that the related weights should be sum to 1."
        ),
    )

    def analysis_node(state: ComparisonState) -> Command[Literal["validator"]]:
        """Compare repositories and calculate relative weights"""
        print("debug >>> : state", state)

        repo_a_result = analysis_agent.invoke(
            {
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"Analyze impact for repository: {state['repo_a']['url']}"
                    ),
                    state["repo_a"]["metrics"],
                ]
            }
        )

        repo_b_result = analysis_agent.invoke(
            {
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"Analyze impact for repository: {state['repo_b']['url']}"
                    ),
                    state["repo_b"]["metrics"],
                ]
            }
        )

        # Calculate final weights
        weights_result = analysis_agent.invoke(
            {
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"Calculate relative weights (two float numbers a and b) for the two repositories. \
                    The final output is two float numbers that sum to 1.0. \
                    Example ouput: The relative wight for {state['repo_a']['url']} is a, the relative wight for {state['repo_b']['url']} is b."
                    ),
                    HumanMessage(
                        content=repo_a_result["messages"][-1].content, name="analyzer"
                    ),
                    HumanMessage(
                        content=repo_b_result["messages"][-1].content, name="analyzer"
                    ),
                ]
            }
        )

        return Command(
            update={
                "analysis": {
                    "repo_a": HumanMessage(
                        content=repo_a_result["messages"][-1].content, name="analyzer"
                    ),
                    "repo_b": HumanMessage(
                        content=repo_b_result["messages"][-1].content, name="analyzer"
                    ),
                    "weights": HumanMessage(
                        content=weights_result["messages"][-1].content, name="analyzer"
                    ),
                },
                "phase": "validate",
            },
            goto="validator",
        )

    return analysis_node


def create_validation_node():
    """Creates node for validating comparison results"""

    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    async def validation_node(
        state: ComparisonState,
    ) -> Command[Literal["analyzer", END]]:
        """Validate analysis results and provide explanation"""

        class ValidationResult(BaseModel):
            need_revision: bool = Field(
                description=f"Whether the analysis needs revision for repositories"
            )
            weight_a: float = Field(
                description=f"The weight of the first repository {state['repo_a']['url']}"
            )
            weight_b: float = Field(
                description=f"The weight of the second repository {state['repo_b']['url']}"
            )
            explanation: str = Field(
                description=f"Explanation whether the weights are valid or not"
            )

        validation_agent = model.with_structured_output(ValidationResult)

        try:
            validation_result = await validation_agent.ainvoke(
                [
                    SystemMessage(
                        content=f"Validate analysis for repositories: {state['repo_a']['url']} and {state['repo_b']['url']}"
                    ),
                    state["analysis"]["repo_a"],
                    state["analysis"]["repo_b"],
                    state["analysis"]["weights"],
                ]
            )

            # Check if results need revision
            needs_revision = validation_result.need_revision
        except Exception as e:
            raise e

        return Command(
            update={
                "analysis": {
                    **state["analysis"],
                    "weights": {
                        state["repo_a"]["url"]: validation_result.weight_a,
                        state["repo_b"]["url"]: validation_result.weight_b,
                    },
                    "validation": validation_result.explanation,
                    "final": not needs_revision,
                },
                "phase": "analyze" if needs_revision else "complete",
            },
            goto="analyzer" if needs_revision else END,
        )

    return validation_node


def create_comparison_graph():
    """Creates the repository comparison workflow graph"""
    workflow = StateGraph(ComparisonState)

    # Add nodes
    workflow.add_node("metrics_collector", create_metrics_node())
    workflow.add_node("analyzer", create_analysis_node())
    workflow.add_node("validator", create_validation_node())

    # Add edges
    workflow.add_edge(START, "metrics_collector")
    workflow.add_edge("metrics_collector", "analyzer")
    workflow.add_edge("analyzer", "validator")

    # Add conditional edge for revision
    def needs_revision(state: ComparisonState):
        return state["phase"] == "analyze"

    workflow.add_conditional_edges(
        "validator",
        needs_revision,
        {True: "analyzer", False: END},  # Revise analysis  # Complete
    )

    return workflow.compile()


def run_comparison(repo_a_key: str, repo_b_key: str):
    """Run repository comparison with optional tracing"""

    # Mock data for repositories
    repo_a_data = MOCK_REPO_DATA[repo_a_key]  # Use mock data for repo A
    repo_b_data = MOCK_REPO_DATA[repo_b_key]  # Use mock data for repo B

    # Create graph
    graph = create_comparison_graph()
    print("Graph created successfully")

    # Save visualization if requested
    graph_image = Image(graph.get_graph().draw_mermaid_png())
    with open("comparison_workflow.png", "wb") as f:
        f.write(graph_image.data)
        print("Graph visualization saved!")

    # Initialize state with mock data
    initial_state = {
        "messages": [],
        "repo_a": {"url": f"https://github.com/{repo_a_key}", **repo_a_data},
        "repo_b": {"url": f"https://github.com/{repo_b_key}", **repo_b_data},
        "analysis": {},
        "phase": "collect",
    }
    print(f"Initial state: {initial_state}")

    try:
        for event in graph.stream(initial_state):
            print(f"Raw event: {event}")  # Debug print

            # Extract phase from the correct location in event
            phase = ""
            for key in event:
                if isinstance(event[key], dict) and "phase" in event[key]:
                    phase = event[key]["phase"]
                    break

            print(f"Current phase: {phase}")

            if phase == "complete":
                print("Comparison complete!")
                # Find the node output containing the analysis
                analysis = None
                for value in event.values():
                    if isinstance(value, dict) and "analysis" in value:
                        analysis = value.get("analysis", {})
                        break

                if not analysis:
                    logger.warning("No analysis found in event")
                    analysis = {"error": "No analysis results"}

                print(f"Analysis results: {analysis}")

                weights = analysis.get("weights", {})
                explanation = analysis.get("validation")

                results = {
                    "relative_weights": weights,
                    "explanation": explanation,
                    "trace_url": None,
                }

                # Add trace URL if tracing enabled
                try:
                    runs = client.list_runs(
                        project_name=os.getenv("LANGCHAIN_PROJECT"),
                        execution_order=1,
                        error=False,
                    )
                    if runs:
                        latest_run = runs[0]
                        results["trace_url"] = (
                            f"https://smith.langchain.com/public/{latest_run.id}/r"
                        )
                except Exception as e:
                    logger.error(f"Error getting trace URL: {str(e)}")

                print(f"Final results: {results}")
                return results

    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        raise


async def main():
    """Main entry point"""
    # Example usage
    repo_a_key = "repo1"  # Use the key for repo A
    repo_b_key = "repo2"  # Use the key for repo B

    try:
        results = await run_comparison(
            repo_a_key, repo_b_key, trace=True, visualize=False
        )

        if not results:
            print("\nError: No results returned from comparison")
            return

        print("\nComparison Results:")
        print("==================")

        weights = results.get("relative_weights", {})
        if weights:
            print(f"\nRelative Weights:")
            for repo_url, weight in weights.items():
                print(f"{repo_url}: {weight}")
        else:
            print("\nNo weights available")

        explanation = results.get("explanation")
        if explanation:
            print(f"\nExplanation:\n{explanation}")
        else:
            print("\nNo explanation available")

        trace_url = results.get("trace_url")
        if trace_url:
            print(f"\nTrace URL: {trace_url}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
