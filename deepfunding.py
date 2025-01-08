# %%
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Literal, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config import BASE_URL
from dotenv import load_dotenv
import os
import asyncio
import logging
from langgraph.graph.message import add_messages
from langsmith import Client
from langchain_core.tracers import ConsoleCallbackHandler
from tests.test_utils import MOCK_REPO_DATA  # Import the mock data

logger = logging.getLogger(__name__)

from tools.data_collection import (
    fetch_repo_metrics, analyze_code_quality, get_dependencies,
    get_contributor_metrics, analyze_issues_prs, get_activity_metrics
)
from tools.comparison import (
    assess_project_impact,
    compare_project_metrics,
    calculate_relative_weights,
    normalize_comparison_scores
)
from tools.validation import (
    validate_transitivity, verify_weight_distribution,
    calibrate_with_historical, explain_weights
)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

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
            get_activity_metrics
        ],
        state_modifier=SystemMessage(content="You are a repository analysis specialist. Collect and analyze repository metrics comprehensively.")
    )

    def metrics_node(state: ComparisonState):
        """Collect metrics for both repositories"""
        logger.info(f"Processing repos: {state['repo_a']['url']} and {state['repo_b']['url']}")
        
        # First repository
        repo_a_data = {
            "messages": state["messages"] + [SystemMessage(content=f"Analyze repository at {state['repo_a']['url']}")],  # Include URL in messages
            "url": state["repo_a"]["url"],
            "command": "analyze_repository"
        }
        repo_a_metrics = metrics_agent.invoke(repo_a_data)
        logger.info(f"Repo A metrics collected: {repo_a_metrics}")
        
        # Second repository
        repo_b_data = {
            "messages": state["messages"] + [SystemMessage(content=f"Analyze repository at {state['repo_b']['url']}")],  # Include URL in messages
            "url": state["repo_b"]["url"],
            "command": "analyze_repository"
        }
        repo_b_metrics = metrics_agent.invoke(repo_b_data)
        logger.info(f"Repo B metrics collected: {repo_b_metrics}")

        # Update state with metrics
        updated_state = {
            **state,
            "repo_a": {
                **state["repo_a"],
                "metrics": repo_a_metrics
            },
            "repo_b": {
                **state["repo_b"],
                "metrics": repo_b_metrics
            },
            "phase": "analyze"
        }
        
        logger.info("Metrics collection complete")
        return updated_state

    return metrics_node

def create_analysis_node():
    """Creates node for comparative analysis"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
    
    analysis_agent = create_react_agent(
        model,
        [
            assess_project_impact,
            compare_project_metrics,
            calculate_relative_weights
        ],
        state_modifier=SystemMessage(content="You are a comparison specialist. Compare repositories and determine their relative strengths.")
    )

    def analysis_node(state: ComparisonState):
        """Compare repositories and calculate relative weights"""
        # First assess impact for each repo individually
        repo_a_impact = analysis_agent.invoke({
            "messages": state["messages"] + [
                SystemMessage(content=f"Analyze impact for repository: {state['repo_a']['url']}")
            ],
            "input_data": {
                "repo_data": state["repo_a"]["metrics"]  # Pass the metrics data
            }
        })
        
        repo_b_impact = analysis_agent.invoke({
            "messages": state["messages"] + [
                SystemMessage(content=f"Analyze impact for repository: {state['repo_b']['url']}")
            ],
            "input_data": {
                "repo_data": state["repo_b"]["metrics"]  # Pass the metrics data
            }
        })
        
        # Then compare the repositories
        comparison_results = analysis_agent.invoke({
            "messages": state["messages"] + [
                SystemMessage(content=f"Compare repositories: {state['repo_a']['url']} and {state['repo_b']['url']}")
            ],
            "input_data": {
                "repo_a": state["repo_a"]["metrics"],  # Pass the metrics data
                "repo_b": state["repo_b"]["metrics"]   # Pass the metrics data
            }
        })
        
        # Calculate final weights
        weights = analysis_agent.invoke({
            "messages": state["messages"] + [
                SystemMessage(content="Calculate relative weights based on impact scores")
            ],
            "input_data": {
                "impact_scores": {
                    "repo_a": repo_a_impact,
                    "repo_b": repo_b_impact
                }
            }
        })

        return {
            **state,
            "analysis": {
                "repo_a_impact": repo_a_impact,
                "repo_b_impact": repo_b_impact,
                "comparison": comparison_results,
                "weights": weights
            },
            "phase": "validate"
        }

    return analysis_node

def create_validation_node():
    """Creates node for validating comparison results"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
    
    validation_agent = create_react_agent(
        model,
        [verify_weight_distribution, explain_weights],
        state_modifier=SystemMessage(content="You are a validation specialist. Verify the comparison results are reasonable and well-explained.")
    )

    def validation_node(state: ComparisonState):
        """Validate analysis results and provide explanation"""
        validation_results = validation_agent.invoke({
            "messages": state["messages"] + [
                SystemMessage(content=f"Validate analysis for repositories: {state['repo_a']['url']} and {state['repo_b']['url']}")
            ],
            "analysis": state["analysis"]
        })

        # Check if results need revision
        needs_revision = validation_results.get("needs_revision", False)
        
        return {
            **state,
            "analysis": {
                **state["analysis"],
                "validation": validation_results,
                "final": not needs_revision
            },
            "phase": "analyze" if needs_revision else "complete"
        }

    return validation_node

def create_comparison_graph():
    """Creates the repository comparison workflow graph"""
    workflow = StateGraph(ComparisonState)
    
    # Add nodes
    workflow.add_node("collect_metrics", create_metrics_node())
    workflow.add_node("analyze", create_analysis_node())
    workflow.add_node("validate", create_validation_node())

    # Add edges
    workflow.add_edge(START, "collect_metrics")
    workflow.add_edge("collect_metrics", "analyze")
    workflow.add_edge("analyze", "validate")
    
    # Add conditional edge for revision
    def needs_revision(state: ComparisonState):
        return state["phase"] == "analyze"
        
    workflow.add_conditional_edges(
        "validate",
        needs_revision,
        {
            True: "analyze",  # Revise analysis
            False: END  # Complete
        }
    )
    
    return workflow.compile()

async def run_comparison(repo_a_key: str, repo_b_key: str, trace: bool = False, visualize: bool = False):
    """Run repository comparison with optional tracing"""
    
    # Mock data for repositories
    repo_a_data = MOCK_REPO_DATA[repo_a_key]  # Use mock data for repo A
    repo_b_data = MOCK_REPO_DATA[repo_b_key]  # Use mock data for repo B

    # Set up callbacks for tracing
    callbacks = []
    if trace:
        # Add console tracing for debugging
        callbacks.append(ConsoleCallbackHandler())
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "deepfunding"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

    # Create graph
    graph = create_comparison_graph()
    print("Graph created successfully")
    
    # Save visualization if requested 
    if visualize:
        from IPython.display import Image
        graph_image = Image(graph.get_graph().draw_mermaid_png())
        with open('comparison_workflow.png', 'wb') as f:
            f.write(graph_image.data)
        print("Graph visualization saved!")

    # Initialize state with mock data
    initial_state = {
        "messages": [],
        "repo_a": {"url": f"https://github.com/{repo_a_key}", **repo_a_data},  # Include mock data
        "repo_b": {"url": f"https://github.com/{repo_b_key}", **repo_b_data},  # Include mock data
        "analysis": {},
        "phase": "collect"
    }
    print(f"Initial state: {initial_state}")

    # Stream execution with callbacks
    config = {"configurable": {"callbacks": callbacks}}
    
    try:
        async for event in graph.astream(initial_state, config=config):
            phase = event.get("phase", "")
            print(f"Current phase: {phase}")
            print(f"Current event: {event}")
            
            if phase == "complete":
                print("Comparison complete!")
                analysis = event.get("analysis", {})
                print(f"Analysis results: {analysis}")
                
                weights = analysis.get("weights", {})
                explanation = analysis.get("validation", {}).get("explanation", "No explanation available")
                
                if not weights:
                    logger.warning("No weights found in analysis results")
                    weights = {"error": "No weights calculated"}
                
                results = {
                    "relative_weights": weights,
                    "explanation": explanation,
                    "trace_url": None
                }
                
                # Add trace URL if tracing enabled
                if trace:
                    try:
                        # Get latest run from project
                        runs = client.list_runs(
                            project_name="deepfunding",
                            execution_order=1, 
                            error=False
                        )
                        if runs:
                            latest_run = runs[0]
                            results["trace_url"] = f"https://smith.langchain.com/public/{latest_run.id}/r"
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
            repo_a_key,
            repo_b_key,
            trace=True,
            visualize=False
        )
        
        if not results:
            print("\nError: No results returned from comparison")
            return
            
        print("\nComparison Results:")
        print("==================")
        
        weights = results.get("relative_weights", {})
        if weights:
            print(f"\nRelative Weights:")
            for metric, weight in weights.items():
                print(f"{metric}: {weight}")
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
    import asyncio
    asyncio.run(main())