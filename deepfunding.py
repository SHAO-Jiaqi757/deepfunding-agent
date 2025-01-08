# %%
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Literal
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config import BASE_URL
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define core state schema
class FundingState(TypedDict):
    """State for the funding comparison workflow"""
    messages: List[BaseMessage]  # Chat history
    repo_data: Dict  # Repository metrics and data
    current_phase: Literal["data_collection", "comparison", "validation"]
    results: Dict  # Analysis results and explanations

# 1. Data Collection Node
def create_data_collection_node():
    """Creates the data collection node to analyze repository metrics"""
    model = ChatOpenAI(model="gpt-4", base_url=BASE_URL, api_key=api_key)
    
    # Create agent for repository analysis
    analysis_agent = create_react_agent(
        model,
        tools=[
            # Tool definitions would go here
            tool(lambda x: {"stars": 100, "forks": 50})(lambda x: "Fetch repo metrics"),
            tool(lambda x: {"quality_score": 0.8})(lambda x: "Analyze code quality"),
            tool(lambda x: {"contributors": 20})(lambda x: "Get contributor metrics"),
        ],
        system_message="""You are a repository analysis specialist. Analyze GitHub repositories 
        by examining code quality, community metrics, and development activity."""
    )

    def data_collection_node(state: FundingState):
        """Collect and analyze repository data"""
        repo_a = state["repo_data"]["repo_a"] 
        repo_b = state["repo_data"]["repo_b"]
        
        # Analyze both repositories
        analysis_a = analysis_agent.invoke({
            "messages": state["messages"],
            "repo": repo_a
        })
        
        analysis_b = analysis_agent.invoke({
            "messages": state["messages"],
            "repo": repo_b
        })

        return {
            **state,
            "repo_data": {
                "repo_a": analysis_a,
                "repo_b": analysis_b
            },
            "current_phase": "comparison"
        }
    
    return data_collection_node

# 2. Comparison Node  
def create_comparison_node():
    """Creates the node for comparing repositories"""
    model = ChatOpenAI(model="gpt-4", base_url=BASE_URL, api_key=api_key)
    
    comparison_agent = create_react_agent(
        model,
        tools=[
            # Tool definitions would go here
            tool(lambda x: {"relative_impact": 0.7})(lambda x: "Compare impact"),
            tool(lambda x: {"relative_quality": 0.6})(lambda x: "Compare quality"),
        ],
        system_message="""You are a project comparison specialist. Compare repositories 
        based on their metrics and determine their relative strengths."""
    )

    def comparison_node(state: FundingState):
        """Generate direct comparison between repositories"""
        comparison = comparison_agent.invoke({
            "messages": state["messages"],
            "repo_data": state["repo_data"]
        })
        
        # Calculate relative weights between repos (0-1 scale)
        repo_a_weight = comparison["relative_impact"] * 0.6 + comparison["relative_quality"] * 0.4
        repo_b_weight = 1 - repo_a_weight
        
        return {
            **state,
            "results": {
                "weights": {
                    "repo_a": repo_a_weight,
                    "repo_b": repo_b_weight
                },
                "comparison_details": comparison
            },
            "current_phase": "validation" 
        }
    
    return comparison_node

# 3. Validation Node
def create_validation_node():
    """Creates the validation node to verify comparison results"""
    model = ChatOpenAI(model="gpt-4", base_url=BASE_URL, api_key=api_key)
    
    validation_agent = create_react_agent(
        model,
        tools=[
            # Tool definitions would go here
            tool(lambda x: {"is_valid": True})(lambda x: "Validate weights"),
            tool(lambda x: {"explanation": "..."})(lambda x: "Generate explanation"),
        ],
        system_message="""You are a validation specialist. Verify that repository 
        comparisons are reasonable and well-justified."""
    )

    def validation_node(state: FundingState):
        """Validate the comparison results"""
        validation = validation_agent.invoke({
            "messages": state["messages"],
            "results": state["results"]
        })
        
        needs_review = not validation["is_valid"]
        
        return {
            **state,
            "results": {
                **state["results"],
                "validation": validation,
                "final_explanation": validation["explanation"]
            },
            "current_phase": "comparison" if needs_review else "complete"
        }
    
    return validation_node

def create_funding_graph():
    """Creates the repository comparison workflow graph"""
    
    # Initialize graph
    workflow = StateGraph(FundingState)
    
    # Add nodes
    workflow.add_node("collect_data", create_data_collection_node())
    workflow.add_node("compare", create_comparison_node()) 
    workflow.add_node("validate", create_validation_node())

    # Add edges
    workflow.add_edge(START, "collect_data")
    workflow.add_edge("collect_data", "compare")
    workflow.add_edge("compare", "validate")
    
    # Add conditional edge for review if needed
    def needs_review(state: FundingState):
        return state["current_phase"] == "comparison"
        
    workflow.add_conditional_edges(
        "validate",
        needs_review,
        {
            True: "compare",  # Review needed
            False: END  # Complete
        }
    )
    
    return workflow.compile()

async def compare_repositories(repo_a: str, repo_b: str):
    """Compare two repositories and determine their relative weights"""
    graph = create_funding_graph()
    
    initial_state = {
        "messages": [],
        "repo_data": {
            "repo_a": repo_a,
            "repo_b": repo_b
        },
        "current_phase": "data_collection",
        "results": {}
    }
    
    # Run comparison workflow
    for event in graph.stream(initial_state):
        phase = event.get("current_phase", "")
        print(f"Phase: {phase}")
        
        if phase == "complete":
            results = event["results"]
            print(f"\nFinal weights:")
            print(f"Repo A: {results['weights']['repo_a']:.2f}")
            print(f"Repo B: {results['weights']['repo_b']:.2f}")
            print(f"\nExplanation: {results['final_explanation']}")
            return results



