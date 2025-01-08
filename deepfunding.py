# %%
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config import BASE_URL
from dotenv import load_dotenv
import os
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

# Define core state schema with compiler-style planning
class FundingState(TypedDict):
    messages: List[BaseMessage]
    repo_data: Dict  
    comparisons: List[Dict]
    current_phase: str
    execution_plan: List[Dict]
    results: Dict


# 1. Data Collection & Analysis Team
class DataCollectionTeam:
    def __init__(self):
        self.model_collector = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
        self.model_community = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
        
        self.agents = {
            "metrics_collector": create_react_agent(self.model_collector, [
                fetch_repo_metrics,
                analyze_code_quality,
                get_dependencies
            ]),
            "community_analyzer": create_react_agent(self.model_community, [
                get_contributor_metrics,
                analyze_issues_prs,
                get_activity_metrics
            ])
        }
    
    async def collect_data(self, state: FundingState) -> FundingState:
        """Collect comprehensive data about repositories"""
        repo_a = state["repo_data"]["repo_a"]
        repo_b = state["repo_data"]["repo_b"]
        
        # Collect metrics in parallel
        metrics_a = await self.agents["metrics_collector"].acollect(repo_a)
        metrics_b = await self.agents["metrics_collector"].acollect(repo_b)
        
        # Analyze community aspects
        community_a = await self.agents["community_analyzer"].acollect(repo_a) 
        community_b = await self.agents["community_analyzer"].acollect(repo_b)
        
        return {
            **state,
            "repo_data": {
                "repo_a": {**metrics_a, **community_a},
                "repo_b": {**metrics_b, **community_b}
            }
        }

# 2. Comparison & Analysis Team  
class ComparisonTeam:
    def __init__(self):
        self.model_impact = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
        self.model_weight = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
        self.agents = {
            "impact_analyzer": create_react_agent(self.model_impact, [
                assess_project_impact,
                compare_project_metrics
            ]),
            "weight_calculator": create_react_agent(self.model_weight, [
                calculate_relative_weights,
                normalize_comparison_scores
            ])
        }

    async def compare_repos(self, state: FundingState) -> FundingState:
        """Generate relative weights between repositories"""
        # Analyze impact and importance
        impact_scores = await self.agents["impact_analyzer"].aanalyze(
            state["repo_data"]
        )
        
        # Calculate normalized weights
        weights = await self.agents["weight_calculator"].acalculate(
            impact_scores
        )
        
        return {
            **state,
            "results": {
                "relative_weights": weights,
                "impact_analysis": impact_scores
            }
        }

# 3. Validation & Calibration Team
class ValidationTeam:
    def __init__(self):
        self.model_consistency = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
        self.model_calibration = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
        self.agents = {
            "consistency_checker": create_react_agent(self.model_consistency, [
                validate_transitivity,
                verify_weight_distribution
            ]),
            "calibration_agent": create_react_agent(self.model_calibration, [
                calibrate_with_historical,
                explain_weights
            ])
        }

    async def validate_results(self, state: FundingState) -> FundingState:
        """Validate and calibrate the comparison results"""
        # Check consistency
        consistency_result = await self.agents["consistency_checker"].avalidate(
            state["results"]
        )
        
        # Calibrate if needed
        if consistency_result["needs_calibration"]:
            calibrated_weights = await self.agents["calibration_agent"].acalibrate(
                state["results"]
            )
            state["results"]["relative_weights"] = calibrated_weights
            
        # Generate explanation
        explanation = await self.agents["calibration_agent"].aexplain(
            state["results"]
        )
        
        return {
            **state,
            "results": {
                **state["results"],
                "explanation": explanation,
                "validation": consistency_result
            }
        }

# Graph Implementation
def create_comparison_graph():
    graph = StateGraph(FundingState)
    
    # Create teams
    data_team = DataCollectionTeam()
    comparison_team = ComparisonTeam()
    validation_team = ValidationTeam()
    
    # Add nodes
    graph.add_node("collect_data", data_team.collect_data)
    graph.add_node("compare_repos", comparison_team.compare_repos)
    graph.add_node("validate_results", validation_team.validate_results)
    
    # Define edges - add START edge
    graph.set_entry_point("collect_data")
    
    graph.add_edge("collect_data", "compare_repos")
    graph.add_edge("compare_repos", "validate_results")
    
    # Add conditional edge for recalibration if needed
    def needs_recalibration(state):
        return state["results"]["validation"]["needs_calibration"]
        
    graph.add_conditional_edges(
        "validate_results",
        needs_recalibration,
        {
            True: "compare_repos",
            False: END
        }
    )
    
    return graph.compile()

# Example Usage
async def compare_repositories(repo_a: str, repo_b: str):
    graph = create_comparison_graph()
    
    initial_state = {
        "messages": [],
        "repo_data": {
            "repo_a": repo_a,
            "repo_b": repo_b
        },
        "comparisons": [],
        "current_phase": "data_collection",
        "execution_plan": [],
        "results": {}
    }
    
    result = await graph.arun(initial_state)
    return result["results"]["relative_weights"]



