import json
import logging
import os
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from IPython.display import Image
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langsmith import Client
from pydantic import BaseModel, Field

from prompts.analyzer import (
    COMMUNITY_ADVOCATE_PROMPT,
    FUNDING_STRATEGIST_PROMPT,
    PROJECT_ANALYZER_PROMPT,
)
from prompts.validator import VALIDATOR_PROMPT

logger = logging.getLogger(__name__)

from langgraph.types import Command

# from tools.data_collection import fetch_repo_metrics

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
        tools=[],
        state_modifier=SystemMessage(
            content="You are a repository analysis specialist. Collect and analyze repository metrics comprehensively."
        ),
    )

    def metrics_node(state: ComparisonState):
        """Collect metrics for both repositories"""
        logger.info(
            f"Processing repos: {state['repo_a']['url']} and {state['repo_b']['url']}"
        )

        # First repository
        repo_a_data = {
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"Analyze repository at {state['repo_a']['url']}"
                ),
                HumanMessage(content=f"Metrics: {state['repo_a']}"),
            ],
            "url": state["repo_a"]["url"],
            "metrics": state["repo_a"],
            "command": "analyze_repository",
        }
        repo_a_result = metrics_agent.invoke(repo_a_data)

        # Second repository
        repo_b_data = {
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"Analyze repository at {state['repo_b']['url']}"
                ),
                HumanMessage(content=f"Metrics: {state['repo_b']}"),
            ],
            "url": state["repo_b"]["url"],
            "metrics": state["repo_b"],
            "command": "analyze_repository",
        }
        repo_b_result = metrics_agent.invoke(repo_b_data)

        logger.info("Metrics collection complete")
        return {
            "messages": state["messages"] + [
                HumanMessage(content=repo_a_result["messages"][-1].content, name="metrics_collector"),
                HumanMessage(content=repo_b_result["messages"][-1].content, name="metrics_collector")
            ],
            "repo_a": {
                **state["repo_a"],
                    "metrics": HumanMessage(content=repo_a_result["messages"][-1].content, name="metrics_collector")
                },
            "repo_b": {
                    **state["repo_b"],
                    "metrics": HumanMessage(
                        content=repo_b_result["messages"][-1].content,
                        name="metrics_collector",
                    ),
                },
            "phase": "analyze"
        }

    return metrics_node


def create_supervisor_node():
    """Creates the supervisor node to orchestrate between analyzers"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
    
    # Remove FINISH from options since validator controls completion
    members = ["project_analyzer", "funding_strategist", "community_advocate", "validator"]
    
    system_prompt = (
        "You are a supervisor tasked with managing analysis between the"
        f" following specialists: {members}. Given the repository metrics"
        " and current analysis state, determine which specialist should analyze next."
        " Once all specialists have provided their analysis, route to the validator"
        " for verification."
    )
    
    class Router(TypedDict):
        """Worker to route to next."""
        next: Literal[*members]
    
    def supervisor_node(state: ComparisonState) -> Command[Literal[*members]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        
        # Check if all analyzers have contributed
        analyzer_names = ["project_analyzer", "funding_strategist", "community_advocate"]
        all_analyzed = all(
            any(msg.name == name for msg in state["messages"]) 
            for name in analyzer_names
        )
        
        # If all have analyzed but not validated, force route to validator
        if all_analyzed and not any(msg.name == "validator" for msg in state["messages"]):
            return Command(goto="validator")
            
        response = model.with_structured_output(Router).invoke(messages)
        return Command(goto=response["next"])
        
    return supervisor_node


def create_project_analyzer_node():
    """Creates node for project analysis"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
    
    analyzer = create_react_agent(
        model,
        tools=[],
        state_modifier=SystemMessage(content=PROJECT_ANALYZER_PROMPT)
    )
    
    def project_analyzer_node(state: ComparisonState) -> Command[Literal["supervisor"]]:
        result = analyzer.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=f'Project Analyzer: {result["messages"][-1].content}', name="project_analyzer")
                ], 
                "phase": "analyze"
            },
            goto="supervisor"
        )
        
    return project_analyzer_node


def create_funding_strategist_node():
    """Creates node for funding strategy analysis"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
    
    strategist = create_react_agent(
        model,
        tools=[],
        state_modifier=SystemMessage(content=FUNDING_STRATEGIST_PROMPT)
    )
    
    def funding_strategist_node(state: ComparisonState) -> Command[Literal["supervisor"]]:
        result = strategist.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=f'Funding Strategist: {result["messages"][-1].content}', name="funding_strategist")
                ], 
                "phase": "analyze"
            },
            goto="supervisor"
        )
        
    return funding_strategist_node


def create_community_advocate_node():
    """Creates node for community analysis"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)
    
    advocate = create_react_agent(
        model,
        tools=[],
        state_modifier=SystemMessage(content=COMMUNITY_ADVOCATE_PROMPT)
    )
    
    def community_advocate_node(state: ComparisonState) -> Command[Literal["supervisor"]]:
        
        result = advocate.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=f'Community Advocate: {result["messages"][-1].content}', name="community_advocate")
                ], 
                "phase": "analyze"
            },
            goto="supervisor"
        )
        
    return community_advocate_node


def create_validator_node():
    """Creates node for validating analysis results"""
    model = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, api_key=api_key)

    def validator_node(state: ComparisonState) -> Command[Literal["supervisor", "__end__"]]:

        class ValidationResult(BaseModel):
            is_valid: bool = Field(description="Whether the analysis is valid and complete")
            revision_needed: Optional[Literal["project_analyzer", "funding_strategist", "community_advocate"]] = Field(
                description="Which analyzer needs to revise their analysis, if any"
            )
            explanation: str = Field(description="Explanation of validation result or needed revisions")
            weight_a: Optional[float] = Field(
                description=f"Final validated (averaged) weight for {state['repo_a']['url']}"
            )
            weight_b: Optional[float] = Field(
                description=f"Final validated (averaged) weight for {state['repo_b']['url']}"
            )

        # Collect all analyses
        analyses = {
            msg.name: msg.content 
            for msg in state["messages"] 
            if msg.name in ["project_analyzer", "funding_strategist", "community_advocate"]
        }

        validation_prompt = [
            SystemMessage(content=VALIDATOR_PROMPT),
            HumanMessage(content=f"Validate the following analyses:\n{json.dumps(analyses, indent=2)}")
        ]

        result = model.with_structured_output(ValidationResult).invoke(validation_prompt)

        if result.is_valid:
            # If valid, update state and END
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "validation": result.explanation,
                                    "weights": {
                                        state["repo_a"]["url"]: result.weight_a,
                                        state["repo_b"]["url"]: result.weight_b,
                                    },
                                }
                            ),
                            name="validator",
                        )
                    ],
                    "analysis": {
                        "weights": {
                            state["repo_a"]["url"]: result.weight_a,
                            state["repo_b"]["url"]: result.weight_b,
                        },
                        "validation": result.explanation,
                        "final": True,
                    },
                    "phase": "complete",
                },
                goto=END,  # Validator decides to end the process
            )
        else:
            # If invalid, update state and return to supervisor for revision
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=json.dumps({
                                "validation": result.explanation,
                                "revision_needed": result.revision_needed
                            }),
                            name="validator"
                        )
                    ],
                    "phase": "analyze"
                },
                goto="supervisor"  # Return to supervisor to route to appropriate analyzer
            )

    return validator_node


def create_comparison_graph():
    """Creates the repository comparison workflow graph"""
    workflow = StateGraph(ComparisonState)

    # Add nodes
    workflow.add_node("metrics_collector", create_metrics_node())
    workflow.add_node("supervisor", create_supervisor_node())
    workflow.add_node("project_analyzer", create_project_analyzer_node())
    workflow.add_node("funding_strategist", create_funding_strategist_node()) 
    workflow.add_node("community_advocate", create_community_advocate_node())
    workflow.add_node("validator", create_validator_node())

    # Add edges
    workflow.add_edge(START, "metrics_collector")
    workflow.add_edge("metrics_collector", "supervisor")
    # workflow.add_edge("project_analyzer", "supervisor")
    # workflow.add_edge("funding_strategist", "supervisor")
    # workflow.add_edge("community_advocate", "supervisor")
    # workflow.add_edge("validator", "supervisor")

    return workflow.compile()


def save_visualization():
    """Save the visualization of the Agent comparison workflow"""
    graph = create_comparison_graph()
    graph_image = Image(graph.get_graph().draw_mermaid_png())
    with open("comparison_workflow.png", "wb") as f:
        f.write(graph_image.data)
    logger.info("Graph visualization saved!")


def run_comparison(repo_a: Dict, repo_b: Dict):
    """Run repository comparison with optional tracing

    Args:
        repo_a: Dict of first repository, containing url, metrics, and other relevant data
        repo_b: Dict of second repository, containing url, metrics, and other relevant data

    """

    graph = create_comparison_graph()
    # Initialize state with mock data
    initial_state = {
        "messages": [],
        "repo_a": repo_a,
        "repo_b": repo_b,
        "analysis": {},
        "phase": "collect",
    }
    # print(f"Initial state: {initial_state}")

    try:
        for event in graph.stream(initial_state, config={"recursion_limit": 25}):
            logger.info(f"Raw event: {event}")  # Debug print

            # Extract phase from the correct location in event
            phase = ""
            for key in event:
                if isinstance(event[key], dict) and "phase" in event[key]:
                    phase = event[key]["phase"]
                    break

            logger.info(f"Current phase: {phase}")

            if phase == "complete":
                logger.info("Comparison complete!")
                # Find the node output containing the analysis
                analysis = None
                for value in event.values():
                    if isinstance(value, dict) and "analysis" in value:
                        analysis = value.get("analysis", {})
                        break

                if not analysis:
                    logger.warning("No analysis found in event")
                    analysis = {"error": "No analysis results"}

                logger.info(f"Analysis results: {analysis}")

                weights = analysis.get("weights", {})
                explanation = analysis.get("validation")

                results = {
                    "weights": weights,
                    "explanation": explanation,
                    "trace_url": None,
                }

                # Add trace URL if tracing enabled
                # try:
                #     runs = client.list_runs(
                #         project_name=os.getenv("LANGCHAIN_PROJECT"),
                #         execution_order=1,
                #         error=False,
                #     )
                #     if runs:
                #         latest_run = runs[0]
                #         results["trace_url"] = (
                #             f"https://smith.langchain.com/public/{latest_run.id}/r"
                #         )
                # except Exception as e:
                #     logger.error(f"Error getting trace URL: {str(e)}")

                logger.info(f"Final results: {results}")
                return results

    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        raise


def main():
    """Main entry point"""
    # Example usage
    repo_a = {
        "level": 1,
        "language": "Go",
        "status": "indexed",
        "isFork": False,
        "createdAt": "2013-12-26",
        "updatedAt": "2024-12-29",
        "starCount": 47988,
        "forkCount": 20343,
        "numPackages": 1,
        "numDependentsInOso": 238,
        "listOfFunders": ["Gitcoin", "Optimism"],
        "totalFundingUsd": 2657310.811802441,
        "totalFundingUsdSince2023": 2496187.621307001,
        "osoDependencyRank": 0.38474672737620946,
        "numReposInSameLanguage": 325,
        "osoDependencyRankForLanguage": 0.9629629629629629,
        "url": "https://github.com/ethereum/go-ethereum",
    }
    repo_b = {
        "level": 2,
        "language": "Go",
        "status": "indexed",
        "isFork": False,
        "createdAt": "2016-08-23",
        "updatedAt": "2024-12-12",
        "starCount": 157,
        "forkCount": 47,
        "numPackages": 1,
        "numDependentsInOso": 165,
        "listOfFunders": ["Optimism", "Gitcoin"],
        "totalFundingUsd": 791769.407770362,
        "totalFundingUsdSince2023": 791769.407770362,
        "osoDependencyRank": 0.3164484917472965,
        "numReposInSameLanguage": 325,
        "osoDependencyRankForLanguage": 0.8641975308641975,
        "url": "https://github.com/ipfs/go-cid",
    }

    try:
        save_visualization()
        results = run_comparison(repo_a, repo_b)

        if not results:
            logger.info("\nError: No results returned from comparison")
            return

        logger.info("\nComparison Results:")
        logger.info("==================")

        weights = results.get("weights", {})
        if weights:
            logger.info(f"\nRelative Weights:")
            for repo_url, weight in weights.items():
                logger.info(f"{repo_url}: {weight}")
        else:
            logger.info("\nNo weights available")

        explanation = results.get("explanation")
        if explanation:
            logger.info(f"\nExplanation:\n{explanation}")
        else:
            logger.info("\nNo explanation available")

        trace_url = results.get("trace_url")
        if trace_url:
            logger.info(f"\nTrace URL: {trace_url}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
