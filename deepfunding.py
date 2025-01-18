import json
import logging
import os
import uuid
from operator import add
from typing import Annotated, Dict, List, Literal, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from IPython.display import Image
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import Client
from pydantic import BaseModel, Field

from prompts.analyzer import (
    COMMUNITY_ADVOCATE_PROMPT,
    FUNDING_STRATEGIST_PROMPT,
    PROJECT_ANALYZER_PROMPT,
)
from prompts.validator import VALIDATOR_PROMPT
from tools.consensus import GraphConsensusAnalyzer
from tools.search import fetch_readme, generate_search_queries, search_tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from langgraph.types import Command

# from tools.data_collection import fetch_repo_metrics

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Add after load_dotenv()
client = Client(api_key=langchain_api_key)

# Load model name from environment
MODEL = os.getenv("MODEL")


class AgentAnalysis(BaseModel):
    """Structured analysis from each agent"""

    weight_a: float = Field(description="Weight for repo A")
    weight_b: float = Field(description="Weight for repo B")
    confidence: float = Field(description="Confidence in the analysis, 0-1")
    reasoning: str = Field(
        description="Reasoning behind the analysis, e.g. why the weight was assigned"
    )
    metrics_used: List[str] = Field(
        description="Metrics used in the analysis, e.g. starCount, forkCount, searchResults, etc."
    )


def merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """Reducer function to update dictionaries"""
    return {**dict1, **dict2}


# Define state schema for repository comparison
class ComparisonState(TypedDict):
    """Enhanced state for comparing repositories"""

    messages: Annotated[List[BaseMessage], add_messages]
    repo_a: Dict
    repo_b: Dict
    analysis: Dict
    agent_analyses: Annotated[Dict[str, AgentAnalysis], merge_dict]
    consensus_data: Optional[Dict]
    analyzers_to_run: List[str]


def create_metrics_node():
    """Creates node for collecting repository metrics with enhanced search capabilities"""

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=api_key)

    def metrics_node(state: ComparisonState):
        """Collect metrics and enhanced context for both repositories"""
        logger.info(
            f"Processing repos: {state['repo_a']['url']} and {state['repo_b']['url']}"
        )

        # Helper function to process a single repository
        def process_repository(repo_data):
            # Fetch README content first (you'll need to implement this)
            repo_url = repo_data["url"]
            readme_content = fetch_readme(repo_url)

            # Generate search queries based on README content
            search_prompt = f"""
            Based on this repository's README content, generate relevant search queries:
            
            Repository URL: {repo_url}
            README Content: {readme_content}
            """

            repo_search = generate_search_queries(llm, search_prompt)
            repo_results = []

            for query in repo_search.queries:
                logger.info(f"Searching for: {query}")
                search_results = search_tool.invoke(query)
                for result in search_results:
                    repo_results.append(result["content"])

            return {
                **repo_data,
                "readme": readme_content,
                "searchResults": repo_results,
            }

        # Process both repositories
        repo_a_processed = process_repository(state["repo_a"])
        repo_b_processed = process_repository(state["repo_b"])

        # Format analysis message
        analysis_message = f"""
        Repository Analysis Results:
        
        Repository A ({repo_a_processed['url']}):
        README: {repo_a_processed['readme']}
        Search Results: {repo_a_processed['searchResults']}
        
        Repository B ({repo_b_processed['url']}):
        README: {repo_b_processed['readme']}
        Search Results: {repo_b_processed['searchResults']}
        """

        return Command(
            update={
                "messages": state["messages"]
                + [HumanMessage(content=analysis_message)],
                "repo_a": repo_a_processed,
                "repo_b": repo_b_processed,
                "agent_analyses": {},
                "analyzers_to_run": all_analyzers,
            },
        )

    return metrics_node


def create_project_analyzer_node():
    """Creates node for project analysis"""
    model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=api_key)

    def project_analyzer_node(
        state: ComparisonState,
    ) -> Dict:  # Return Dict instead of Command
        prompt = [
            SystemMessage(content=PROJECT_ANALYZER_PROMPT),
            HumanMessage(
                content=f"Analyze the following two repositories with relevant metrics and search results: \nRepo A: {state['repo_a']}\nRepo B: {state['repo_b']}"
            ),
        ]
        result = model.with_structured_output(AgentAnalysis).invoke(prompt)
        print(f"Project Analyzer Result: {result}")

        return {
            "messages": [
                HumanMessage(
                    content=f"Project Analyzer: {json.dumps(result.dict(), indent=2)}",
                    name="project_analyzer",
                )
            ],
            "agent_analyses": {"project_analyzer": result},
        }

    return project_analyzer_node


def create_funding_strategist_node():
    """Creates node for funding strategy analysis"""
    model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=api_key)

    def funding_strategist_node(
        state: ComparisonState,
    ) -> Command:
        prompt = [
            SystemMessage(content=FUNDING_STRATEGIST_PROMPT),
            HumanMessage(
                content=f"Analyze the following repository metrics: \nRepo A: {state['repo_a']}\nRepo B: {state['repo_b']}"
            ),
        ]
        result = model.with_structured_output(AgentAnalysis).invoke(prompt)
        print(f"Funding Strategist Result: {result}")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"Funding Strategist: {json.dumps(result.model_dump(), indent=2)}",
                        name="funding_strategist",
                    )
                ],
                "agent_analyses": {
                    **state["agent_analyses"],
                    "funding_strategist": result,
                },
            },
        )

    return funding_strategist_node


def create_community_advocate_node():
    """Creates node for community analysis"""
    model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=api_key)

    def community_advocate_node(
        state: ComparisonState,
    ) -> Command:
        prompt = [
            SystemMessage(content=COMMUNITY_ADVOCATE_PROMPT),
            HumanMessage(
                content=f"Analyze the following repository metrics: \nRepo A: {state['repo_a']}\nRepo B: {state['repo_b']}"
            ),
        ]
        result = model.with_structured_output(AgentAnalysis).invoke(prompt)
        print(f"Community Advocate Result: {result}")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"Community Advocate: {json.dumps(result.model_dump(), indent=2)}",
                        name="community_advocate",
                    )
                ],
                "agent_analyses": {
                    **state["agent_analyses"],
                    "community_advocate": result,
                },
            },
        )

    return community_advocate_node


def create_validator_node():
    """Creates node for validating analysis results"""
    model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=api_key)

    def validator_node(
        state: ComparisonState,
    ) -> Command:

        class ValidationResult(BaseModel):
            is_valid: bool = Field(
                description="Whether the analysis is valid and complete"
            )
            revision_needed: Optional[
                List[
                    Literal[
                        "project_analyzer", "funding_strategist", "community_advocate"
                    ]
                ]
            ] = Field(
                description="Which analyzers need to revise their analysis, if any"
            )
            explanation: str = Field(
                description="Explanation of validation result or needed revisions"
            )
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
            if msg.name
            in ["project_analyzer", "funding_strategist", "community_advocate"]
        }

        validation_prompt = [
            SystemMessage(content=VALIDATOR_PROMPT),
            HumanMessage(
                content=f"Validate the following analyses:\n{json.dumps(analyses, indent=2)}"
            ),
        ]

        result = model.with_structured_output(ValidationResult).invoke(
            validation_prompt
        )

        if result.is_valid:
            return Command(update={"analyzers_to_run": ["consensus"]})
        else:
            # Update state with analyzers needing revision
            return Command(update={"analyzers_to_run": result.revision_needed})

    return validator_node


def create_consensus_node():
    """Creates consensus node for the workflow"""
    consensus_analyzer = GraphConsensusAnalyzer()

    def consensus_node(state: ComparisonState) -> Command:
        # Extract analyses from state
        for agent, analysis in state["agent_analyses"].items():
            consensus_analyzer.add_agent_analysis(
                agent,
                analysis,
            )

        # Compute consensus
        consensus_result = consensus_analyzer.compute_consensus()

        return Command(
            update={
                "messages": state["messages"]
                + [
                    HumanMessage(
                        content=json.dumps(consensus_result.model_dump(), indent=2),
                        name="consensus",
                    )
                ],
                "weights": {
                    state["repo_a"]["url"]: consensus_result.weight_a,
                    state["repo_b"]["url"]: consensus_result.weight_b,
                },
            }
        )

    return consensus_node


all_analyzers = [
    "project_analyzer",
    "funding_strategist",
    "community_advocate",
]


# Function to determine which analyzers to run
def goto_analyzer_or_consensus(state: ComparisonState) -> Sequence[str]:
    return state.get("analyzers_to_run", [])


def create_comparison_graph():
    """Creates enhanced comparison workflow graph"""
    workflow = StateGraph(ComparisonState)

    # Add nodes
    workflow.add_node("metrics_collector", create_metrics_node())
    workflow.add_node("project_analyzer", create_project_analyzer_node())
    workflow.add_node("funding_strategist", create_funding_strategist_node())
    workflow.add_node("community_advocate", create_community_advocate_node())
    workflow.add_node("validator", create_validator_node())
    workflow.add_node("consensus", create_consensus_node())

    # Define edges
    workflow.add_edge(START, "metrics_collector")
    for analyzer in all_analyzers:
        workflow.add_edge("metrics_collector", analyzer)
    # Connect analyzers to validator
    workflow.add_edge(all_analyzers, "validator")
    # Conditional branching to analyzers
    workflow.add_conditional_edges(
        "validator",
        goto_analyzer_or_consensus,
        all_analyzers + ["consensus"],
    )
    workflow.add_edge("consensus", END)

    # Compile the workflow
    compiled_workflow = workflow.compile()

    return compiled_workflow


def save_visualization():
    """Save the visualization of the Agent comparison workflow"""
    graph = create_comparison_graph()
    graph_image = Image(graph.get_graph().draw_mermaid_png())
    with open("comparison_workflow.png", "wb") as f:
        f.write(graph_image.data)
    print("Graph visualization saved!")
    logger.info("Graph visualization saved!")


def run_comparison(repo_a: Dict, repo_b: Dict) -> Dict:
    """LLM Agents compare two repositories.
    
    Args:
        repo_a (Dict): First repository containing url, metrics, and other relevant data
        repo_b (Dict): Second repository containing url, metrics, and other relevant data

    Returns:
        Dict: Results containing repository weights and trace URL

    Example:
        >>> repo_a = {
                "url": "https://github.com/ethereum/go-ethereum",
                "starCount": 47988,
                "forkCount": 20343
            }
        >>> repo_b = {
                "url": "https://github.com/ipfs/go-cid", 
                "starCount": 157,
                "forkCount": 47
            }
        >>> result = run_comparison(repo_a, repo_b)
        >>> result
        {
            'weights': {
                'https://github.com/ethereum/go-ethereum': 0.8,
                'https://github.com/ipfs/go-cid': 0.2
            },
            'trace_url': 'https://smith.langchain.com/public/ece07a96-f9ab-4490-bda4-41ad180bcfec/r'
        }
    """

    graph = create_comparison_graph()
    results = {}
    run_id = uuid.uuid4()
    initial_state = {
        "messages": [],
        "repo_a": repo_a,
        "repo_b": repo_b,
        "analysis": {},
    }
    for event in graph.stream(
        initial_state, config={"recursion_limit": 25, "run_id": run_id}
    ):
        logger.info(f"Raw event: {event}")
        results["weights"] = event.get("weights")

    results["trace_url"] = client.share_run(run_id)
    wait_for_all_tracers()
    logger.info(f"Final results: {results}")
    return results


def main():
    """Main entry point"""
    logging.basicConfig(force=True)

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

    save_visualization()
    results = run_comparison(repo_a, repo_b)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
