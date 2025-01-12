from langchain_core.tools import tool
from typing import Dict
from tests.test_utils import (
    calculate_ecosystem_impact,
    assess_community_health,
    evaluate_technical_value,
    compare_activity,
    compare_impact,
    compare_health,
    normalize_weights
)
import json
from pydantic import BaseModel, Field

@tool
def assess_project_impact(metrics: Dict) -> Dict:
    """Assess the overall impact and importance of the project.

    Args:
        metrics (Dict): repository metrics and data to analyze

    Returns:
        Dict: A dictionary containing the assessed impact metrics
    """
    print("debug >>> : metrics", metrics)
    metrics = json.loads(metrics)
    try:
        return {
            "ecosystem_impact": calculate_ecosystem_impact(metrics),
            "community_health": assess_community_health(metrics),
            "technical_value": evaluate_technical_value(metrics)
        }
    except Exception as e:
        return {
            "error": f"Impact assessment failed: {str(e)}",
            "ecosystem_impact": 0,
            "community_health": 0,
            "technical_value": 0
        }

@tool
def compare_project_metrics(impact_scores: Dict) -> Dict:
    """Compare metrics between two repositories.

    Args:
        impact_scores (dict): contains impact scores for the repositories
         - impact_score_a (float): impact score for the first repository
         - impact_score_b (float): impact score for the second repository   

    Returns:
        Dict: A dictionary containing the comparison metrics
    """
    impact_scores = json.loads(impact_scores)
    repo_a = impact_scores.get("impact_score_a", 0)
    repo_b = impact_scores.get("impact_score_b", 0)

    if not repo_a or not repo_b:
        return {
            "error": "Missing metrics data",
            "relative_activity": 0,
            "relative_impact": 0,
            "relative_health": 0
        }

    try:
        return {
            "relative_activity": compare_activity(repo_a, repo_b),
            "relative_impact": compare_impact(repo_a, repo_b),
            "relative_health": compare_health(repo_a, repo_b)
        }
    except Exception as e:
        return {
            "error": f"Comparison failed: {str(e)}",
            "relative_activity": 0,
            "relative_impact": 0,
            "relative_health": 0
        }
        
@tool
def calculate_relative_weights(impact_scores: Dict) -> Dict:
    """Calculate relative weights between repositories.

    Args:
        impact_scores (dict): contains impact scores for the repositories
         - impact_score_a (float): impact score for the first repository
         - impact_score_b (float): impact score for the second repository

    Returns:
        Dict: A dictionary containing:
            - "weights": List of normalized weights (float) for each repository
    """
    print("debug >>> : impact_scores", impact_scores)
    impact_score_a = impact_scores.get("impact_score_a", 0)
    impact_score_b = impact_scores.get("impact_score_b", 0)
    print("debug >>> : impact_score_a", impact_score_a)
    print("debug >>> : impact_score_b", impact_score_b)

    
    return {
        "weights": normalize_weights([impact_score_a, impact_score_b]),
    }


@tool
def normalize_comparison_scores(scores: str) -> Dict:
    """Normalize comparison scores to ensure fair comparison.

    Args:
        scores (Dict): Raw scores to be normalized

    Returns:
        Dict: A dictionary containing:
            - "normalized_scores": A dictionary of normalized scores.
            - "scaling_factor": The scaling factor used for normalization.
    """
    scores = kwargs.get("scores", {})
    normalized = {}
    total = sum(scores.values())
    
    for key, value in scores.items():
        normalized[key] = value / total if total > 0 else 0
        
    return {
        "normalized_scores": normalized,
        "scaling_factor": 1.0 / total if total > 0 else 0
    }

    
if __name__ == "__main__":
    print(assess_project_impact(MOCK_REPO_DATA))
