from langchain_core.tools import tool
from typing import Dict
from tests.test_utils import (
    calculate_ecosystem_impact,
    assess_community_health,
    evaluate_technical_value,
    compare_activity,
    compare_impact,
    compare_health,
    normalize_weights,
    calculate_confidence_score
)

@tool
def assess_project_impact(input_data: Dict) -> Dict:
    """Assess the overall impact and importance of the project.

    Args:
        input_data (Dict): A dictionary containing:
            - repo_data (Dict): Repository metrics and data

    Returns:
        Dict: A dictionary containing the assessed impact metrics
    """
    # Extract metrics data from the input structure
    metrics = input_data.get("repo_data", {})
    
    if not metrics:
        return {
            "error": "Missing repository metrics",
            "ecosystem_impact": 0,
            "community_health": 0,
            "technical_value": 0
        }

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
def compare_project_metrics(input_data: Dict) -> Dict:
    """Compare metrics between two repositories.

    Args:
        input_data (Dict): A dictionary containing:
            - repo_a (Dict): Data about the first repository
            - repo_b (Dict): Data about the second repository

    Returns:
        Dict: A dictionary containing the comparison metrics
    """
    repo_a_metrics = input_data.get("repo_a", {})
    repo_b_metrics = input_data.get("repo_b", {})
    

    if not repo_a_metrics or not repo_b_metrics:
        return {
            "error": "Missing metrics data",
            "relative_activity": 0,
            "relative_impact": 0,
            "relative_health": 0
        }

    try:
        return {
            "relative_activity": compare_activity(repo_a_metrics, repo_b_metrics),
            "relative_impact": compare_impact(repo_a_metrics, repo_b_metrics),
            "relative_health": compare_health(repo_a_metrics, repo_b_metrics)
        }
    except Exception as e:
        return {
            "error": f"Comparison failed: {str(e)}",
            "relative_activity": 0,
            "relative_impact": 0,
            "relative_health": 0
        }

@tool
def calculate_relative_weights(input_data: Dict) -> Dict:
    """Calculate relative weights between repositories.

    Args:
        input_data (Dict): A dictionary containing:
            - impact_scores (Dict): Impact scores for the repositories

    Returns:
        Dict: A dictionary containing:
            - "weights": The normalized weights calculated from the impact scores.
            - "confidence": The confidence score calculated from the impact scores.
    """
    impact_scores = input_data.get("impact_scores", {})
    
    return {
        "weights": normalize_weights(impact_scores),
        "confidence": calculate_confidence_score(impact_scores)
    }

@tool
def normalize_comparison_scores(input_data: Dict) -> Dict:
    """Normalize comparison scores to ensure fair comparison.

    Args:
        input_data (Dict): A dictionary containing:
            - scores (Dict): Raw scores to be normalized

    Returns:
        Dict: A dictionary containing:
            - "normalized_scores": A dictionary of normalized scores.
            - "scaling_factor": The scaling factor used for normalization.
    """
    scores = input_data.get("scores", {})
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
