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
def assess_project_impact(repo_data: Dict) -> Dict:
    """Assess the overall impact and importance of the project.

    Args:
        repo_data (Dict): A dictionary containing data about the repository, 
                          including metrics related to ecosystem, community, 
                          and technical aspects.

    Returns:
        Dict: A dictionary containing the assessed impact metrics:
            - "ecosystem_impact": The calculated ecosystem impact score.
            - "community_health": The assessed community health score.
            - "technical_value": The evaluated technical value score.
    """
    return {
        "ecosystem_impact": calculate_ecosystem_impact(repo_data),
        "community_health": assess_community_health(repo_data),
        "technical_value": evaluate_technical_value(repo_data)
    }

@tool
def compare_project_metrics(repo_a: Dict, repo_b: Dict) -> Dict:
    """Compare metrics between two repositories.

    Args:
        repo_a (Dict): A dictionary containing data about the first repository.
        repo_b (Dict): A dictionary containing data about the second repository.

    Returns:
        Dict: A dictionary containing the comparison metrics:
            - "relative_activity": The relative activity score between the two repositories.
            - "relative_impact": The relative impact score between the two repositories.
            - "relative_health": The relative health score between the two repositories.
    """
    return {
        "relative_activity": compare_activity(repo_a, repo_b),
        "relative_impact": compare_impact(repo_a, repo_b),
        "relative_health": compare_health(repo_a, repo_b)
    }

@tool
def calculate_relative_weights(impact_scores: Dict) -> Dict:
    """Calculate relative weights between repositories.

    Args:
        impact_scores (Dict): A dictionary containing impact scores for the repositories.

    Returns:
        Dict: A dictionary containing:
            - "weights": The normalized weights calculated from the impact scores.
            - "confidence": The confidence score calculated from the impact scores.
    """
    return {
        "weights": normalize_weights(impact_scores),
        "confidence": calculate_confidence_score(impact_scores)
    }

@tool
def normalize_comparison_scores(scores: Dict) -> Dict:
    """Normalize comparison scores to ensure fair comparison.

    Args:
        scores (Dict): A dictionary containing scores to be normalized.

    Returns:
        Dict: A dictionary containing:
            - "normalized_scores": A dictionary of normalized scores.
            - "scaling_factor": The scaling factor used for normalization.
    """
    normalized = {}
    total = sum(scores.values())
    for key, value in scores.items():
        normalized[key] = value / total if total > 0 else 0
    return {
        "normalized_scores": normalized,
        "scaling_factor": 1.0 / total if total > 0 else 0
    } 