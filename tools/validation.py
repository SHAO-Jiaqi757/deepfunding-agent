from langchain_core.tools import tool
from typing import Dict
from tests.test_utils import (
    check_transitivity,
    find_transitivity_violations,
    determine_calibration_need,
    verify_distribution,
    calculate_distribution_stats,
    detect_weight_anomalies,
    perform_calibration,
    calculate_adjustment_factors,
    assess_calibration_confidence
)

@tool
def validate_transitivity(weights: Dict) -> Dict:
    """Check transitivity of relative weights (if a > b and b > c, then a > c).

    Args:
        weights (Dict): A dictionary containing relative weights to be validated.

    Returns:
        Dict: A dictionary containing the results of the transitivity check:
            - "is_transitive": A boolean indicating if the weights are transitive.
            - "violations": A list of any transitivity violations found.
            - "needs_calibration": A boolean indicating if calibration is needed.
    """
    return {
        "is_transitive": check_transitivity(weights),
        "violations": find_transitivity_violations(weights),
        "needs_calibration": determine_calibration_need(weights)
    }

@tool
def verify_weight_distribution(weights: Dict) -> Dict:
    """Verify the distribution of weights for consistency.

    Args:
        weights (Dict): A dictionary containing weights to be verified.

    Returns:
        Dict: A dictionary containing the results of the weight distribution verification:
            - "is_valid": A boolean indicating if the weight distribution is valid.
            - "distribution_stats": A dictionary containing statistics about the weight distribution.
            - "anomalies": A list of any anomalies detected in the weight distribution.
    """
    return {
        "is_valid": verify_distribution(weights),
        "distribution_stats": calculate_distribution_stats(weights),
        "anomalies": detect_weight_anomalies(weights)
    }

@tool
def calibrate_with_historical(weights: Dict) -> Dict:
    """Calibrate weights using historical funding data.

    Args:
        weights (Dict): A dictionary containing weights to be calibrated.

    Returns:
        Dict: A dictionary containing the results of the calibration:
            - "calibrated_weights": The weights after calibration.
            - "adjustment_factors": The factors used to adjust the weights.
            - "confidence_score": A score indicating the confidence in the calibration.
    """
    calibrated_weights = perform_calibration(weights)
    return {
        "calibrated_weights": calibrated_weights,
        "adjustment_factors": calculate_adjustment_factors(weights),
        "confidence_score": assess_calibration_confidence(weights)
    }

@tool
def explain_weights(results: Dict) -> str:
    """Generate detailed explanation for the weight calculations.

    Args:
        results (Dict): A dictionary containing results of weight calculations.

    Returns:
        str: A detailed explanation of the weight calculations.
    """
    return generate_weight_explanation(results) 