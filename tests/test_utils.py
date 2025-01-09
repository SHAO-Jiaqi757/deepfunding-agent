from typing import Dict, List
# Mock repository URLs for testing
TEST_REPOS = {
    "repo_a": "https://github.com/test/repo-a",
    "repo_b": "https://github.com/test/repo-b"
}
# Mock data for repository metrics
MOCK_REPO_DATA = {
    "repo1": {
        "stars": 1000,
        "forks": 200,
        "watchers": 150,
        "commits": 500,
        "contributors": 30,
        "issues": {
            "open": 50,
            "closed": 200
        },
        "pull_requests": {
            "open": 10,
            "merged": 300
        }
    },
    "repo2": {
        "stars": 500,
        "forks": 100,
        "watchers": 75,
        "commits": 300,
        "contributors": 15,
        "issues": {
            "open": 25,
            "closed": 100
        },
        "pull_requests": {
            "open": 5,
            "merged": 150
        }
    }
}

# Mock implementations for repository metrics functions
def get_stars(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["stars"]

def get_forks(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["forks"]

def get_watchers(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["watchers"]

def get_commit_count(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["commits"]

def get_contributor_count(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["contributors"]

def count_open_issues(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["issues"]["open"]

def calculate_pr_velocity(repo_url: str) -> float:
    prs = MOCK_REPO_DATA[repo_url]["pull_requests"]
    return prs["merged"] / (prs["merged"] + prs["open"])

def calculate_issue_resolution_time(repo_url: str) -> float:
    # Mock average resolution time in days
    return 7.5

# Mock implementations for code quality functions
def calculate_test_coverage(repo_url: str) -> float:
    return 0.85

def calculate_doc_coverage(repo_url: str) -> float:
    return 0.75

def analyze_complexity(repo_url: str) -> Dict:
    return {
        "cyclomatic_complexity": 12.5,
        "maintainability_index": 85.0
    } 

# Mock implementations for comparison functions
def calculate_ecosystem_impact(repo_data: Dict) -> float:
    """Calculate ecosystem impact score based on stars and forks."""
    # Ensure we're accessing dictionary values correctly with defaults
    stars = repo_data.get("stars", 0)
    forks = repo_data.get("forks", 0)
    watchers = repo_data.get("watchers", 0)
    
    # Calculate impact score
    return (stars * 0.5 + forks * 0.3 + watchers * 0.2) / 1000

def assess_community_health(repo_data: Dict) -> float:
    """Calculate community health score based on contributors and PRs."""
    # Get values with defaults
    contributors = repo_data.get("contributors", 0)
    pull_requests = repo_data.get("pull_requests", {})
    merged_prs = pull_requests.get("merged", 0)
    issues = repo_data.get("issues", {})
    closed_issues = issues.get("closed", 0)
    
    # Calculate health score
    return (contributors * 0.4 + merged_prs * 0.4 + closed_issues * 0.2) / 200

def evaluate_technical_value(repo_data: Dict) -> float:
    """Evaluate technical value of the repository."""
    # Get values with defaults
    commits = repo_data.get("commits", 0)
    contributors = repo_data.get("contributors", 0)
    
    # Simple mock implementation
    base_score = 0.85
    if commits > 0 and contributors > 0:
        # Adjust score based on activity
        base_score *= (1 + (commits / 1000) * 0.1)
    
    return min(1.0, base_score)  # Cap at 1.0

def compare_activity(repo_a: Dict, repo_b: Dict) -> float:
    activity_a = repo_a["commits"] + repo_a["pull_requests"]["merged"]
    activity_b = repo_b["commits"] + repo_b["pull_requests"]["merged"]
    return activity_a / activity_b if activity_b > 0 else float('inf')

def compare_impact(repo_a: Dict, repo_b: Dict) -> float:
    impact_a = repo_a["stars"] + repo_a["forks"]
    impact_b = repo_b["stars"] + repo_b["forks"]
    return impact_a / impact_b if impact_b > 0 else float('inf')

def compare_health(repo_a: Dict, repo_b: Dict) -> float:
    health_a = repo_a["contributors"] + repo_a["pull_requests"]["merged"]
    health_b = repo_b["contributors"] + repo_b["pull_requests"]["merged"]
    return health_a / health_b if health_b > 0 else float('inf')

def normalize_weights(impact_scores: List[float]) -> List[float]:
    """Normalize weights between repositories.
    
    Args:
        impact_scores (List[float]): List of impact scores for the repositories
        
    Returns:
        List[float]: Normalized weights for each repository
    """
    # Ensure we have valid scores
    total = sum(impact_scores)
    if total <= 0:
        raise ValueError("Impact scores must be larger than 0")
    return [score / total for score in impact_scores]
    
# Mock implementations for validation functions
def check_transitivity(weights: Dict) -> bool:
    # Simple mock implementation
    return True

def find_transitivity_violations(weights: Dict) -> List:
    # Mock implementation
    return []

def determine_calibration_need(weights: Dict) -> bool:
    # Mock implementation
    return False

def verify_distribution(weights: Dict) -> bool:
    return abs(sum(weights.values()) - 1.0) < 0.001

def calculate_distribution_stats(weights: Dict) -> Dict:
    return {
        "mean": sum(weights.values()) / len(weights),
        "max": max(weights.values()),
        "min": min(weights.values())
    }

def detect_weight_anomalies(weights: Dict) -> List:
    return []

def perform_calibration(weights: Dict) -> Dict:
    # Mock implementation - return same weights
    return weights

def calculate_adjustment_factors(weights: Dict) -> Dict:
    return {k: 1.0 for k in weights.keys()}

def assess_calibration_confidence(weights: Dict) -> float:
    return 0.9 

# Mock implementations for missing functions
def count_active_contributors(repo_url: str) -> int:
    return MOCK_REPO_DATA[repo_url]["contributors"]  # Mocked to return the number of contributors

def extract_direct_dependencies(repo_url: str) -> List[str]:
    return ["dependency1", "dependency2"]  # Mocked list of direct dependencies

def extract_indirect_dependencies(repo_url: str) -> List[str]:
    return ["indirect_dependency1", "indirect_dependency2"]  # Mocked list of indirect dependencies

def build_dependency_graph(repo_url: str) -> Dict:
    """Mock implementation to build a dependency graph."""
    return {
        "nodes": ["dependency1", "dependency2"],
        "edges": [("dependency1", "dependency2")]
    }

def analyze_contribution_frequency(repo_url: str) -> float:
    """Mock implementation to analyze contribution frequency."""
    return 5.0  # Mocked value for contribution frequency

def calculate_maintainer_response_time(repo_url: str) -> float:
    """Mock implementation to calculate average maintainer response time in hours."""
    # Return a mock value representing average response time in hours
    return 24.0  # Mock 24-hour response time

# Add this line to make MOCK_REPO_DATA accessible
__all__ = ["MOCK_REPO_DATA", "get_stars", "get_forks", "get_watchers", "get_commit_count", "get_contributor_count", "count_open_issues", "calculate_pr_velocity", "calculate_issue_resolution_time", "calculate_test_coverage", "calculate_doc_coverage", "analyze_complexity", "count_active_contributors", "extract_direct_dependencies", "extract_indirect_dependencies", "calculate_maintainer_response_time"] 