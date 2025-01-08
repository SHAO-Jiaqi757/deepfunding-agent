from langchain_core.tools import tool
from typing import Dict
from tests.test_utils import (
    get_stars,
    get_forks,
    get_watchers,
    get_commit_count,
    get_contributor_count,
    count_open_issues,
    calculate_pr_velocity,
    calculate_issue_resolution_time,
    calculate_test_coverage,
    calculate_doc_coverage,
    analyze_complexity
)

@tool
def fetch_repo_metrics(repo_url: str) -> Dict:
    """Fetch key metrics from a GitHub repository including stars, forks, watchers, etc.

    Args:
        repo_url (str): The URL of the GitHub repository to fetch metrics from.

    Returns:
        Dict: A dictionary containing key metrics of the repository:
            - "stars": The number of stars the repository has received.
            - "forks": The number of forks of the repository.
            - "watchers": The number of watchers of the repository.
            - "commits": The total number of commits in the repository.
            - "contributors": The total number of contributors to the repository.
    """
    return {
        "stars": get_stars(repo_url),
        "forks": get_forks(repo_url),
        "watchers": get_watchers(repo_url),
        "commits": get_commit_count(repo_url),
        "contributors": get_contributor_count(repo_url)
    }

@tool
def analyze_code_quality(repo_url: str) -> Dict:
    """Analyze code quality metrics including test coverage, documentation, etc.

    Args:
        repo_url (str): The URL of the GitHub repository to analyze.

    Returns:
        Dict: A dictionary containing code quality metrics:
            - "test_coverage": The percentage of code covered by tests.
            - "doc_coverage": The percentage of documentation coverage.
            - "code_complexity": A measure of the complexity of the codebase.
    """
    return {
        "test_coverage": calculate_test_coverage(repo_url),
        "doc_coverage": calculate_doc_coverage(repo_url),
        "code_complexity": analyze_complexity(repo_url)
    }

@tool
def get_dependencies(repo_url: str) -> Dict:
    """Extract dependency information from the repository.

    Args:
        repo_url (str): The URL of the GitHub repository to extract dependencies from.

    Returns:
        Dict: A dictionary containing dependency information:
            - "direct_deps": A list of direct dependencies.
            - "indirect_deps": A list of indirect dependencies.
            - "dep_graph": A graphical representation of the dependencies.
    """
    return {
        "direct_deps": extract_direct_dependencies(repo_url),
        "indirect_deps": extract_indirect_dependencies(repo_url),
        "dep_graph": build_dependency_graph(repo_url)
    }

@tool
def get_contributor_metrics(repo_url: str) -> Dict:
    """Analyze contributor activity and engagement.

    Args:
        repo_url (str): The URL of the GitHub repository to analyze contributor metrics.

    Returns:
        Dict: A dictionary containing contributor metrics:
            - "active_contributors": The number of active contributors.
            - "contribution_frequency": The frequency of contributions made by contributors.
            - "maintainer_responsiveness": The average response time of maintainers to contributions.
    """
    return {
        "active_contributors": count_active_contributors(repo_url),
        "contribution_frequency": analyze_contribution_frequency(repo_url),
        "maintainer_responsiveness": calculate_maintainer_response_time(repo_url)
    }

@tool
def analyze_issues_prs(repo_url: str) -> Dict:
    """Analyze issues and pull requests patterns.

    Args:
        repo_url (str): The URL of the GitHub repository to analyze.

    Returns:
        Dict: A dictionary containing analysis of issues and pull requests:
            - "open_issues": The number of open issues in the repository.
            - "pr_velocity": The velocity of pull requests being merged.
            - "issue_resolution_time": The average time taken to resolve issues.
    """
    return {
        "open_issues": count_open_issues(repo_url),
        "pr_velocity": calculate_pr_velocity(repo_url),
        "issue_resolution_time": calculate_issue_resolution_time(repo_url)
    }

@tool
def get_activity_metrics(repo_url: str) -> Dict:
    """Get repository activity metrics over time.

    Args:
        repo_url (str): The URL of the GitHub repository to analyze activity metrics.

    Returns:
        Dict: A dictionary containing activity metrics:
            - "open_issues": The number of open issues in the repository.
            - "pr_velocity": The velocity of pull requests being merged.
            - "issue_resolution_time": The average time taken to resolve issues.
    """
    return {
        "open_issues": count_open_issues(repo_url),
        "pr_velocity": calculate_pr_velocity(repo_url),
        "issue_resolution_time": calculate_issue_resolution_time(repo_url)
    } 