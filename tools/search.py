import os
from datetime import datetime, timedelta, timezone
from typing import List
from urllib.parse import urlparse

from dotenv import load_dotenv
from github import Github
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize the search tool
search_tool = TavilySearchResults(max_results=2)

class SearchQuery(BaseModel):
    queries: List[str] = Field(description="A list of search queries for the repository, maximum 2")
    reason: str = Field(description="The reason for the search queries")

def generate_search_queries(llm, repo_info: str) -> SearchQuery:
    """Use the LLM to generate search queries based on the repository URL.
    Args:
        llm: The language model to use for generating the search queries.
        repo_info: The information about the repository to generate search queries for.
    Returns:
        The search queries as a list of strings.
    """
    prompt = f"""Given the project information '{repo_info}', to broadly learn about this project and its' impact on the community, provide a list of 2 search queries. 
    """
    response = llm.with_structured_output(SearchQuery).invoke(prompt)
    return response

def parse_github_url(repo_url: str) -> tuple[str, str]:
    """Parse GitHub URL to extract owner and repo name."""
    path = urlparse(repo_url).path.strip('/')
    owner, repo = path.split('/')[:2]
    return owner, repo

def fetch_readme(repo_url: str) -> str:
    """Fetch README content directly for the given repository URL.
    Args:
        repo_url: The URL of the repository to fetch the README content from.
    Returns:
        The README content as a string.
    """
    try:
        g = Github(os.getenv('GITHUB_TOKEN'))
        owner, repo = parse_github_url(repo_url)
        repository = g.get_repo(f"{owner}/{repo}")
        readme = repository.get_readme()
        content = readme.decoded_content.decode('utf-8')

        # Clean up the content
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip badge/shield lines
            if '![' in line and '](' in line and any(x in line.lower() for x in ['badge', 'shield', 'status']):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)
    except Exception as e:
        # Fallback to search if GitHub API fails
        print(f"Failed to fetch README via GitHub API: {e}")
        return ""


def count_issues_and_prs(issues):
    """Count issues and pull requests from a list of issues.
    Args:
        issues: A list of issues from the GitHub repository.
    Returns:
        A tuple containing the count of issues and pull requests.
    """
    # GitHub's REST API considers every pull request an issue, but not every issue is a pull request.
    # For this reason, "Issues" endpoints may return both issues and pull requests in the response.
    # You can identify pull requests by the pull_request key.
    pr_count = sum(1 for issue in issues if hasattr(issue, "pull_request"))
    issue_count = len(issues) - pr_count
    return issue_count, pr_count


def fetch_repo_metrics(repo_url: str) -> dict:
    """Fetch repository metrics such as stars, forks, and pulse metrics.
    Args:
        repo_url: The URL of the repository to fetch metrics for.
    Returns:
        A dictionary containing the repository metrics.
    """
    g = Github(os.getenv("GITHUB_TOKEN"))
    owner, repo = parse_github_url(repo_url)
    repository = g.get_repo(f"{owner}/{repo}")

    # Basic metrics
    stars = repository.stargazers_count
    forks = repository.forks
    watchers = repository.watchers
    contributors = repository.get_contributors()
    open_issues_prs = repository.get_issues(state="open")
    closed_issues_prs = repository.get_issues(state="closed")

    # Pulse since last 6 months
    last_6months = datetime.now(tz=timezone.utc) - timedelta(days=180)
    last_6months_closed_issues_prs = repository.get_issues(
        state="closed", since=last_6months
    )
    last_6months_open_issues_prs = repository.get_issues(
        state="open", since=last_6months
    )

    return {
        "stars": stars,
        "forks": forks,
        "watchers": watchers,
        "contributors": contributors.totalCount,
        "total_open_issues_prs": open_issues_prs.totalCount,
        "total_closed_issues_prs": closed_issues_prs.totalCount,
        "last_6months_open_issues_prs": last_6months_open_issues_prs.totalCount,
        "last_6months_closed_issues_prs": last_6months_closed_issues_prs.totalCount,
    }


# Test the fetch_readme function
if __name__ == "__main__":
    test_url = "https://github.com/prettier-solidity/prettier-plugin-solidity"
    # readme_content = fetch_readme(test_url)
    # print("README Content:")
    # print("=" * 50)
    # print(readme_content + "...") # Print first 500 chars to avoid cluttering console
    from pprint import pprint

    pprint(fetch_repo_metrics(test_url))
