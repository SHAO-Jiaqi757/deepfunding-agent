import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from github import Github
from urllib.parse import urlparse
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

# Test the fetch_readme function
if __name__ == "__main__":
    test_url = "https://github.com/ipfs/go-cid"
    readme_content = fetch_readme(test_url)
    print("README Content:")
    print("=" * 50)
    print(readme_content + "...") # Print first 500 chars to avoid cluttering console
