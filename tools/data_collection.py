import os
from typing import Dict

import dotenv
import requests
from langchain_core.tools import tool

dotenv.load_dotenv()

@tool
def fetch_repo_metrics(repo_url: str) -> Dict:
    """Fetches various metrics for a GitHub repository using the OpenSource Observer API.

    Args:
        repo_url (str): The GitHub repository URL

    Returns:
        Dict: A dictionary containing repository metrics including:
            - activeDeveloperCount6Months: Number of active developers in last 6 months
            - closedIssueCount6Months: Number of issues closed in last 6 months
            - commitCount6Months: Number of commits in last 6 months
            - contributorCount: Total number of contributors
            - contributorCount6Months: Number of contributors in last 6 months
            - firstCommitDate: Date of first commit
            - forkCount: Number of forks
            - fulltimeDeveloperAverage6Months: Average number of full-time developers in last 6 months
            - lastCommitDate: Date of last commit
            - mergedPullRequestCount6Months: Number of PRs merged in last 6 months
            - newContributorCount6Months: Number of new contributors in last 6 months
            - openedIssueCount6Months: Number of issues opened in last 6 months
            - openedPullRequestCount6Months: Number of PRs opened in last 6 months
            - starCount: Number of stars
            - total_funders_count: Total number of funders
            - total_funding_received_usd: Total funding received in USD
            - total_funding_received_usd_6_months: Funding received in last 6 months in USD
    """
    namespace = repo_url.split("/")[-2]
    project = repo_url.split("/")[-1]

    # Execute the first query to get the projectId
    project_data = execute_graphql_query(
        find_projectId_query, {"namespace": namespace, "project": project}
    )
    project_id = project_data["data"]["oso_artifactsByProjectV1"][0]["projectId"]

    # Execute the second query to get the metrics
    metrics_data = execute_graphql_query(metrics_query, {"projectId": project_id})

    # Aggregate the funding metrics
    total_funders_count = sum(
        int(metric["totalFundersCount"])
        for metric in metrics_data["data"]["fundingMetrics"]
    )
    total_funding_received_usd = sum(
        metric["totalFundingReceivedUsd"]
        for metric in metrics_data["data"]["fundingMetrics"]
    )
    total_funding_received_usd_6_months = sum(
        metric["totalFundingReceivedUsd6Months"]
        for metric in metrics_data["data"]["fundingMetrics"]
    )

    return {
        **metrics_data["data"]["codeMetrics"][0],
        "total_funders_count": total_funders_count,
        "total_funding_received_usd": total_funding_received_usd,
        "total_funding_received_usd_6_months": total_funding_received_usd_6_months,
    }


# Function to execute a GraphQL query
def execute_graphql_query(query, variables):
    headers = {
        "Content-Type": "application/json",
    }
    if os.getenv("OPENSOURCE_OBSERVER_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('OPENSOURCE_OBSERVER_API_KEY')}"
    # Define the GraphQL endpoint
    graphql_endpoint = "https://www.opensource.observer/api/v1/graphql"
    response = requests.post(
        graphql_endpoint,
        json={"query": query, "variables": variables},
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


# Query to find the projectId for GitHub projects
find_projectId_query = """
query FindProjectByArtifact($namespace: String!, $project: String!) {
  oso_artifactsByProjectV1(
    where: {
      artifactNamespace: { _eq: $namespace }
      artifactSource: { _eq: "GITHUB" }
      artifactName: { _eq: $project }
    }
  ) {
    projectId
  }
}
"""

# Query to get metrics using projectId
metrics_query = """
query MetricsForProject($projectId: String!) {
  codeMetrics: oso_codeMetricsByProjectV1(
    where: { projectId: { _eq: $projectId } }
  ) {
    activeDeveloperCount6Months
    closedIssueCount6Months
    commitCount6Months
    contributorCount
    contributorCount6Months
    firstCommitDate
    forkCount
    fulltimeDeveloperAverage6Months
    lastCommitDate
    mergedPullRequestCount6Months
    newContributorCount6Months
    openedIssueCount6Months
    openedPullRequestCount6Months
    starCount
  }
  fundingMetrics: oso_fundingMetricsByProjectV1(
    where: { projectId: { _eq: $projectId } }
  ) {
    totalFundersCount
    totalFundingReceivedUsd
    totalFundingReceivedUsd6Months
  }
}
"""

if __name__ == "__main__":
    res = fetch_repo_metrics(
        "https://github.com/prettier-solidity/prettier-plugin-solidity"
    )
    from pprint import pprint

    pprint(res)
