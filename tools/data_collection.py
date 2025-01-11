from typing import Dict

import pandas as pd
from langchain_core.tools import tool

# Load the metrics DataFrame
metrics_df = pd.read_csv("./datasets/oso/repo_and_funding_stats.csv", index_col=0)
# Preprocess the data
funding_averages = metrics_df.groupby("oso_project_id")[
    ["gitcoin_grants_usd", "retro_funding_usd"]
].transform("mean")
metrics_df["gitcoin_grants_usd"] = funding_averages["gitcoin_grants_usd"]
metrics_df["retro_funding_usd"] = funding_averages["retro_funding_usd"]


@tool
def fetch_repo_metrics(repo_url: str) -> Dict:
    """
    Fetch key metrics from a GitHub repository.

    This function retrieves all available metrics for a specified GitHub repository
    from a pre-loaded DataFrame. The metrics include various statistics and information
    about the repository, such as stars, forks, and other relevant data fields.

    Args:
        repo_url (str): The URL of the GitHub repository to fetch metrics for.

    Returns:
        Dict: A dictionary containing all the fields present in the DataFrame for the
        specified repository. If the repository is not found, an empty dictionary is returned.
    """
    if repo_url not in metrics_df.index:
        raise ValueError(f"No metrics data found for repository: {repo_url}")
    return metrics_df.loc[repo_url].to_dict()
