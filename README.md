# Deepfunding-Agent
Build AI Agent to better funding public goods, specifically for [deepfunding.org](https://deepfunding.org)

## Usage
1. Install [Poetry](https://github.com/python-poetry/poetry) for Python dependency management.
2. Prepare .env
```bash
poetry install
cp .env.example .env
```
Huggingface part1 [python notebook](./mini-contest.ipynb) and [partial result](./mini-contest/dataset-agent.csv)

## How It Works
![Workflow](./comparison_workflow.png)

1. **Metrics Collector**
- Gathers repository metrics from [OSO](https://docs.opensource.observer/)
- Fetch README content
- Reasoning and Plan what to search online for additional information
- Send all metrics to Analyzer Agents
2. **Multi Analyzer Agents**
- Involving three experts: _Project Analyzer_, _Funding Strategist_, and _Community Advocate_. 
- Each agent reads the metrics and provides structured analysis including weights, reasoning, and confidence.
- Project Analyzer: Evaluates technical aspects and project fundamentals
- Funding Strategist: Focuses on funding history and resource allocation
- Community Advocate: Analyzes community engagement and ecosystem impact
3. **Validator**: Check each agent's analysis is comprehensive and well-justified. If not, send back to analyzer agent for revision.
4. **Consensus**: Combine results from all agents, calculate final weights

## Example Output

### Multi Analyzer Agents

```json
Project Analyzer: {
  "weight_a": 0.4,
  "weight_b": 0.6,
  "confidence": 0.9,
  "reasoning": "Repo A (WalletConnect) has a high number of active developers (30) and significant recent activity with over 1954 commits in the last 6 months. However, it has a large number of open issues (250) and closed issues (212), indicating potential sustainability issues and a backlog of user concerns. Despite the promising metrics, the lack of funding in the last 6 months raises red flags about future support. Meanwhile, Repo B (Tokio) has a solid star count (86203) and a more extensive contributor base (107). Although its recent activity metrics are lower than Repo A, it has received funding recently and has a robust history and community backing, suggesting strong potential for future sustainability. The relative metrics indicate that while Repo A shows immediate activity, Repo B has stronger community and financial backing, which suggests long-term viability. Given these factors, I assign a weight of 0.4 to Repo A and 0.6 to Repo B.",
  "metrics_used": [
    "activeDeveloperCount6Months",
    "closedIssueCount6Months",
    "commitCount6Months",
    "contributorCount",
    "forkCount",
    "starCount",
    "total_funding_received_usd",
    "newContributorCount6Months",
    "openedIssueCount6Months"
  ]
}
```


### Validator

```json
{"validation": "The analyses provided by all three analyzers are comprehensive, well-justified, and actionable. Each analysis covers project health metrics (star count, fork count, dependents), funding strategy (total funding), and community impact (dependency rank). They all arrive at the conclusion that Repo A deserves a higher weight due to its superior metrics. The reasoning includes concrete metrics and logical deductions supporting their claims. However, the Community Advocate's analysis could benefit from a bit more detail on how Repo B's niche role might affect future contributions or funding. Therefore, the Community Advocate should revisit their analysis to enhance clarity in that aspect.", "revision_needed": "community_advocate"}
```

### Consensus

```json
{
  "weight_a": 0.44648544059324285,
  "weight_b": 0.5535145594067573,
  "confidence": 0.9,
  "agent_influence": {
    "project_analyzer": 0.33215368880442675,
    "funding_strategist": 0.33960699996742183,
    "community_advocate": 0.3282393112281511
  },
  "metric_importance": {
    "activeDeveloperCount6Months": 0.3213927660247748,
    "closedIssueCount6Months": 0.3213927660247748,
    "commitCount6Months": 0.3213927660247748,
    "contributorCount": 0.3213927660247748,
    "forkCount": 0.3213927660247748,
    "starCount": 0.3213927660247748,
    "total_funding_received_usd": 0.3213927660247748,
    "newContributorCount6Months": 0.22332407749776817,
    "openedIssueCount6Months": 0.3213927660247748,
    "mergedPullRequestCount6Months": 0.23202309793729778,
    "openedPullRequestCount6Months": 0.23202309793729778,
    "contributorCount6Months": 0.12692690822864902
  }
}
```

## Development

### Add/Change Agents
[langchain tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

`workflow.add_node` in [deepfunding.py](./deepfunding.py)

### VsCode
Get the poetry environment path and paste it to your VsCode Python: Select Interpreter
```bash
poetry env info --path | pbcopy
```

### Poetry
Add new dependencies
```bash
poetry add <package>
```
