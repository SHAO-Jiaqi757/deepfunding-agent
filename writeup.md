# Deepfunding-Agent
> We believe AI Agents can make **better** funding decisions than humans.
> AI Agents collect more data than humans
> AI Agents provide better reasoning for its decision

GitHub: https://github.com/SHAO-Jiaqi757/deepfunding

This project aims to serve as a AI Agent framework for Deepfunding. 
Welcome everyone to build your own agent and contribute to this project.

## How It Works
![Workflow](./comparison_workflow.png)

1. Metrics Collector
- Gathers repository metrics from [OSO](https://docs.opensource.observer/)
- Fetch README content
- Online search for additional information
- Send metrics to Analyzer Agents
2. Multi Analyzer Agents
- Each agent reads the metrics and provides structured analysis including weights, reasoning, and confidence.
- Project Analyzer: Evaluates technical aspects and project fundamentals
- Funding Strategist: Focuses on funding history and resource allocation
- Community Advocate: Analyzes community engagement and ecosystem impact
3. Validator: Check each agent's analysis is comprehensive and well-justified. If not, send back to analyzer agent for revision.
4. Consensus: Combine results from all agents, calculate final weights

## Results
For now, we only have a small set of result for Huggingface part1. So no MSE is available.

We noticed that agent's results are quite different from human's.

### Case Study
Take a look at this extreme example. All 3 agent give opposite result from human's, they think teku should get more funding than prettier-plugin-solidity.

| Repository | Agent Score | Human Score |
|------------|-------------|-------------|
| prettier-solidity/prettier-plugin-solidity | 0.38 | 0.67 |
| consensys/teku | 0.62 | 0.33 |

### Project Analyzer
- **Weights**: prettier-plugin-solidity=0.35, teku=0.65
- **Confidence**: 0.85
- **Reasoning**: Teku demonstrates significantly higher activity and community engagement:
  - Active developers: teku(8) vs prettier-plugin-solidity(1)
  - Recent commits: teku(287) vs prettier-plugin-solidity(15)
  - Closed issues: teku(135) vs prettier-plugin-solidity(2)
  - Despite lower funding, teku shows better sustainability indicators

### Funding Strategist
- **Weights**: prettier-plugin-solidity=0.40, teku=0.60
- **Confidence**: 0.90
- **Reasoning**:
  - prettier-plugin-solidity:
    - Strong contributor base (108 total, 5 active)
    - Low recent activity (15 commits, 2 closed issues)
    - High existing funding
  - teku:
    - Higher activity (287 commits, 135 closed issues)
    - Larger community (276 total, 39 active)
    - Critical infrastructure role
    - Lower funding received

### Community Advocate
- **Weights**: prettier-plugin-solidity=0.40, teku=0.60
- **Confidence**: 0.90
- **Reasoning**:
  - prettier-plugin-solidity shows good adoption (737 stars) but concerning activity (1 active dev)
  - teku demonstrates:
    - Stronger recent activity (8 active devs, 287 commits)
    - Healthy issue resolution (135 closed)
    - Lower funding ($120k vs $399k)
    - Better sustainability indicators