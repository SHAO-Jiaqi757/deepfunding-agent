from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import networkx as nx
import numpy as np
from pydantic import BaseModel
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class AgentAnalysis:
    """Flexible analysis structure for agents"""
    weight_a: float
    weight_b: float
    confidence: float
    reasoning: str
    metrics_used: Set[str]  # Dynamic set of metrics
    insights: List[Dict]  # List of structured insights

class ConsensusResult(BaseModel):
    """Enhanced consensus result structure"""
    weight_a: float
    weight_b: float
    confidence: float
    agent_influence: Dict[str, float]
    metric_importance: Dict[str, float]
    
class GraphConsensusAnalyzer:
    """Dynamic consensus analyzer with emergent patterns"""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.analyses: Dict[str, AgentAnalysis] = {}
        self.metric_graph = nx.Graph()  # Graph for metric relationships
        
    def add_agent_analysis(self, agent_name: str, analysis: AgentAnalysis):
        """Add agent analysis and update knowledge graphs"""
        logger.info(f"Processing analysis from {agent_name}")
        
        # Store analysis
        self.analyses[agent_name] = analysis
        
        # Update agent relationship graph
        self._update_agent_graph(agent_name)
        
        # Update metric relationship graph
        self._update_metric_graph(analysis)
        
    def _update_agent_graph(self, new_agent: str):
        """Update agent relationship graph based on analysis similarity"""
        if new_agent not in self.G:
            self.G.add_node(new_agent)
            
        for other_agent, other_analysis in self.analyses.items():
            if other_agent != new_agent:
                similarity = self._calculate_analysis_similarity(
                    self.analyses[new_agent],
                    other_analysis
                )
                self.G.add_edge(new_agent, other_agent, weight=similarity)
                self.G.add_edge(other_agent, new_agent, weight=similarity)
                
    def _update_metric_graph(self, analysis: AgentAnalysis):
        """Update metric relationship graph based on co-occurrence"""
        metrics = list(analysis.metrics_used)
        
        # Add nodes for new metrics
        for metric in metrics:
            if metric not in self.metric_graph:
                self.metric_graph.add_node(metric, frequency=1)
            else:
                self.metric_graph.nodes[metric]['frequency'] += 1
        
        # Update edges for co-occurring metrics
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                if not self.metric_graph.has_edge(metric1, metric2):
                    self.metric_graph.add_edge(metric1, metric2, weight=1)
                else:
                    self.metric_graph[metric1][metric2]['weight'] += 1
                    
    def _calculate_analysis_similarity(
        self, 
        analysis1: AgentAnalysis, 
        analysis2: AgentAnalysis
    ) -> float:
        """Calculate similarity between two analyses"""
        # Weight similarity - using weight_a and weight_b directly
        weight_diff = abs(analysis1.weight_a - analysis2.weight_a) + \
                     abs(analysis1.weight_b - analysis2.weight_b)
        weight_similarity = 1 - (weight_diff / 2)  # Normalize to [0,1]
        
        # Metric overlap
        metric_overlap = len(
            set(analysis1.metrics_used).intersection(set(analysis2.metrics_used))
        ) / len(set(analysis1.metrics_used).union(set(analysis2.metrics_used))) if analysis1.metrics_used or analysis2.metrics_used else 0.0
        
        
        # Combine similarities with confidence
        combined_similarity = (
            0.5 * weight_similarity +
            0.5 * metric_overlap
        )
        
        return combined_similarity * min(analysis1.confidence, analysis2.confidence)
    

    def compute_consensus(self) -> ConsensusResult:
        """Compute consensus using emergent patterns"""
        if not self.analyses:
            raise ValueError("No analyses available for consensus")
            
        # 1. Calculate agent influence using PageRank
        agent_influence = nx.pagerank(self.G)
        
        # 2. Identify metric clusters using community detection
        metric_communities = list(nx.community.greedy_modularity_communities(
            self.metric_graph
        ))
        
        # 3. Calculate metric importance using eigenvector centrality
        metric_importance = nx.eigenvector_centrality(
            self.metric_graph, 
            weight='weight'
        )
        
        # 4. Compute weighted consensus
        final_weights = defaultdict(float)
        total_weight = 0.0
        
        for agent, analysis in self.analyses.items():
            # Calculate agent's effective influence
            influence = agent_influence[agent]
            
            # Weight by metric importance
            metric_weight = sum(
                metric_importance.get(metric, 0)
                for metric in analysis.metrics_used
            ) / len(analysis.metrics_used)
            
            effective_weight = influence * analysis.confidence * metric_weight
            
            # Add weighted contribution
            final_weights["repo_a"] += analysis.weight_a * effective_weight
            final_weights["repo_b"] += analysis.weight_b * effective_weight
            total_weight += effective_weight
        
        # Normalize weights
        if total_weight > 0:
            final_weights = {
                k: v/total_weight 
                for k, v in final_weights.items()
            }
            
        # 5. Generate insight clusters
        
        return ConsensusResult(
            weight_a=final_weights["repo_a"],
            weight_b=final_weights["repo_b"],
            confidence=self._calculate_consensus_confidence(),
            agent_influence=agent_influence,
            metric_importance=metric_importance
        )
    
    def _calculate_consensus_confidence(self) -> float:
        """Calculate overall confidence in consensus"""
        if not self.analyses:
            return 0.0
        
        confidence_scores = [analysis.confidence for analysis in self.analyses.values()]
        
        if not confidence_scores:
            return max(confidence_scores)
        
        return sum(confidence_scores) / len(confidence_scores)