from typing import List, Optional
from .schemas import KernelCandidate, Score

class Scorer:
    """Scores and ranks kernel candidates based on performance metrics."""
    
    def __init__(self):
        """Initialize the scorer."""
        pass
    
    def score_candidates(self, candidates: List[KernelCandidate]) -> List[Score]:
        """Score a list of kernel candidates.
        
        Args:
            candidates: List of kernel candidates to score
            
        Returns:
            List of scores for each candidate
        """
        scores = []
        
        for candidate in candidates:
            score = Score(
                candidate_id=candidate.id if candidate.id is not None else -1,
                correct=candidate.correct,
                latency_ms=candidate.latency_ms if candidate.latency_ms is not None else float('inf')
            )
            scores.append(score)
            
        return scores
    
    def rank_candidates(self, candidates: List[KernelCandidate], top_k: Optional[int] = None) -> List[KernelCandidate]:
        """Rank candidates by performance (lowest latency among correct ones).
        
        Args:
            candidates: List of kernel candidates to rank
            top_k: Optional limit on number of candidates to return
            
        Returns:
            List of candidates, ranked by performance
        """
        # Filter out incorrect candidates
        correct_candidates = [c for c in candidates if c.correct]
        
        # Sort by latency (lowest first)
        ranked_candidates = sorted(correct_candidates, key=lambda c: c.latency_ms if c.latency_ms is not None else float('inf'))
        
        # Apply top_k limit if specified
        if top_k is not None:
            ranked_candidates = ranked_candidates[:top_k]
            
        return ranked_candidates
    
    def select_best_candidate(self, candidates: List[KernelCandidate]) -> Optional[KernelCandidate]:
        """Select the best candidate based on performance.
        
        Args:
            candidates: List of kernel candidates to choose from
            
        Returns:
            The best candidate, or None if no valid candidates
        """
        ranked_candidates = self.rank_candidates(candidates)
        
        if not ranked_candidates:
            return None
            
        return ranked_candidates[0]