import duckdb
import os
from typing import List, Optional, Dict, Any
from .schemas import KernelCandidate

class Storage:
    """DuckDB wrapper for storing kernel candidates and their scores."""
    
    def __init__(self, db_path: str = "search.db"):
        """Initialize storage with database path.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize the database schema."""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS kernels (
          id INTEGER PRIMARY KEY,
          problem TEXT,
          round INTEGER,
          code TEXT,
          idea TEXT,
          correct BOOLEAN,
          latency_ms DOUBLE
        )
        """)
    
    def save_candidate(self, candidate: KernelCandidate) -> int:
        """Save a kernel candidate to the database.
        
        Args:
            candidate: The candidate to save
            
        Returns:
            The ID of the saved candidate
        """
        # Get the next available ID
        result = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM kernels")
        next_id = result.fetchone()[0]
        
        # Insert the candidate with the generated ID
        self.conn.execute("""
        INSERT INTO kernels (id, problem, round, code, idea, correct, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id,
            candidate.problem, 
            candidate.round, 
            candidate.code, 
            candidate.idea, 
            candidate.correct, 
            candidate.latency_ms
        ])
        
        # Return the generated ID
        return next_id
    
    def update_candidate(self, candidate: KernelCandidate):
        """Update an existing kernel candidate.
        
        Args:
            candidate: The candidate to update (must have id)
        """
        if candidate.id is None:
            raise ValueError("Cannot update candidate without id")
        
        self.conn.execute("""
        UPDATE kernels
        SET correct = ?, latency_ms = ?
        WHERE id = ?
        """, [candidate.correct, candidate.latency_ms, candidate.id])
    
    def get_candidates(self, 
                       problem: str, 
                       round_num: Optional[int] = None, 
                       correct_only: bool = False, 
                       limit: Optional[int] = None) -> List[KernelCandidate]:
        """Get kernel candidates from the database.
        
        Args:
            problem: Problem name
            round_num: Optional round number to filter by
            correct_only: If True, only return correct candidates
            limit: Optional limit on number of candidates to return
            
        Returns:
            List of kernel candidates
        """
        query = "SELECT id, problem, round, code, idea, correct, latency_ms FROM kernels WHERE problem = ?"
        params = [problem]
        
        if round_num is not None:
            query += " AND round = ?"
            params.append(round_num)
            
        if correct_only:
            query += " AND correct = TRUE"
            
        query += " ORDER BY latency_ms ASC"
        
        if limit is not None:
            query += f" LIMIT {limit}"
            
        result = self.conn.execute(query, params)
        rows = result.fetchall()
        
        candidates = []
        for row in rows:
            candidates.append(KernelCandidate(
                id=row[0],
                problem=row[1],
                round=row[2],
                code=row[3],
                idea=row[4],
                correct=row[5],
                latency_ms=row[6]
            ))
            
        return candidates
    
    def close(self):
        """Close the database connection."""
        self.conn.close()