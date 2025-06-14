"""State management for the index graph - Local documents only."""

from dataclasses import dataclass, field
from typing import List


@dataclass(kw_only=True)
class InputState:
    """The input state for the index graph."""
    
    pdf_root: str = ""
    """The root directory path containing PDF files to index."""


@dataclass(kw_only=True)
class IndexState(InputState):
    """Represents the state for local document indexing.

    This class defines the structure for indexing PDFs stored locally,
    with tracking of processing status and statistics.
    """

    status: str = field(default="")
    """Status of the indexing operation."""
    
    processed_files: List[str] = field(default_factory=list)
    """List of successfully processed PDF files."""
    
    failed_files: List[str] = field(default_factory=list)
    """List of PDF files that failed to process."""
    
    total_chunks: int = field(default=0)
    """Total number of text chunks successfully indexed."""
    
    total_files_found: int = field(default=0)
    """Total number of PDF files found in the directory."""