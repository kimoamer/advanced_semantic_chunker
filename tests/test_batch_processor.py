"""
Unit tests for BatchProcessor class.

Tests batch processing functionality including:
- Sequential processing
- Parallel processing
- Progress callbacks
- Iterator-based processing
"""

import pytest
from unittest.mock import Mock, MagicMock, call
from chunker.batch_processor import BatchProcessor
from chunker.models import Chunk, ChunkMetadata, Language


@pytest.fixture
def mock_chunker():
    """Create a mock SemanticChunker for testing."""
    chunker = Mock()
    
    def mock_chunk(text, source_file=""):
        """Mock chunk method that returns predictable chunks."""
        # Create a simple chunk for each text
        chunk = Chunk(
            text=f"Chunk from: {text[:20]}",
            metadata=ChunkMetadata(
                source_file=source_file,
                language=Language.ENGLISH,
                token_count=len(text.split()),
                char_count=len(text),
            )
        )
        return [chunk]
    
    chunker.chunk = Mock(side_effect=mock_chunk)
    return chunker


class TestBatchProcessorInit:
    """Test BatchProcessor initialization."""
    
    def test_init_with_defaults(self, mock_chunker):
        """Test initialization with default parameters."""
        processor = BatchProcessor(mock_chunker)
        
        assert processor.chunker is mock_chunker
        assert processor.batch_size == 32
        assert processor.progress_callback is None

    def test_init_with_custom_batch_size(self, mock_chunker):
        """Test initialization with custom batch size."""
        processor = BatchProcessor(mock_chunker, batch_size=64)
        
        assert processor.batch_size == 64
    
    def test_init_with_progress_callback(self, mock_chunker):
        """Test initialization with progress callback."""
        callback = Mock()
        processor = BatchProcessor(mock_chunker, progress_callback=callback)
        
        assert processor.progress_callback is callback


class TestProcessBatch:
    """Test process_batch method."""
    
    def test_process_empty_list(self, mock_chunker):
        """Test processing empty list returns empty results."""
        processor = BatchProcessor(mock_chunker)
        results = processor.process_batch([])
        
        assert results == []
        mock_chunker.chunk.assert_not_called()
    
    def test_process_single_document(self, mock_chunker):
        """Test processing a single document."""
        processor = BatchProcessor(mock_chunker)
        texts = ["This is a test document."]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 1
        assert len(results[0]) == 1
        assert "Chunk from:" in results[0][0].text
        mock_chunker.chunk.assert_called_once_with("This is a test document.", source_file="")
    
    def test_process_multiple_documents(self, mock_chunker):
        """Test processing multiple documents."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Document 1", "Document 2", "Document 3"]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 3
        assert mock_chunker.chunk.call_count == 3

    def test_process_with_source_files(self, mock_chunker):
        """Test processing with source file names."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Doc 1", "Doc 2"]
        source_files = ["file1.txt", "file2.txt"]
        
        results = processor.process_batch(texts, source_files=source_files)
        
        assert len(results) == 2
        calls = mock_chunker.chunk.call_args_list
        assert calls[0] == call("Doc 1", source_file="file1.txt")
        assert calls[1] == call("Doc 2", source_file="file2.txt")
    
    def test_process_with_fewer_source_files(self, mock_chunker):
        """Test processing when source_files list is shorter than texts."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        source_files = ["file1.txt"]
        
        results = processor.process_batch(texts, source_files=source_files)
        
        assert len(results) == 3
        calls = mock_chunker.chunk.call_args_list
        assert calls[0] == call("Doc 1", source_file="file1.txt")
        assert calls[1] == call("Doc 2", source_file="")
        assert calls[2] == call("Doc 3", source_file="")
    
    def test_process_with_progress_callback(self, mock_chunker):
        """Test that progress callback is invoked correctly."""
        callback = Mock()
        processor = BatchProcessor(mock_chunker, progress_callback=callback)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 3
        # Callback should be called after each document
        assert callback.call_count == 3
        callback.assert_any_call(1, 3)
        callback.assert_any_call(2, 3)
        callback.assert_any_call(3, 3)

    def test_process_parallel(self, mock_chunker):
        """Test parallel processing."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        
        results = processor.process_batch(texts, parallel=True, num_workers=2)
        
        assert len(results) == 4
        # All documents should be processed
        assert mock_chunker.chunk.call_count == 4
        # Results should be in correct order
        for i, result in enumerate(results):
            assert len(result) == 1
    
    def test_process_parallel_with_progress_callback(self, mock_chunker):
        """Test parallel processing with progress callback."""
        callback = Mock()
        processor = BatchProcessor(mock_chunker, progress_callback=callback)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        
        results = processor.process_batch(texts, parallel=True)
        
        assert len(results) == 3
        # Callback should be called 3 times (once per document)
        assert callback.call_count == 3
        # Check that final call has correct total
        final_call = [c for c in callback.call_args_list if c[0][0] == 3]
        assert len(final_call) == 1
        assert final_call[0] == call(3, 3)


class TestProcessWithProgress:
    """Test process_with_progress method."""
    
    def test_process_with_progress_iterator(self, mock_chunker):
        """Test iterator-based processing."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        
        results = list(processor.process_with_progress(texts))
        
        assert len(results) == 3
        # Check that results are tuples of (index, chunks)
        for i, (index, chunks) in enumerate(results):
            assert index == i
            assert len(chunks) == 1

    def test_process_with_progress_streaming(self, mock_chunker):
        """Test that iterator allows streaming processing."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        
        # Process one at a time
        iterator = processor.process_with_progress(texts)
        
        # First document
        index, chunks = next(iterator)
        assert index == 0
        assert len(chunks) == 1
        assert mock_chunker.chunk.call_count == 1
        
        # Second document
        index, chunks = next(iterator)
        assert index == 1
        assert len(chunks) == 1
        assert mock_chunker.chunk.call_count == 2
        
        # Third document
        index, chunks = next(iterator)
        assert index == 2
        assert len(chunks) == 1
        assert mock_chunker.chunk.call_count == 3
    
    def test_process_with_progress_with_source_files(self, mock_chunker):
        """Test iterator processing with source files."""
        processor = BatchProcessor(mock_chunker)
        texts = ["Doc 1", "Doc 2"]
        source_files = ["file1.txt", "file2.txt"]
        
        results = list(processor.process_with_progress(texts, source_files=source_files))
        
        assert len(results) == 2
        calls = mock_chunker.chunk.call_args_list
        assert calls[0] == call("Doc 1", source_file="file1.txt")
        assert calls[1] == call("Doc 2", source_file="file2.txt")
    
    def test_process_with_progress_empty_list(self, mock_chunker):
        """Test iterator with empty list."""
        processor = BatchProcessor(mock_chunker)
        
        results = list(processor.process_with_progress([]))
        
        assert results == []
        mock_chunker.chunk.assert_not_called()
