"""
Integration tests for BatchProcessor with real SemanticChunker.

Tests the BatchProcessor with actual chunker instances to verify
end-to-end functionality.
"""

import pytest
from chunker import SemanticChunker
from chunker.config import ChunkerConfig, StrategyType
from chunker.batch_processor import BatchProcessor


@pytest.fixture
def chunker():
    """Create a real SemanticChunker instance for testing."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,  # Use simple strategy to avoid embedding dependencies
        chunk_size=100,
        chunk_overlap=20,
    )
    return SemanticChunker(config)


class TestBatchProcessorIntegration:
    """Integration tests with real SemanticChunker."""
    
    def test_process_batch_with_real_chunker(self, chunker):
        """Test batch processing with real chunker."""
        processor = BatchProcessor(chunker)
        texts = [
            "This is the first document. It has multiple sentences. Each sentence is important.",
            "This is the second document. It also has multiple sentences. Testing is crucial.",
            "This is the third document. More sentences here. Quality matters.",
        ]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 3
        # Each document should produce at least one chunk
        for result in results:
            assert len(result) > 0
            for chunk in result:
                assert chunk.text
                assert chunk.metadata.token_count > 0

    def test_process_with_source_files_integration(self, chunker):
        """Test batch processing with source files."""
        processor = BatchProcessor(chunker)
        texts = ["Document one content.", "Document two content."]
        source_files = ["doc1.txt", "doc2.txt"]
        
        results = processor.process_batch(texts, source_files=source_files)
        
        assert len(results) == 2
        # Verify source files are set correctly
        assert results[0][0].metadata.source_file == "doc1.txt"
        assert results[1][0].metadata.source_file == "doc2.txt"
    
    def test_progress_callback_integration(self, chunker):
        """Test progress callback with real chunker."""
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        processor = BatchProcessor(chunker, progress_callback=progress_callback)
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 3
        assert len(progress_calls) == 3
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]
    
    def test_parallel_processing_integration(self, chunker):
        """Test parallel processing with real chunker."""
        processor = BatchProcessor(chunker)
        texts = [
            "First document with some content.",
            "Second document with different content.",
            "Third document with more content.",
            "Fourth document with additional content.",
        ]
        
        results = processor.process_batch(texts, parallel=True, num_workers=2)
        
        assert len(results) == 4
        # All documents should be processed
        for result in results:
            assert len(result) > 0
    
    def test_process_with_progress_iterator_integration(self, chunker):
        """Test iterator-based processing with real chunker."""
        processor = BatchProcessor(chunker)
        texts = ["Doc 1 content.", "Doc 2 content.", "Doc 3 content."]
        
        results = []
        for index, chunks in processor.process_with_progress(texts):
            results.append((index, chunks))
            assert len(chunks) > 0
        
        assert len(results) == 3
        assert results[0][0] == 0
        assert results[1][0] == 1
        assert results[2][0] == 2

    def test_empty_documents_handling(self, chunker):
        """Test handling of empty documents in batch."""
        processor = BatchProcessor(chunker)
        texts = ["Valid document.", "", "Another valid document."]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 3
        # First and third should have chunks
        assert len(results[0]) > 0
        assert len(results[2]) > 0
        # Second (empty) should return empty list
        assert len(results[1]) == 0
    
    def test_large_batch_processing(self, chunker):
        """Test processing a larger batch of documents."""
        processor = BatchProcessor(chunker)
        # Create 50 documents
        texts = [f"Document {i} with some content for testing." for i in range(50)]
        
        results = processor.process_batch(texts)
        
        assert len(results) == 50
        for result in results:
            assert len(result) > 0
    
    def test_batch_processor_preserves_order(self, chunker):
        """Test that batch processing preserves document order."""
        processor = BatchProcessor(chunker)
        texts = [
            "First document content.",
            "Second document content.",
            "Third document content.",
        ]
        
        results = processor.process_batch(texts)
        
        # Verify order is preserved by checking content
        assert "First" in results[0][0].text
        assert "Second" in results[1][0].text
        assert "Third" in results[2][0].text
    
    def test_parallel_preserves_order(self, chunker):
        """Test that parallel processing preserves document order."""
        processor = BatchProcessor(chunker)
        texts = [f"Document number {i} content." for i in range(10)]
        
        results = processor.process_batch(texts, parallel=True, num_workers=4)
        
        # Verify order is preserved
        for i, result in enumerate(results):
            assert f"number {i}" in result[0].text
