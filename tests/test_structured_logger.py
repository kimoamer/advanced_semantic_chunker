"""
Tests for StructuredLogger class.

Validates Requirements 3.1, 3.2, 3.3 for structured logging functionality.
"""

import json
import logging
from io import StringIO
from unittest import TestCase

from chunker.structured_logger import StructuredLogger
from chunker.config import StrategyType
from chunker.models import Language


class TestStructuredLogger(TestCase):
    """Test StructuredLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a string buffer to capture log output
        self.log_buffer = StringIO()
        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setLevel(logging.DEBUG)
        
        # Create logger with our handler
        self.logger = StructuredLogger("test_logger", level=logging.DEBUG, structured=True)
        self.logger.logger.handlers.clear()  # Remove default handlers
        self.logger.logger.addHandler(self.handler)
    
    def tearDown(self):
        """Clean up after tests."""
        self.handler.close()
    
    def _get_log_events(self):
        """Parse JSON log events from buffer."""
        log_output = self.log_buffer.getvalue()
        log_lines = [line for line in log_output.strip().split('\n') if line]
        events = []
        for line in log_lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events
    
    def test_log_chunking_start(self):
        """Test logging chunking start event."""
        self.logger.log_chunking_start(
            doc_id="doc123",
            text_length=1000,
            strategy=StrategyType.SEMANTIC
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'chunking_start')
        self.assertEqual(event['component'], 'SemanticChunker')
        self.assertEqual(event['doc_id'], 'doc123')
        self.assertEqual(event['text_length'], 1000)
        self.assertEqual(event['strategy'], 'semantic')
        self.assertIn('timestamp', event)
    
    def test_log_chunking_complete(self):
        """Test logging chunking complete event."""
        self.logger.log_chunking_complete(
            doc_id="doc123",
            chunk_count=5,
            duration_ms=234.5,
            strategy=StrategyType.SENTENCE
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'chunking_complete')
        self.assertEqual(event['component'], 'SemanticChunker')
        self.assertEqual(event['doc_id'], 'doc123')
        self.assertEqual(event['chunk_count'], 5)
        self.assertEqual(event['duration_ms'], 234.5)
        self.assertEqual(event['strategy'], 'sentence')
    
    def test_log_language_detected(self):
        """Test logging language detection event."""
        self.logger.log_language_detected(
            doc_id="doc123",
            language=Language.ENGLISH,
            confidence=0.95
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'language_detected')
        self.assertEqual(event['component'], 'LanguageDetector')
        self.assertEqual(event['doc_id'], 'doc123')
        self.assertEqual(event['language'], 'en')
        self.assertEqual(event['confidence'], 0.95)
    
    def test_log_cache_hit(self):
        """Test logging cache hit event."""
        self.logger.log_cache_hit(
            cache_type="embedding",
            key_hash="abc123"
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'cache_hit')
        self.assertEqual(event['component'], 'CacheManager')
        self.assertEqual(event['cache_type'], 'embedding')
        self.assertEqual(event['key_hash'], 'abc123')
    
    def test_log_fallback(self):
        """Test logging fallback activation."""
        self.logger.log_fallback(
            component="SemanticChunker",
            reason="embedding provider failure",
            fallback_action="using recursive strategy"
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'fallback_activated')
        self.assertEqual(event['component'], 'SemanticChunker')
        self.assertEqual(event['reason'], 'embedding provider failure')
        self.assertEqual(event['fallback_action'], 'using recursive strategy')
    
    def test_log_error(self):
        """Test logging error event."""
        test_error = ValueError("Test error")
        
        self.logger.log_error(
            component="TestComponent",
            error=test_error,
            recoverable=True,
            extra_context="additional info"
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'error_occurred')
        self.assertEqual(event['component'], 'TestComponent')
        self.assertEqual(event['error_type'], 'ValueError')
        self.assertEqual(event['error_message'], 'Test error')
        self.assertEqual(event['recoverable'], True)
        self.assertEqual(event['extra_context'], 'additional info')
    
    def test_log_timing(self):
        """Test logging timing information."""
        self.logger.log_timing(
            operation="strategy_chunking",
            duration_ms=123.45,
            strategy="semantic",
            sentence_count=10
        )
        
        events = self._get_log_events()
        self.assertEqual(len(events), 1)
        
        event = events[0]
        self.assertEqual(event['event'], 'operation_timing')
        self.assertEqual(event['operation'], 'strategy_chunking')
        self.assertEqual(event['duration_ms'], 123.45)
        self.assertEqual(event['strategy'], 'semantic')
        self.assertEqual(event['sentence_count'], 10)
    
    def test_non_structured_logging(self):
        """Test non-structured (text) logging mode."""
        logger = StructuredLogger("test_text", level=logging.INFO, structured=False)
        logger.logger.handlers.clear()
        logger.logger.addHandler(self.handler)
        
        logger.log_chunking_start(
            doc_id="doc123",
            text_length=1000,
            strategy=StrategyType.SEMANTIC
        )
        
        log_output = self.log_buffer.getvalue()
        
        # Should be text format, not JSON
        self.assertIn('[chunking_start]', log_output)
        self.assertIn('doc_id=doc123', log_output)
        self.assertIn('text_length=1000', log_output)
    
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        # Create logger with INFO level
        logger = StructuredLogger("test_level", level=logging.INFO, structured=True)
        logger.logger.handlers.clear()
        logger.logger.addHandler(self.handler)
        
        # Log DEBUG event (should be filtered out)
        logger.log_cache_hit(cache_type="test", key_hash="abc")
        
        # Log INFO event (should be logged)
        logger.log_chunking_start(
            doc_id="doc123",
            text_length=1000,
            strategy=StrategyType.SEMANTIC
        )
        
        events = self._get_log_events()
        
        # Only INFO event should be logged (cache_hit is DEBUG level)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'chunking_start')


if __name__ == '__main__':
    import unittest
    unittest.main()
