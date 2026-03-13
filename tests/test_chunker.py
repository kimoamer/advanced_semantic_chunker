"""
Tests for the Semantic Chunker — core functionality.

These tests verify all chunking strategies, language detection,
normalization, and metadata generation without requiring
embedding models or API keys (mocked where needed).
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from chunker.config import ChunkerConfig, StrategyType, ThresholdType
from chunker.language.detector import LanguageDetector
from chunker.language.english import EnglishProcessor
from chunker.language.arabic import ArabicProcessor
from chunker.language.normalizer import TextNormalizer
from chunker.models import Chunk, ChunkMetadata, Language
from chunker.strategies.base import BaseStrategy
from chunker.strategies.sentence import SentenceStrategy
from chunker.strategies.recursive import RecursiveStrategy
from chunker.strategies.fixed import FixedSizeStrategy
from chunker.strategies.hierarchical import HierarchicalStrategy
from chunker.strategies.structure_aware import StructureAwareStrategy
from chunker.document_tree import parse_document_tree, build_sections, NodeType
from chunker.utils import estimate_tokens, is_structural_element, clean_whitespace


# ═══════════════════════════════════════════════════════════════
# Language Detection Tests
# ═══════════════════════════════════════════════════════════════


class TestLanguageDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LanguageDetector()

    def test_detect_english(self):
        text = "This is a simple English sentence for testing purposes."
        self.assertEqual(self.detector.detect(text), Language.ENGLISH)

    def test_detect_arabic(self):
        text = "هذا نص باللغة العربية لاختبار كاشف اللغة"
        self.assertEqual(self.detector.detect(text), Language.ARABIC)

    def test_detect_mixed(self):
        text = "This is English هذا عربي mixed together"
        result = self.detector.detect(text)
        self.assertIn(result, [Language.MIXED, Language.ENGLISH, Language.ARABIC])

    def test_detect_empty(self):
        self.assertEqual(self.detector.detect(""), Language.UNKNOWN)
        self.assertEqual(self.detector.detect("   "), Language.UNKNOWN)

    def test_detect_numbers_only(self):
        self.assertEqual(self.detector.detect("12345 67890"), Language.UNKNOWN)

    def test_find_language_boundaries(self):
        sentences = [
            "Hello world.",
            "This is English.",
            "هذا نص عربي",
            "المزيد من العربية",
            "Back to English now.",
        ]
        boundaries = self.detector.find_language_boundaries(sentences)
        self.assertGreaterEqual(len(boundaries), 2)

    def test_detect_batch_empty_list(self):
        """Test batch detection with empty list."""
        result = self.detector.detect_batch([])
        self.assertEqual(result, [])

    def test_detect_batch_single_text(self):
        """Test batch detection with single text."""
        texts = ["This is English text."]
        result = self.detector.detect_batch(texts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], Language.ENGLISH)

    def test_detect_batch_multiple_texts(self):
        """Test batch detection with multiple texts."""
        texts = [
            "This is English text.",
            "هذا نص باللغة العربية",
            "Another English sentence.",
            "المزيد من النص العربي",
        ]
        result = self.detector.detect_batch(texts)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], Language.ENGLISH)
        self.assertEqual(result[1], Language.ARABIC)
        self.assertEqual(result[2], Language.ENGLISH)
        self.assertEqual(result[3], Language.ARABIC)

    def test_detect_batch_consistency(self):
        """Test that batch detection gives same results as individual detection."""
        texts = [
            "This is English text.",
            "هذا نص باللغة العربية",
            "12345",
            "",
        ]
        batch_result = self.detector.detect_batch(texts)
        individual_results = [self.detector.detect(text) for text in texts]
        self.assertEqual(batch_result, individual_results)


# ═══════════════════════════════════════════════════════════════
# English Processor Tests
# ═══════════════════════════════════════════════════════════════


class TestEnglishProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = EnglishProcessor(use_nltk=False, use_spacy=False)

    def test_segment_simple(self):
        text = "Hello world. This is a test. Another sentence here."
        sentences = self.processor.segment_sentences(text)
        self.assertGreaterEqual(len(sentences), 2)

    def test_segment_empty(self):
        self.assertEqual(self.processor.segment_sentences(""), [])
        self.assertEqual(self.processor.segment_sentences("   "), [])

    def test_detect_structure_headers(self):
        text = "# Header\nSome content here."
        structure = self.processor.detect_structure(text)
        self.assertTrue(structure["has_headers"])

    def test_detect_structure_code(self):
        text = "Some text\n```python\nprint('hello')\n```\nMore text"
        structure = self.processor.detect_structure(text)
        self.assertTrue(structure["has_code"])

    def test_detect_structure_tables(self):
        text = "| Col1 | Col2 |\n| --- | --- |\n| a | b |"
        structure = self.processor.detect_structure(text)
        self.assertTrue(structure["has_tables"])

    def test_extract_sections(self):
        text = "# Section 1\nContent one.\n# Section 2\nContent two."
        sections = self.processor.extract_sections(text)
        self.assertGreaterEqual(len(sections), 2)

    def test_estimate_tokens(self):
        text = "This is approximately twenty characters."
        tokens = EnglishProcessor.estimate_tokens(text)
        self.assertGreater(tokens, 0)


# ═══════════════════════════════════════════════════════════════
# Arabic Processor Tests
# ═══════════════════════════════════════════════════════════════


class TestArabicProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ArabicProcessor(use_camel=False, use_stanza=False)

    def test_normalize_tashkeel_removal(self):
        # Text with diacritics
        text = "كِتَابٌ"
        normalized = self.processor.normalize(text)
        # Tashkeel should be removed
        self.assertNotIn("\u0650", normalized)  # kasra
        self.assertNotIn("\u064E", normalized)  # fatha

    def test_normalize_hamza(self):
        # أحمد should become احمد
        text = "أحمد"
        normalized = self.processor.normalize(text)
        self.assertTrue(normalized.startswith("ا"))

    def test_normalize_tatweel(self):
        # تـــعـــلم → تعلم
        text = "تـــعـــلم"
        normalized = self.processor.normalize(text)
        self.assertNotIn("\u0640", normalized)

    def test_segment_sentences(self):
        text = "هذه جملة أولى. هذه جملة ثانية. وهذه جملة ثالثة."
        sentences = self.processor.segment_sentences(text)
        self.assertGreaterEqual(len(sentences), 2)

    def test_segment_arabic_question_mark(self):
        text = "ما هذا؟ هذا اختبار."
        sentences = self.processor.segment_sentences(text)
        self.assertGreaterEqual(len(sentences), 1)

    def test_segment_empty(self):
        self.assertEqual(self.processor.segment_sentences(""), [])

    def test_is_arabic(self):
        self.assertTrue(self.processor.is_arabic("هذا نص عربي"))
        self.assertFalse(self.processor.is_arabic("This is English"))


# ═══════════════════════════════════════════════════════════════
# Text Normalizer Tests
# ═══════════════════════════════════════════════════════════════


class TestTextNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = TextNormalizer()

    def test_normalize_whitespace(self):
        text = "Hello    world   test"
        result = self.normalizer.normalize(text)
        self.assertEqual(result, "Hello world test")

    def test_normalize_control_chars(self):
        text = "Hello\x00\x01world"
        result = self.normalizer.normalize(text)
        self.assertNotIn("\x00", result)

    def test_normalize_newlines(self):
        text = "Hello\n\n\n\n\nworld"
        result = self.normalizer.normalize(text)
        self.assertNotIn("\n\n\n", result)

    def test_normalize_empty(self):
        self.assertEqual(self.normalizer.normalize(""), "")
        self.assertIsNone(self.normalizer.normalize(None))


# ═══════════════════════════════════════════════════════════════
# Strategy Tests
# ═══════════════════════════════════════════════════════════════


class TestSentenceStrategy(unittest.TestCase):
    def setUp(self):
        self.config = ChunkerConfig(
            strategy=StrategyType.SENTENCE,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=10,
            max_chunk_size=500,
        )
        self.strategy = SentenceStrategy(self.config)

    def test_basic_chunking(self):
        sentences = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
            "This is the fourth sentence.",
            "This is the fifth sentence.",
        ]
        chunks = self.strategy.chunk(sentences)
        self.assertGreater(len(chunks), 0)
        # All text should be preserved
        all_text = " ".join(chunks)
        for sent in sentences:
            self.assertIn(sent, all_text)

    def test_empty_input(self):
        self.assertEqual(self.strategy.chunk([]), [])

    def test_single_sentence(self):
        sentences = ["Just one sentence."]
        chunks = self.strategy.chunk(sentences)
        self.assertEqual(len(chunks), 1)


class TestRecursiveStrategy(unittest.TestCase):
    def setUp(self):
        self.config = ChunkerConfig(
            strategy=StrategyType.RECURSIVE,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=5,
            max_chunk_size=500,
        )
        self.strategy = RecursiveStrategy(self.config)

    def test_splits_on_paragraphs(self):
        text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
        chunks = self.strategy.chunk([], text)
        self.assertGreater(len(chunks), 0)

    def test_empty_input(self):
        self.assertEqual(self.strategy.chunk([], ""), [])


class TestFixedSizeStrategy(unittest.TestCase):
    def setUp(self):
        self.config = ChunkerConfig(
            strategy=StrategyType.FIXED,
            chunk_size=100,
            chunk_overlap=5,
            min_chunk_size=10,
            max_chunk_size=500,
        )
        self.strategy = FixedSizeStrategy(self.config)

    def test_creates_chunks(self):
        text = "A " * 200
        chunks = self.strategy.chunk([], text)
        self.assertGreater(len(chunks), 1)

    def test_empty_input(self):
        self.assertEqual(self.strategy.chunk([], ""), [])


class TestHierarchicalStrategy(unittest.TestCase):
    def setUp(self):
        self.config = ChunkerConfig(
            strategy=StrategyType.HIERARCHICAL,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=10,
        )
        self.strategy = HierarchicalStrategy(self.config)

    def test_sections_detected(self):
        text = "# Section 1\nContent of section one.\n\n# Section 2\nContent of section two."
        chunks = self.strategy.chunk([], text)
        self.assertGreater(len(chunks), 0)

    def test_empty_input(self):
        self.assertEqual(self.strategy.chunk([], ""), [])


# ═══════════════════════════════════════════════════════════════
# Document Tree Tests
# ═══════════════════════════════════════════════════════════════


class TestDocumentTree(unittest.TestCase):
    def test_parse_headings(self):
        text = "# Title\nContent\n## Sub\nMore content"
        nodes = parse_document_tree(text)
        headings = [n for n in nodes if n.node_type == NodeType.HEADING]
        self.assertEqual(len(headings), 2)
        self.assertEqual(headings[0].heading_level, 1)
        self.assertEqual(headings[0].heading_title, "Title")
        self.assertEqual(headings[1].heading_level, 2)

    def test_parse_code_block(self):
        text = "Some text\n```python\nprint('hello')\n```\nMore text"
        nodes = parse_document_tree(text)
        code_nodes = [n for n in nodes if n.node_type == NodeType.CODE]
        self.assertEqual(len(code_nodes), 1)

    def test_parse_table(self):
        text = "| Col1 | Col2 |\n| --- | --- |\n| a | b |"
        nodes = parse_document_tree(text)
        table_nodes = [n for n in nodes if n.node_type == NodeType.TABLE]
        self.assertEqual(len(table_nodes), 1)

    def test_parse_list(self):
        text = "- Item 1\n- Item 2\n- Item 3"
        nodes = parse_document_tree(text)
        list_nodes = [n for n in nodes if n.node_type == NodeType.LIST]
        self.assertEqual(len(list_nodes), 1)
        self.assertEqual(len(list_nodes[0].list_items), 3)

    def test_build_sections(self):
        text = "# Top\nContent 1\n## Sub A\nContent A\n## Sub B\nContent B"
        nodes = parse_document_tree(text)
        sections = build_sections(nodes)
        self.assertGreaterEqual(len(sections), 3)

    def test_heading_path_breadcrumb(self):
        text = "# Top\nContent\n## Sub\nSub content\n### Detail\nDetail content"
        nodes = parse_document_tree(text)
        sections = build_sections(nodes)
        # Find the ### Detail section
        detail = [s for s in sections if s.heading_title == "Detail"]
        self.assertEqual(len(detail), 1)
        self.assertEqual(detail[0].heading_path, ["Top", "Sub", "Detail"])

    def test_heading_path_sibling_reset(self):
        text = "# Top\n## Sub A\nContent A\n## Sub B\nContent B"
        nodes = parse_document_tree(text)
        sections = build_sections(nodes)
        sub_b = [s for s in sections if s.heading_title == "Sub B"]
        self.assertEqual(len(sub_b), 1)
        self.assertEqual(sub_b[0].heading_path, ["Top", "Sub B"])

    def test_empty_input(self):
        self.assertEqual(parse_document_tree(""), [])
        self.assertEqual(build_sections([]), [])


# ═══════════════════════════════════════════════════════════════
# Structure-Aware Strategy Tests
# ═══════════════════════════════════════════════════════════════


class TestStructureAwareStrategy(unittest.TestCase):
    def setUp(self):
        self.config = ChunkerConfig(
            strategy=StrategyType.STRUCTURE_AWARE,
            chunk_size=220,
            chunk_overlap=0,
            min_chunk_size=50,
            max_chunk_size=400,
        )
        self.strategy = StructureAwareStrategy(self.config)

    def test_heading_hard_boundary(self):
        """Headings should never appear at the end of a chunk."""
        text = (
            "# Section One\n"
            "First section has some content here that we want to read.\n\n"
            "# Section Two\n"
            "Second section has different content about another topic."
        )
        chunks = self.strategy.chunk([], text)
        self.assertGreater(len(chunks), 0)
        # No chunk should end with a heading
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            last_line = lines[-1].strip()
            self.assertFalse(
                last_line.startswith("#"),
                f"Heading should not be at end of chunk: {last_line}"
            )

    def test_section_metadata_populated(self):
        """section_metadata should be populated with heading paths."""
        text = (
            "# Title\n"
            "Intro content paragraph.\n\n"
            "## Subsection\n"
            "Subsection content here."
        )
        chunks = self.strategy.chunk([], text)
        meta = self.strategy.section_metadata
        self.assertEqual(len(meta), len(chunks))
        # At least one should have heading_path
        paths = [m.get("heading_path", []) for m in meta]
        self.assertTrue(any(len(p) > 0 for p in paths))

    def test_small_tail_merged(self):
        """Small tail fragments below min_chunk_size should be merged."""
        config = ChunkerConfig(
            strategy=StrategyType.STRUCTURE_AWARE,
            chunk_size=300,
            chunk_overlap=0,
            min_chunk_size=80,
            max_chunk_size=600,
        )
        strategy = StructureAwareStrategy(config)
        text = (
            "# Main Section\n"
            "This is a long paragraph with many sentences. "
            "We want to make sure tail fragments are merged. "
            "Short."
        )
        chunks = strategy.chunk([], text)
        # Should merge the short tail into previous
        for chunk in chunks:
            # The very short "Short." should not be a standalone chunk
            self.assertNotEqual(chunk.strip(), "Short.")

    def test_tables_kept_intact(self):
        """Tables should not be split across chunks."""
        text = (
            "# Data\n"
            "| Name | Value |\n"
            "| --- | --- |\n"
            "| Alice | 100 |\n"
            "| Bob | 200 |"
        )
        chunks = self.strategy.chunk([], text)
        # Table should be in a single chunk
        table_chunks = [c for c in chunks if "|" in c]
        self.assertGreater(len(table_chunks), 0)
        # All table rows should be in the same chunk
        for tc in table_chunks:
            if "Alice" in tc:
                self.assertIn("Bob", tc)

    def test_code_blocks_kept_intact(self):
        """Code blocks should not be split across chunks."""
        text = (
            "# Examples\n"
            "```python\n"
            "def hello():\n"
            "    print('hello')\n"
            "```"
        )
        chunks = self.strategy.chunk([], text)
        code_chunks = [c for c in chunks if "def hello" in c]
        self.assertEqual(len(code_chunks), 1)
        self.assertIn("print", code_chunks[0])

    def test_arabic_document(self):
        """Structure-aware should work with Arabic documents."""
        text = (
            "# مقدمة\n"
            "هذا هو المحتوى الأول في القسم. نريد التأكد من أنه يعمل بشكل صحيح.\n\n"
            "## القسم الفرعي\n"
            "محتوى القسم الفرعي هنا."
        )
        chunks = self.strategy.chunk([], text)
        self.assertGreater(len(chunks), 0)

    def test_no_overlap_across_headings(self):
        """Overlap should never cross heading boundaries."""
        config = ChunkerConfig(
            strategy=StrategyType.STRUCTURE_AWARE,
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,
            max_chunk_size=500,
        )
        strategy = StructureAwareStrategy(config)
        text = (
            "# Section One\n"
            "Content from section one goes here.\n\n"
            "# Section Two\n"
            "Content from section two is different."
        )
        chunks = strategy.chunk([], text)
        # Content from section one should not leak into section two
        for chunk in chunks:
            if "section two" in chunk.lower():
                self.assertNotIn("section one", chunk.lower())

    def test_empty_input(self):
        self.assertEqual(self.strategy.chunk([], ""), [])
        self.assertEqual(self.strategy.chunk([]), [])


# ═══════════════════════════════════════════════════════════════
# Semantic Strategy Tests (with mocked embeddings)
# ═══════════════════════════════════════════════════════════════


class TestSemanticStrategy(unittest.TestCase):
    def setUp(self):
        self.config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            chunk_size=200,
            chunk_overlap=0,
            min_chunk_size=10,
            threshold_type=ThresholdType.PERCENTILE,
            threshold_amount=50.0,  # Low threshold for testing
        )

    def _create_mock_provider(self, n_sentences: int, break_at: int):
        """
        Create a mock embedding provider that produces embeddings
        with a sharp cosine distance jump at `break_at`.
        """
        mock = MagicMock()
        mock.get_dimension.return_value = 8

        # Generate embeddings: similar vectors except at break point
        embeddings = []
        for i in range(n_sentences):
            if i < break_at:
                vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + np.random.normal(0, 0.05, 8)
            else:
                vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) + np.random.normal(0, 0.05, 8)
            embeddings.append(vec)

        mock.embed.return_value = np.array(embeddings)
        mock.embed_single.side_effect = lambda t: embeddings[0]

        return mock

    def test_detects_topic_shift(self):
        from chunker.strategies.semantic import SemanticStrategy

        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Neural networks learn patterns from large datasets.",
            "The weather today is sunny and warm.",
            "Tomorrow will be cloudy with a chance of rain.",
            "Temperature will drop to around 15 degrees.",
        ]

        mock_provider = self._create_mock_provider(len(sentences), break_at=3)
        strategy = SemanticStrategy(self.config, mock_provider)
        chunks = strategy.chunk(sentences)

        # Should detect the topic shift and create at least 2 chunks
        self.assertGreaterEqual(len(chunks), 2)

    def test_single_sentence(self):
        from chunker.strategies.semantic import SemanticStrategy

        mock_provider = self._create_mock_provider(1, break_at=1)
        strategy = SemanticStrategy(self.config, mock_provider)
        chunks = strategy.chunk(["Single sentence."])

        self.assertEqual(len(chunks), 1)


# ═══════════════════════════════════════════════════════════════
# Model Tests
# ═══════════════════════════════════════════════════════════════


class TestChunkModel(unittest.TestCase):
    def test_chunk_creation(self):
        chunk = Chunk(text="Hello world")
        self.assertEqual(chunk.text, "Hello world")
        self.assertFalse(chunk.is_empty)

    def test_chunk_hash(self):
        chunk = Chunk(text="Hello world")
        self.assertEqual(len(chunk.content_hash), 64)

    def test_chunk_to_dict(self):
        chunk = Chunk(text="Hello world")
        d = chunk.to_dict()
        self.assertIn("text", d)
        self.assertIn("metadata", d)
        self.assertEqual(d["text"], "Hello world")

    def test_empty_chunk(self):
        chunk = Chunk(text="")
        self.assertTrue(chunk.is_empty)

    def test_chunk_repr(self):
        chunk = Chunk(text="Hello world")
        repr_str = repr(chunk)
        self.assertIn("Hello world", repr_str)


# ═══════════════════════════════════════════════════════════════
# Utility Tests
# ═══════════════════════════════════════════════════════════════


class TestUtils(unittest.TestCase):
    def test_estimate_tokens_english(self):
        tokens = estimate_tokens("Hello world test", "en")
        self.assertGreater(tokens, 0)

    def test_estimate_tokens_arabic(self):
        tokens = estimate_tokens("مرحبا بالعالم", "ar")
        self.assertGreater(tokens, 0)

    def test_structural_element_header(self):
        self.assertEqual(is_structural_element("# Header"), "header")
        self.assertEqual(is_structural_element("## Sub"), "header")

    def test_structural_element_table(self):
        self.assertEqual(is_structural_element("| a | b |"), "table_row")

    def test_structural_element_list(self):
        self.assertEqual(is_structural_element("- Item"), "list_item")
        self.assertEqual(is_structural_element("1. Item"), "list_item")

    def test_structural_element_code(self):
        self.assertEqual(is_structural_element("```python"), "code_fence")

    def test_structural_element_none(self):
        self.assertIsNone(is_structural_element("Just regular text"))

    def test_clean_whitespace(self):
        text = "Hello    world\n\n\n\n\ntest"
        result = clean_whitespace(text)
        self.assertNotIn("    ", result)


# ═══════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        config = ChunkerConfig()
        config.validate()  # Should not raise

    def test_invalid_overlap(self):
        config = ChunkerConfig(chunk_size=100, chunk_overlap=200)
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_percentile(self):
        config = ChunkerConfig(
            threshold_type=ThresholdType.PERCENTILE,
            threshold_amount=150.0,
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_agentic_requires_key(self):
        config = ChunkerConfig(strategy=StrategyType.AGENTIC, openai_api_key=None)
        with self.assertRaises(ValueError):
            config.validate()


# ═══════════════════════════════════════════════════════════════
# Integration Test (mocked — no real embeddings)
# ═══════════════════════════════════════════════════════════════


class TestSemanticChunkerIntegration(unittest.TestCase):
    """Integration tests using non-semantic strategies (no model needed)."""

    def test_sentence_strategy_english(self):
        from chunker.core import SemanticChunker

        config = ChunkerConfig(
            strategy=StrategyType.SENTENCE,
            chunk_size=100,
            chunk_overlap=0,
            detect_language=True,
        )
        chunker = SemanticChunker(config)
        text = (
            "Machine learning is transforming how we interact with technology. "
            "Deep learning, a subset of machine learning, uses neural networks. "
            "These networks can learn complex patterns from large datasets. "
            "Applications range from image recognition to natural language processing. "
            "The field continues to grow rapidly with new breakthroughs every year."
        )
        chunks = chunker.chunk(text, source_file="test.txt")

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata.source_file, "test.txt")
        self.assertEqual(chunks[0].metadata.strategy, "sentence")

    def test_sentence_strategy_arabic(self):
        from chunker.core import SemanticChunker

        config = ChunkerConfig(
            strategy=StrategyType.SENTENCE,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=10,
            max_chunk_size=500,
        )
        chunker = SemanticChunker(config)
        text = "التعلم الآلي يغير طريقة تفاعلنا مع التكنولوجيا. التعلم العميق يستخدم الشبكات العصبية. هذه الشبكات يمكنها تعلم أنماط معقدة."
        chunks = chunker.chunk(text)

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata.language, Language.ARABIC)

    def test_recursive_strategy(self):
        from chunker.core import SemanticChunker

        config = ChunkerConfig(
            strategy=StrategyType.RECURSIVE,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=5,
            max_chunk_size=500,
        )
        chunker = SemanticChunker(config)
        text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph here."
        chunks = chunker.chunk(text)

        self.assertGreater(len(chunks), 0)

    def test_get_stats(self):
        from chunker.core import SemanticChunker

        config = ChunkerConfig(strategy=StrategyType.SENTENCE, chunk_size=200)
        chunker = SemanticChunker(config)
        text = "Hello world. This is a test. Another sentence."
        chunks = chunker.chunk(text)
        stats = chunker.get_stats(chunks)

        self.assertIn("total_chunks", stats)
        self.assertIn("total_tokens", stats)
        self.assertIn("language_distribution", stats)

    def test_to_dicts(self):
        from chunker.core import SemanticChunker

        config = ChunkerConfig(strategy=StrategyType.SENTENCE, chunk_size=200)
        chunker = SemanticChunker(config)
        text = "Hello world. This is a test."
        chunks = chunker.chunk(text)
        dicts = chunker.to_dicts(chunks)

        self.assertIsInstance(dicts, list)
        self.assertIn("text", dicts[0])
        self.assertIn("metadata", dicts[0])

    def test_empty_input(self):
        from chunker.core import SemanticChunker

        config = ChunkerConfig(strategy=StrategyType.SENTENCE)
        chunker = SemanticChunker(config)
        chunks = chunker.chunk("")

        self.assertEqual(len(chunks), 0)
    def test_structure_aware_default_strategy(self):
        """Default strategy should be STRUCTURE_AWARE."""
        from chunker.core import SemanticChunker

        config = ChunkerConfig()
        self.assertEqual(config.strategy, StrategyType.STRUCTURE_AWARE)

        chunker = SemanticChunker(config)
        text = (
            "# Introduction\n"
            "Machine learning is transforming technology. "
            "It enables systems to learn from data.\n\n"
            "## Deep Learning\n"
            "Deep learning uses neural networks with many layers."
        )
        chunks = chunker.chunk(text)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].metadata.strategy, "structure_aware")
        # Should have heading_path populated
        self.assertGreater(len(chunks[0].metadata.heading_path), 0)

    def test_heading_path_in_output(self):
        """heading_path should be serialized in to_dict output."""
        from chunker.core import SemanticChunker

        config = ChunkerConfig()
        chunker = SemanticChunker(config)
        text = "# Title\nContent here. More content."
        chunks = chunker.chunk(text)
        d = chunks[0].to_dict()
        self.assertIn("heading_path", d["metadata"])
        self.assertIn("script", d["metadata"])


if __name__ == "__main__":
    unittest.main()
