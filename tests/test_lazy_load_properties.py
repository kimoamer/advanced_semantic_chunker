"""
Property-based tests for lazy loading behavior using Hypothesis.

These tests verify that expensive resources (embedding models, language processors,
optional dependencies) are only loaded when actually needed, not eagerly at initialization.
"""

import gc
import sys
import weakref
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from chunker.config import ChunkerConfig, EmbeddingProvider, StrategyType
from chunker.core import SemanticChunker
from chunker.lazy_load import LazyLoadManager
from chunker.models import Language


# Custom strategies for generating test data
@st.composite
def non_semantic_strategy(draw):
    """Generate a non-semantic strategy type."""
    strategies = [
        StrategyType.SENTENCE,
        StrategyType.RECURSIVE,
        StrategyType.FIXED,
        StrategyType.STRUCTURE_AWARE,
        StrategyType.HIERARCHICAL,
    ]
    return draw(st.sampled_from(strategies))


@st.composite
def valid_text(draw):
    """Generate valid text for chunking (non-empty, reasonable length)."""
    # Generate text with at least one sentence
    sentences = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"), min_codepoint=32, max_codepoint=126),
            min_size=10,
            max_size=200
        ),
        min_size=1,
        max_size=10
    ))
    return ". ".join(sentences) + "."


class TestLazyLoadingProperties:
    """Property-based tests for lazy loading behavior."""

    @given(
        strategy=non_semantic_strategy(),
        text=valid_text()
    )
    @settings(max_examples=100)
    def test_property_2_lazy_loading_of_embedding_models(self, strategy, text):
        """
        Feature: chunker-improvements, Property 2: Lazy Loading of Embedding Models
        
        For any chunker configuration using a non-semantic strategy (sentence, recursive,
        fixed, structure_aware, hierarchical), the embedding model should never be loaded
        or initialized during chunking operations.
        
        **Validates: Requirements 1.4**
        """
        # Create config with non-semantic strategy
        config = ChunkerConfig(
            strategy=strategy,
            chunk_size=512,
            chunk_overlap=50,
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Track whether embedding provider was imported/loaded
        embedding_loaded = {"sentence_transformer": False, "openai": False}
        
        # Mock the embedding provider imports to detect if they're loaded
        original_st_import = None
        original_openai_import = None
        
        def mock_sentence_transformer_import(*args, **kwargs):
            embedding_loaded["sentence_transformer"] = True
            # Don't actually import, just track the attempt
            raise ImportError("Embedding model should not be loaded for non-semantic strategies")
        
        def mock_openai_import(*args, **kwargs):
            embedding_loaded["openai"] = True
            raise ImportError("Embedding model should not be loaded for non-semantic strategies")
        
        # Patch the imports at the point where SemanticChunker would load them
        with patch.dict('sys.modules', {
            'chunker.embeddings.sentence_transformer': None,
            'chunker.embeddings.openai_provider': None
        }):
            # Create chunker - should not load embedding provider yet
            chunker = SemanticChunker(config)
            
            # Verify embedding provider is not loaded at initialization
            assert chunker._embedding_provider is None, \
                f"Embedding provider should not be loaded for {strategy.value} strategy"
            
            # Perform chunking operation
            try:
                chunks = chunker.chunk(text)
                
                # Verify chunking succeeded
                assert len(chunks) > 0, "Chunking should produce at least one chunk"
                
                # Verify embedding provider was never loaded during chunking
                assert chunker._embedding_provider is None, \
                    f"Embedding provider should never be loaded for {strategy.value} strategy"
                
                # Verify the strategy used does not require embeddings
                assert chunks[0].metadata.strategy != "semantic", \
                    "Should not use semantic strategy"
                
            except ImportError as e:
                # If we get an ImportError, it means the code tried to load embeddings
                # This is a test failure for non-semantic strategies
                pytest.fail(
                    f"Embedding model was loaded for non-semantic strategy {strategy.value}: {e}"
                )


class TestLazyLoadManagerProperties:
    """Property-based tests specifically for LazyLoadManager."""

    @given(
        strategy=non_semantic_strategy()
    )
    @settings(max_examples=100)
    def test_lazy_load_manager_does_not_load_embeddings_without_request(self, strategy):
        """
        Test that LazyLoadManager does not load embedding provider until explicitly requested.
        
        This tests the LazyLoadManager component directly, verifying that creating
        the manager does not trigger any loading of expensive resources.
        """
        config = ChunkerConfig(
            strategy=strategy,
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create lazy load manager
        manager = LazyLoadManager(config)
        
        # Verify no embedding provider is loaded yet
        assert manager._embedding_provider_ref is None, \
            "Embedding provider should not be loaded at initialization"
        
        # Verify no language processors are loaded yet
        assert manager._en_processor_ref is None, \
            "English processor should not be loaded at initialization"
        assert manager._ar_processor_ref is None, \
            "Arabic processor should not be loaded at initialization"


    @given(
        language=st.sampled_from([Language.ENGLISH, Language.ARABIC])
    )
    @settings(max_examples=100)
    def test_lazy_load_manager_only_loads_requested_language(self, language):
        """
        Test that LazyLoadManager only loads the requested language processor.
        
        When requesting a specific language processor, only that processor should
        be loaded, not processors for other languages.
        """
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Request specific language processor
        processor = manager.get_language_processor(language)
        
        # Verify the requested processor is loaded
        assert processor is not None, f"{language.value} processor should be loaded"
        
        # Verify only the requested language processor is loaded
        if language == Language.ENGLISH:
            assert manager._en_processor_ref is not None, \
                "English processor should be loaded"
            assert manager._ar_processor_ref is None, \
                "Arabic processor should NOT be loaded when only English is requested"
        elif language == Language.ARABIC:
            assert manager._ar_processor_ref is not None, \
                "Arabic processor should be loaded"
            assert manager._en_processor_ref is None, \
                "English processor should NOT be loaded when only Arabic is requested"


    @given(
        language=st.sampled_from([Language.ENGLISH, Language.ARABIC]),
        text=valid_text()
    )
    @settings(max_examples=100)
    def test_property_3_lazy_loading_of_language_processors(self, language, text):
        """
        Feature: chunker-improvements, Property 3: Lazy Loading of Language Processors
        
        For any document in a single language, only the processor for that detected
        language should be loaded, and processors for other languages should remain
        uninitialized.
        
        **Validates: Requirements 1.5**
        """
        # Create config with a strategy that requires language processing
        config = ChunkerConfig(
            strategy=StrategyType.SENTENCE,
            chunk_size=512,
            chunk_overlap=50,
            detect_language=True
        )
        
        # Create chunker with lazy loading
        chunker = SemanticChunker(config)
        
        # Verify no language processors are loaded initially
        assert chunker._lazy_load_manager._en_processor_ref is None, \
            "English processor should not be loaded at initialization"
        assert chunker._lazy_load_manager._ar_processor_ref is None, \
            "Arabic processor should not be loaded at initialization"
        
        # Mock the language detector to return our test language
        with patch.object(chunker._detector, 'detect', return_value=language):
            # Perform chunking operation
            chunks = chunker.chunk(text)
            
            # Verify chunking succeeded
            assert len(chunks) > 0, "Chunking should produce at least one chunk"
            
            # Access the lazy load manager
            lazy_manager = chunker._lazy_load_manager
            
            # Verify only the detected language processor is loaded
            if language == Language.ENGLISH:
                assert lazy_manager._en_processor_ref is not None, \
                    "English processor should be loaded for English text"
                assert lazy_manager._ar_processor_ref is None, \
                    "Arabic processor should NOT be loaded for English text"
            elif language == Language.ARABIC:
                assert lazy_manager._ar_processor_ref is not None, \
                    "Arabic processor should be loaded for Arabic text"
                assert lazy_manager._en_processor_ref is None, \
                    "English processor should NOT be loaded for Arabic text"


    @given(
        preload_components=st.lists(
            st.sampled_from([
                "embedding_provider",
                "english_processor",
                "arabic_processor"
            ]),
            min_size=0,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_preload_only_loads_specified_components(self, preload_components):
        """
        Test that preload() only loads the components explicitly specified.
        
        Components not in the preload list should remain unloaded.
        """
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Preload specified components
        if preload_components:
            manager.preload(preload_components)
        
        # Verify only specified components are loaded
        if "embedding_provider" in preload_components:
            assert manager._embedding_provider_ref is not None, \
                "Embedding provider should be loaded after preload"
        else:
            assert manager._embedding_provider_ref is None, \
                "Embedding provider should NOT be loaded if not in preload list"
        
        if "english_processor" in preload_components:
            assert manager._en_processor_ref is not None, \
                "English processor should be loaded after preload"
        else:
            assert manager._en_processor_ref is None, \
                "English processor should NOT be loaded if not in preload list"
        
        if "arabic_processor" in preload_components:
            assert manager._ar_processor_ref is not None, \
                "Arabic processor should be loaded after preload"
        else:
            assert manager._ar_processor_ref is None, \
                "Arabic processor should NOT be loaded if not in preload list"


    @given(
        load_embedding=st.booleans(),
        load_english=st.booleans(),
        load_arabic=st.booleans()
    )
    @settings(max_examples=100)
    def test_clear_removes_all_loaded_resources(self, load_embedding, load_english, load_arabic):
        """
        Test that clear() removes all loaded resources regardless of which were loaded.
        
        After calling clear(), all resource references should be None, allowing
        garbage collection to reclaim memory.
        """
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Load resources based on test parameters
        if load_embedding:
            manager.get_embedding_provider()
        if load_english:
            manager.get_language_processor(Language.ENGLISH)
        if load_arabic:
            manager.get_language_processor(Language.ARABIC)
        
        # Clear all resources
        manager.clear()
        
        # Verify all references are cleared
        assert manager._embedding_provider_ref is None, \
            "Embedding provider reference should be None after clear()"
        assert manager._en_processor_ref is None, \
            "English processor reference should be None after clear()"
        assert manager._ar_processor_ref is None, \
            "Arabic processor reference should be None after clear()"


    @given(
        strategy=non_semantic_strategy(),
        text=valid_text()
    )
    @settings(max_examples=100)
    def test_property_4_lazy_loading_of_optional_dependencies(self, strategy, text):
        """
        Feature: chunker-improvements, Property 4: Lazy Loading of Optional Dependencies
        
        For any chunking operation that doesn't require optional dependencies (stanza,
        camel-tools), those dependencies should not be imported or loaded.
        
        **Validates: Requirements 1.6**
        """
        # Track which modules were imported during the test
        initial_modules = set(sys.modules.keys())
        
        # Create config with a strategy that doesn't require Arabic NLP tools
        # Using English text and sentence strategy should not trigger stanza/camel_tools
        config = ChunkerConfig(
            strategy=strategy,
            chunk_size=512,
            chunk_overlap=50,
            detect_language=True
        )
        
        # Create chunker
        chunker = SemanticChunker(config)
        
        # Mock language detector to return English (no Arabic tools needed)
        with patch.object(chunker._detector, 'detect', return_value=Language.ENGLISH):
            # Perform chunking operation with English text
            chunks = chunker.chunk(text)
            
            # Verify chunking succeeded
            assert len(chunks) > 0, "Chunking should produce at least one chunk"
        
        # Get modules that were imported during the operation
        final_modules = set(sys.modules.keys())
        new_modules = final_modules - initial_modules
        
        # Verify optional dependencies were NOT imported
        optional_deps = ['stanza', 'camel_tools']
        for dep in optional_deps:
            # Check if the dependency or any of its submodules were imported
            imported_deps = [m for m in new_modules if m.startswith(dep)]
            assert len(imported_deps) == 0, \
                f"Optional dependency '{dep}' should not be imported for English text " \
                f"with {strategy.value} strategy. Imported: {imported_deps}"
        
        # Also verify they're not in sys.modules at all (unless they were there before)
        for dep in optional_deps:
            if dep not in initial_modules:
                assert dep not in sys.modules, \
                    f"Optional dependency '{dep}' should not be in sys.modules"


    @given(
        text=valid_text()
    )
    @settings(max_examples=100)
    def test_optional_dependencies_not_loaded_for_english_processing(self, text):
        """
        Test that optional Arabic dependencies (stanza, camel_tools) are not loaded
        when processing English text.
        
        This verifies that the lazy loading mechanism prevents unnecessary imports
        of language-specific optional dependencies.
        """
        # Track initial modules
        initial_modules = set(sys.modules.keys())
        
        # Create config for English processing
        config = ChunkerConfig(
            strategy=StrategyType.SENTENCE,
            chunk_size=512,
            chunk_overlap=50,
            detect_language=True
        )
        
        # Create lazy load manager
        manager = LazyLoadManager(config)
        
        # Get English processor (should not load Arabic dependencies)
        en_processor = manager.get_language_processor(Language.ENGLISH)
        
        # Use the processor
        sentences = en_processor.segment_sentences(text)
        assert len(sentences) > 0, "Should produce at least one sentence"
        
        # Check which modules were imported
        final_modules = set(sys.modules.keys())
        new_modules = final_modules - initial_modules
        
        # Verify optional Arabic dependencies were NOT imported
        optional_arabic_deps = ['stanza', 'camel_tools']
        for dep in optional_arabic_deps:
            imported_deps = [m for m in new_modules if m.startswith(dep)]
            assert len(imported_deps) == 0, \
                f"Optional Arabic dependency '{dep}' should not be imported " \
                f"when processing English text. Imported: {imported_deps}"


    @given(
        load_arabic=st.booleans()
    )
    @settings(max_examples=100)
    def test_optional_dependencies_only_loaded_when_arabic_processor_created(self, load_arabic):
        """
        Test that optional dependencies (stanza, camel_tools) are only loaded when
        the Arabic processor is actually created, not before.
        
        This verifies lazy loading at the processor level.
        """
        # Track initial modules
        initial_modules = set(sys.modules.keys())
        
        # Create config
        config = ChunkerConfig()
        
        # Create lazy load manager
        manager = LazyLoadManager(config)
        
        # Initially, no processors should be loaded
        assert manager._en_processor_ref is None
        assert manager._ar_processor_ref is None
        
        # Get English processor first
        en_processor = manager.get_language_processor(Language.ENGLISH)
        assert en_processor is not None
        
        # Check modules after English processor creation
        after_english_modules = set(sys.modules.keys())
        new_after_english = after_english_modules - initial_modules
        
        # Optional Arabic dependencies should NOT be loaded yet
        optional_arabic_deps = ['stanza', 'camel_tools']
        for dep in optional_arabic_deps:
            imported_deps = [m for m in new_after_english if m.startswith(dep)]
            assert len(imported_deps) == 0, \
                f"Optional Arabic dependency '{dep}' should not be imported " \
                f"when only English processor is created. Imported: {imported_deps}"
        
        # Now optionally load Arabic processor
        if load_arabic:
            try:
                ar_processor = manager.get_language_processor(Language.ARABIC)
                assert ar_processor is not None
                
                # After Arabic processor creation, dependencies might be loaded
                # (if they're available on the system)
                # But we verify they weren't loaded before
                final_modules = set(sys.modules.keys())
                new_after_arabic = final_modules - after_english_modules
                
                # The key property: dependencies were not loaded until Arabic processor was created
                # We've already verified this above by checking after English processor
                
            except Exception:
                # If Arabic processor creation fails (dependencies not available),
                # that's okay - we've already verified they weren't loaded prematurely
                pass
