"""
Demo: Semantic Chunker — Bilingual EN/AR Document Chunking

This example demonstrates all available chunking strategies
on both English and Arabic text, without requiring any embedding
models or API keys (uses sentence strategy by default).

Includes:
  - chunk_docx(): Read a .docx file, chunk it, save chunks to JSON
  - save_chunks(): Save any chunk list to a JSON file
  - Full demo of all strategies

For the full semantic strategy, install sentence-transformers:
    pip install sentence-transformers

For .docx support:
    pip install python-docx
"""

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunker import SemanticChunker, ChunkerConfig, StrategyType
from chunker.config import ThresholdType
from chunker.models import Chunk


# ═══════════════════════════════════════════════════════════════
# Chunk Saving & DOCX Functions
# ═══════════════════════════════════════════════════════════════


def save_chunks(
    chunks: List[Chunk],
    output_path: str,
    include_stats: bool = True,
    chunker: Optional[SemanticChunker] = None,
) -> str:
    """
    Save chunks to a JSON file.

    Parameters
    ----------
    chunks : List[Chunk]
        List of chunks to save.
    output_path : str
        Path to the output JSON file.
    include_stats : bool
        Whether to include chunking statistics in the output.
    chunker : SemanticChunker, optional
        Chunker instance (for stats computation).

    Returns
    -------
    str
        The absolute path to the saved file.
    """
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }

    if include_stats and chunker:
        output["stats"] = chunker.get_stats(chunks)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    abs_path = os.path.abspath(output_path)
    print(f"  💾 Saved {len(chunks)} chunks → {abs_path}")
    return abs_path


def read_docx(docx_path: str) -> str:
    """
    Read text content from a .docx file.

    Extracts all paragraphs and preserves paragraph breaks.
    Requires: pip install python-docx

    Parameters
    ----------
    docx_path : str
        Path to the .docx file.

    Returns
    -------
    str
        Extracted text with paragraphs separated by double newlines.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required to read .docx files. "
            "Install it with: pip install python-docx"
        )

    doc = Document(docx_path)
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect heading styles and convert to markdown-style headers
        if para.style and para.style.name:
            style = para.style.name.lower()
            if "heading 1" in style:
                text = f"# {text}"
            elif "heading 2" in style:
                text = f"## {text}"
            elif "heading 3" in style:
                text = f"### {text}"
            elif "heading 4" in style:
                text = f"#### {text}"

        paragraphs.append(text)

    return "\n\n".join(paragraphs)


def chunk_docx(
    docx_path: str,
    output_path: Optional[str] = None,
    strategy: StrategyType = StrategyType.SENTENCE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs,
) -> List[Chunk]:
    """
    Read a .docx file, chunk its content, and save chunks to a JSON file.

    Parameters
    ----------
    docx_path : str
        Path to the input .docx file.
    output_path : str, optional
        Path to the output JSON file. If not specified, saves next to
        the .docx file with '_chunks.json' suffix.
    strategy : StrategyType
        Chunking strategy to use (default: SENTENCE).
    chunk_size : int
        Target chunk size in tokens (default: 512).
    chunk_overlap : int
        Overlap between chunks in tokens (default: 64).
    **kwargs
        Additional ChunkerConfig parameters.

    Returns
    -------
    List[Chunk]
        The generated chunks.

    Example
    -------
    >>> chunks = chunk_docx("report.docx")
    >>> chunks = chunk_docx("report.docx", output_path="output/report_chunks.json")
    >>> chunks = chunk_docx("report.docx", strategy=StrategyType.RECURSIVE, chunk_size=256)
    """
    docx_path = os.path.abspath(docx_path)

    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    print(f"\n  📄 Reading: {docx_path}")

    # ── Read .docx ──
    text = read_docx(docx_path)
    print(f"  📝 Extracted {len(text)} characters from document")

    if not text.strip():
        print("  ⚠️  Document is empty, no chunks to create.")
        return []

    # ── Determine output path ──
    if output_path is None:
        docx_stem = Path(docx_path).stem
        output_dir = os.path.dirname(docx_path)
        output_path = os.path.join(output_dir, f"{docx_stem}_chunks.json")

    # ── Configure chunker ──
    config_params = {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chunk_size": 50,
        "detect_language": True,
        "preserve_structure": True,
    }
    config_params.update(kwargs)

    config = ChunkerConfig(**config_params)
    chunker = SemanticChunker(config)

    # ── Chunk ──
    source_file = os.path.basename(docx_path)
    chunks = chunker.chunk(text, source_file=source_file)

    # ── Print stats ──
    stats = chunker.get_stats(chunks)
    print(f"  📊 Strategy:  {stats.get('strategy', 'N/A')}")
    print(f"  📊 Chunks:    {stats['total_chunks']}")
    print(f"  📊 Tokens:    {stats['total_tokens']}")
    print(f"  📊 Avg/chunk: {stats['avg_tokens_per_chunk']:.0f} tokens")
    print(f"  📊 Languages: {stats['language_distribution']}")

    # ── Save ──
    save_chunks(chunks, output_path, include_stats=True, chunker=chunker)

    return chunks


# ═══════════════════════════════════════════════════════════════
# Sample Documents
# ═══════════════════════════════════════════════════════════════

ENGLISH_DOC = """
# Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that focuses on building
systems that learn from data. Unlike traditional programming where rules are
explicitly coded, machine learning algorithms identify patterns and make
decisions with minimal human intervention.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data.
It maps input features to known output labels. Common applications include
image classification, spam detection, and medical diagnosis.

### Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to
find hidden patterns or intrinsic structures. Clustering and dimensionality
reduction are typical unsupervised techniques.

### Reinforcement Learning
Reinforcement learning involves an agent that learns to make decisions
by interacting with an environment. The agent receives rewards or penalties
for its actions and learns to maximize cumulative reward over time.

## Deep Learning

Deep learning is a subset of machine learning based on artificial neural
networks with multiple layers. These deep neural networks can learn
hierarchical representations of data, enabling them to solve complex
problems like natural language understanding, computer vision, and
autonomous driving.

## Practical Considerations

When implementing machine learning systems, several factors must be
considered: data quality, model selection, computational resources,
interpretability, and ethical implications. The choice of algorithm
depends heavily on the specific problem and available data.
"""

ARABIC_DOC = """
# مقدمة في التعلم الآلي

التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يركز على بناء أنظمة تتعلم من البيانات.
على عكس البرمجة التقليدية حيث يتم ترميز القواعد بشكل صريح، تحدد خوارزميات التعلم
الآلي الأنماط وتتخذ القرارات بأقل تدخل بشري.

## أنواع التعلم الآلي

### التعلم الموجه
في التعلم الموجه يتعلم الخوارزمية من بيانات التدريب المصنفة.
يقوم بتعيين ميزات الإدخال إلى تسميات الإخراج المعروفة.
تشمل التطبيقات الشائعة تصنيف الصور واكتشاف البريد العشوائي والتشخيص الطبي.

### التعلم غير الموجه
يعمل التعلم غير الموجه مع البيانات غير المصنفة.
يحاول الخوارزمية العثور على أنماط مخفية أو هياكل جوهرية.
التجميع وتقليل الأبعاد هي تقنيات نموذجية للتعلم غير الموجه.

## التعلم العميق

التعلم العميق هو مجموعة فرعية من التعلم الآلي تعتمد على الشبكات العصبية
الاصطناعية ذات الطبقات المتعددة. يمكن لهذه الشبكات العصبية العميقة تعلم
تمثيلات هرمية للبيانات، مما يمكنها من حل مشكلات معقدة مثل فهم اللغة الطبيعية
والرؤية الحاسوبية والقيادة الذاتية.
"""

MIXED_DOC = """
# Bilingual AI Research Report

Artificial Intelligence is transforming industries worldwide.
The latest breakthroughs in deep learning have enabled remarkable progress.

## الملخص العربي
الذكاء الاصطناعي يغير الصناعات في جميع أنحاء العالم.
أحدث الاختراقات في التعلم العميق مكنت من تحقيق تقدم ملحوظ.

## Key Findings
The research demonstrates significant improvements in multilingual NLP.
Models like BGE-M3 show strong performance across both English and Arabic.
"""


def print_separator(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


def print_chunks(chunks, show_metadata: bool = True):
    for i, chunk in enumerate(chunks):
        print(f"  ┌─ Chunk {i+1}/{len(chunks)} ─────────────────────")
        print(f"  │ Language: {chunk.metadata.language.value}")
        print(f"  │ Script:   {chunk.metadata.script}")
        print(f"  │ Tokens:   ~{chunk.metadata.token_count}")
        print(f"  │ Chars:    {chunk.metadata.char_count}")
        print(f"  │ Strategy: {chunk.metadata.strategy}")
        if chunk.metadata.heading_path:
            print(f"  │ Path:     {' › '.join(chunk.metadata.heading_path)}")
        if chunk.metadata.section_title:
            print(f"  │ Section:  {chunk.metadata.section_title}")
        if chunk.metadata.contains_header:
            print(f"  │ Contains: header")
        if chunk.metadata.contains_list:
            print(f"  │ Contains: list")
        if chunk.metadata.contains_table:
            print(f"  │ Contains: table")
        if chunk.metadata.contains_code:
            print(f"  │ Contains: code")
        text_preview = chunk.text[:150].replace("\n", "\\n")
        print(f"  │ Text:     {text_preview}...")
        print(f"  └───────────────────────────────────────")
        print()


def demo_strategy(strategy: StrategyType, text: str, label: str, **kwargs):
    """Run a chunking strategy and display results."""
    config_kwargs = {
        "strategy": strategy,
        "chunk_size": 200,
        "chunk_overlap": 0,
        "min_chunk_size": 30,
        "verbose": False,
    }
    config_kwargs.update(kwargs)

    config = ChunkerConfig(**config_kwargs)
    chunker = SemanticChunker(config)

    chunks = chunker.chunk(text, source_file=f"{label}.txt")
    stats = chunker.get_stats(chunks)

    print(f"  📊 Total chunks: {stats['total_chunks']}")
    print(f"  📊 Total tokens: {stats['total_tokens']}")
    print(f"  📊 Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.0f}")
    print(f"  📊 Languages: {stats['language_distribution']}")
    print()

    print_chunks(chunks)

    # ── Save chunks to JSON ──
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    output_path = os.path.join(output_dir, f"{label}_chunks.json")
    save_chunks(chunks, output_path, include_stats=True, chunker=chunker)

    return chunks


def main():
    print("\n" + "🚀" * 20)
    print("  SEMANTIC CHUNKER — Bilingual EN/AR Demo")
    print("🚀" * 20)

    # ── 1. Structure-Aware (NEW DEFAULT) on English ──
    print_separator("1. STRUCTURE-AWARE STRATEGY (DEFAULT) — English Document")
    demo_strategy(StrategyType.STRUCTURE_AWARE, ENGLISH_DOC, "english_structure_aware")

    # ── 2. Structure-Aware on Arabic ──
    print_separator("2. STRUCTURE-AWARE STRATEGY — Arabic Document")
    demo_strategy(StrategyType.STRUCTURE_AWARE, ARABIC_DOC, "arabic_structure_aware")

    # ── 3. Structure-Aware on Mixed ──
    print_separator("3. STRUCTURE-AWARE STRATEGY — Mixed EN/AR Document")
    demo_strategy(StrategyType.STRUCTURE_AWARE, MIXED_DOC, "mixed_structure_aware")

    # ── 1. Sentence Strategy on English ──
    print_separator("1. SENTENCE STRATEGY — English Document")
    demo_strategy(StrategyType.SENTENCE, ENGLISH_DOC, "english_sentence")

    # ── 2. Sentence Strategy on Arabic ──
    print_separator("2. SENTENCE STRATEGY — Arabic Document")
    demo_strategy(StrategyType.SENTENCE, ARABIC_DOC, "arabic_sentence")

    # ── 3. Recursive Strategy ──
    print_separator("3. RECURSIVE STRATEGY — English Document")
    demo_strategy(StrategyType.RECURSIVE, ENGLISH_DOC, "english_recursive")

    # ── 4. Hierarchical Strategy ──
    print_separator("4. HIERARCHICAL STRATEGY — English Document")
    demo_strategy(StrategyType.HIERARCHICAL, ENGLISH_DOC, "english_hierarchical")

    # ── 5. Fixed-Size Strategy ──
    print_separator("5. FIXED-SIZE STRATEGY — Arabic Document")
    demo_strategy(StrategyType.FIXED, ARABIC_DOC, "arabic_fixed", chunk_size=100)

    # ── 6. Mixed Language Document ──
    print_separator("6. SENTENCE STRATEGY — Mixed EN/AR Document")
    demo_strategy(StrategyType.SENTENCE, MIXED_DOC, "mixed_sentence")

    # ── 7. Serialization Demo ──
    print_separator("7. SERIALIZATION — JSON Export")
    config = ChunkerConfig(strategy=StrategyType.SENTENCE, chunk_size=300)
    chunker = SemanticChunker(config)
    chunks = chunker.chunk(ENGLISH_DOC[:500])

    print("  First chunk as JSON:")
    print(json.dumps(chunks[0].to_dict(), indent=2, ensure_ascii=False)[:500])
    print("  ...")

    # ── 8. DOCX Processing Demo ──
    print_separator("8. DOCX PROCESSING")
    print("  To chunk a .docx file, use the chunk_docx() function:")
    print()
    print("    from examples.demo import chunk_docx")
    print('    chunks = chunk_docx("path/to/document.docx")')
    print()
    print("  Or from command line:")
    print('    py examples/demo.py --docx "path/to/document.docx"')
    print()

    print("\n✅ Demo complete! All strategies working correctly.")
    print(f"📁 Chunk files saved to: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Chunker — Bilingual EN/AR Demo"
    )
    parser.add_argument(
        "--docx",
        type=str,
        help="Path to a .docx file to chunk",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: <docx_name>_chunks.json)",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="sentence",
        choices=[
            "sentence", "recursive", "fixed", "hierarchical",
            "semantic", "structure_aware",
        ],
        help="Chunking strategy (default: sentence)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between chunks in tokens (default: 64)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model for semantic strategy (default: BAAI/bge-m3)",
    )

    args = parser.parse_args()

    if args.docx:
        # ── DOCX mode ──
        strategy_map = {
            "sentence": StrategyType.SENTENCE,
            "recursive": StrategyType.RECURSIVE,
            "fixed": StrategyType.FIXED,
            "hierarchical": StrategyType.HIERARCHICAL,
            "semantic": StrategyType.SEMANTIC,
            "structure_aware": StrategyType.STRUCTURE_AWARE,
        }
        strategy = strategy_map[args.strategy]

        extra_kwargs = {}
        if args.strategy == "semantic":
            extra_kwargs["embedding_model"] = args.embedding_model

        chunks = chunk_docx(
            docx_path=args.docx,
            output_path=args.output,
            strategy=strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            **extra_kwargs,
        )

        print(f"\n  ✅ Done! {len(chunks)} chunks created.\n")
    else:
        # ── Full demo mode ──
        main()
