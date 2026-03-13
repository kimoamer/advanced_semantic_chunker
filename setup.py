from setuptools import setup, find_packages

setup(
    name="advanced-semantic-chunker",
    version="1.1.0",
    description="Powerful bilingual (EN/AR) semantic document chunking engine for RAG pipelines",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "sentence-transformers": ["sentence-transformers>=2.2.0"],
        "openai": ["openai>=1.0.0"],
        "nlp": ["nltk>=3.8.0", "spacy>=3.6.0"],
        "arabic": ["stanza>=1.6.0", "camel-tools>=1.5.0"],
        "readers": ["pypdf>=3.0.0", "beautifulsoup4>=4.12.0", "ebooklib>=0.18"],
        "all": [
            "sentence-transformers>=2.2.0",
            "openai>=1.0.0",
            "nltk>=3.8.0",
            "stanza>=1.6.0",
            "pypdf>=3.0.0",
            "beautifulsoup4>=4.12.0",
            "ebooklib>=0.18",
            "tiktoken>=0.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
