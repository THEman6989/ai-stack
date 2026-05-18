# Helper Repositories

This directory contains local reference checkouts used for AlphaRavis research.
These repositories are not runtime dependencies unless a focused task promotes a
specific implementation into the main codebase.

## `awesome-rag`

Source: <https://github.com/noworneverev/Awesome-RAG>

Use this as a discovery catalogue for RAG frameworks, retrievers, rerankers,
document-processing tools, vector stores, evaluation frameworks, and model
serving projects. Do not import code from the catalogue directly into
AlphaRavis; inspect the linked projects and their official docs before copying a
pattern.

## `langgraph-agentic-rag-template`

Sources:

- <https://docs.langchain.com/oss/python/langgraph/agentic-rag>
- <https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb>

Use this as the concrete LangGraph agentic-RAG Schablone. The current docs page
is the authoritative source; the downloaded notebook is an archival code sample.
The reusable pattern is the graph loop: decide whether to retrieve, run the
retriever tool, grade retrieved chunks, rewrite weak queries, then answer with
bounded context.
