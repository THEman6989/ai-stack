# LangGraph Agentic RAG Template

Local reference copy for AlphaRavis RAG-router work.

## Sources

- Current LangChain docs:
  <https://docs.langchain.com/oss/python/langgraph/agentic-rag>
- Archived LangGraph notebook:
  <https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb>

Downloaded files:

- `langchain_agentic_rag_doc.html`
- `langgraph_agentic_rag.ipynb`

The notebook itself notes that the `examples/rag` directory is archival and no
longer updated. Treat the docs page as the current source and the notebook as a
concrete older implementation sample.

## Pattern To Reuse

The useful architecture is not the demo vector store itself. The reusable part
is the LangGraph control loop:

```text
agent / generate_query_or_respond
  -> decides direct answer vs retriever tool call

retrieve
  -> runs the retriever tool

grade_documents
  -> checks whether retrieved chunks are relevant

rewrite_question
  -> improves the query when retrieval was weak

generate_answer
  -> answers only after relevant retrieved context is available
```

The graph shape is:

```text
START
  -> agent
  -> if tool call: retrieve
  -> grade retrieved docs
  -> if relevant: generate answer -> END
  -> if not relevant: rewrite question -> agent
  -> if no tool call: END
```

## AlphaRavis Mapping

Use this as a Schablone for the next AlphaRavis steps:

- `agent` / `generate_query_or_respond`:
  AlphaRavis LangGraph node decides whether a normal answer is enough, whether
  thread-active document RAG should run automatically, or whether the model
  should call `query_archive(...)`.
- `retrieve`:
  call `retrieval_router.query_sources_with_backends(...)` rather than a demo
  in-memory retriever.
- `grade_documents`:
  future relevance gate over returned AlphaRavis/rag_api chunks before injecting
  them into answer context.
- `rewrite_question`:
  future query rewrite for archive questions such as "wie war das nochmal" into
  a sharper retrieval query.
- `generate_answer`:
  answer with bounded chunks only; keep `read_archive_record(...)` as explicit
  exact-history fallback.

## Not To Copy Directly

- Do not use the demo `InMemoryVectorStore` for AlphaRavis runtime.
- Do not use the demo Lilian Weng web loader flow as production ingest.
- Do not hard-code OpenAI models from the notebook.
- Do not bypass AlphaRavis thread/source authorization or archive ownership.

The implementation should stay behind `retrieval_router.py` so the same
LangGraph control loop can use AlphaRavis pgvector, `rag_api`, direct LangChain
retrievers, and later reranking without changing tool signatures.
