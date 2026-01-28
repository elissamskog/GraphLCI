# Architectural Decision Records

## Decision 1: Per-Row Indexing for WikiFactClaims

**Context**: WikiFactClaims derives from HotpotQA. Each row bundles a question with 10 inline Wikipedia passages and 3 labeled claims. We needed an evaluation strategy for this self-contained structure.

**Options Considered**: (A) Build one global index across all rows' passages. (B) Build a separate index per row.

**Choice**: Per-row indexing — each row's 10 passages are indexed independently.

**Rationale**: Each row is a closed-world evaluation unit. A global corpus would let the retriever accidentally surface passages from unrelated rows, inflating recall through contamination. Per-row indexing also enables direct title-matching between gold supporting-fact annotations and planner scope output, giving clean scope-recall measurement.

**Trade-offs**: Per-row indexing is slower (rebuild per row) and prevents testing cross-row generalization. Graphs are small (10 docs), limiting community detection quality.

---

## Decision 2: Three Reasoning Lanes (A/B/C)

**Context**: Queries vary from simple fact lookups to multi-document comparisons to broad thematic synthesis. A single reasoning path cannot serve all intents efficiently.

**Options Considered**: (A) One universal reasoning path. (B) Two paths (graph vs. vector). (C) Three intent-aligned lanes.

**Choice**: Three lanes — Graph-LCI (A), Graph-Only (B), Vector-Only (C).

**Rationale**: Different intents need different evidence shapes. Comparisons need full documents with edge grammar showing inter-document relationships (Lane A). Sensemaking needs hierarchical community summaries (Lane B). Lookups need fast ranked chunks (Lane C). Intent classification routes to the right lane, with automatic fallback to Lane C when graph scope is too small or confidence is low.

**Trade-offs**: Routing errors send queries to suboptimal lanes. The classifier adds latency and a failure mode. More lanes mean more context-assembly logic to maintain.

---

## Decision 3: Graph + Vector Hybrid Instead of Pure Vector RAG

**Context**: Claim verification requires connecting evidence scattered across documents that may use entirely different vocabulary — e.g., a contract capping liability and a regulation requiring unlimited obligations share entities but no keywords.

**Options Considered**: (A) Pure vector similarity search. (B) Pure knowledge graph traversal. (C) Hybrid: vectors for embedding similarity, graph for structural traversal.

**Choice**: Hybrid — vectors create edges via cosine similarity; graph algorithms (PPR, Leiden) handle traversal and clustering.

**Rationale**: Vector search finds lexically similar passages but misses structurally connected ones. Graph traversal follows entity co-occurrence, synonymy, and cross-document bridges to surface evidence that shares no vocabulary with the query. The "Graph Contribution Delta" metric (target >= 0.10) directly measures whether the graph adds value beyond vector-only retrieval.

**Trade-offs**: Higher indexing cost (entity extraction + graph construction). Graph quality depends on extraction recall. Small corpora produce sparse graphs with limited traversal benefit.

---

## Decision 4: Synthetic NOT_ENOUGH_INFO Claims from Unmentioned Topics

**Context**: A verification system must distinguish "evidence confirms/denies this" from "no relevant evidence exists." Without abstention testing, the system could score well by always guessing SUPPORTS or REFUTES.

**Options Considered**: (A) Only test SUPPORTS and REFUTES claims. (B) Use naturally occurring unanswerable claims from existing datasets. (C) Generate plausible claims about topics absent from the context passages.

**Choice**: Each WikiFactClaims row includes a third claim about a tangentially related but unmentioned topic (e.g., for a question about which magazine started first, a NOT_ENOUGH_INFO claim about their relative circulation figures).

**Rationale**: These claims sound plausible given the subject matter but have no supporting evidence in the 10 context passages. This forces the system to recognize absence of evidence rather than hallucinating a verdict, testing calibration alongside accuracy.

**Trade-offs**: Synthetic claims may be accidentally answerable from world knowledge. Plausibility varies with generation quality. The three-way label split means each verdict class has fewer examples.
