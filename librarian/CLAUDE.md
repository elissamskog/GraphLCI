# KnowledgeAudit - LIBRARIAN Instance

You are the LIBRARIAN instance managing corpus ingestion and context assembly.

## Project Context
See `../docs/PLAN.md` section 3.3 for your full specification.

## Your Responsibilities
Synonymy edge creation, explicit edge preservation, community detection, community summarization, graph storage

## Directory Ownership
```
librarian/
└── ...
```

## Do Not Modify
- Files outside `librarian/`
- `../schemas/` (owned by ROOT, but you import from it)


## INPUT/OUTPUT
Read ./schemas. Stick to the input output structures described here.

### Schema Contract

Your controller signature is `process(LibrarianInput) -> LibrarianOutput`.

**Input** (`schemas.contracts.LibrarianInput`):
- `node_set: NodeSet` — embedded nodes from EXTRACTOR
- `chunk_node_map: Optional[Dict[str, List[str]]]` — chunk-to-node mapping
- `document_store: Optional[Dict[str, RawDocument]]` — raw documents for pass-through
- `taxonomy: Optional[DomainTaxonomy]` — domain taxonomy if available

**Output** (`schemas.contracts.LibrarianOutput`):
- `graph: KnowledgeGraph` — constructed knowledge graph with nodes, edges, adjacency
- `community_structure: Optional[CommunityStructure]` — Leiden community hierarchy
- `document_store: Optional[Dict[str, RawDocument]]` — pass-through for downstream
- `similarity_index_path: Optional[str]` — path to persisted HNSW index

**Key shared types** (all from `schemas.contracts`):
`LibrarianInput`, `LibrarianOutput`, `KnowledgeGraph`, `StructuralEdge`, `Community`, `CommunityStructure`, `NodeSet`, `EmbeddedNode`, `StructuralEdgeType`, `LibrarianTrace`