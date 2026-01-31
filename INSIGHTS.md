# Technical Insights

## Graph Topology & Search
### Hub Suppression in PPR
**Observation:** In graph-based retrieval, high-degree nodes (hubs) like common entities or broad concepts tend to absorb disproportionate probability mass during Personalized PageRank (PPR) random walks. This "gravity well" effect starves specific, lower-degree nodes that are often the crucial bridges for multi-hop reasoning.
**Solution:** Implement **Degree Discounting** in the transition phase. dividing the transition weight by `log2(degree + 2)` normalizes the flow, ensuring that the walker prefers specific, meaningful connections over generic hubs. This significantly gains recall for "bridge" nodes in multi-hop chains.

## Extraction Strategy
### Shared Context & Transitive Inference
**Observation:** Standard OpenIE extraction often treats compound sentences as separate atomic facts, losing the connective tissue. For example, "The library is in the city where the author was born" results in `(Library, in, City)` and `(Author, born_in, City)` but fails to explicitly link `(Library, same_location_as, Author)` if the "City" node is generic or disconnected.
**Solution:** Add a explicit **Shared Context Rule** to the extraction prompt. Instructing the model to generate synthetic "bridge links" (e.g., "X is in the same city as Y") for entities appearing in the same relational scope creates the necessary edges for the graph walker to traverse, enabling transitive inference that purely atomic extraction misses.
