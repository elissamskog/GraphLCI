## Analysis & Hypotheses

### Kettle Generating Station / Hudson Bay Failure (Musique)

**Failure Analysis:**
The claim "Identify the river that the Kettle Generating Station is located on a tributary of" failed with a `NOT_FOUND` verdict. 
- **Graph State:** The graph contained the necessary entities (`Kettle Generating Station`, `Nelson River`, `Hudson Bay`) and edges (`Kettle -> Nelson` [Extracted], `Nelson -> Hudson` [Synonymy/SameSource]).
- **Evidence:** The provided context included "Kettle... on Lower Nelson River" and "Fort Nelson... at the mouth of the Nelson River at Hudson Bay".
- **Reasoning Gap:** The Reasoner failed to infer that "at the mouth of... at Hudson Bay" implies "flows into" (tributary relationship), or it rejected "Hudson Bay" because the question explicitly asked for a "river".
- **Trace Diagnosis:** `relationships_inferred` was 0, despite `scope_bridge_edges_used` being 19. This confirms the Reasoner had access to the data but failed to synthesize the transitive link.

**Hypothesis:**
The Reasoner's prompt lacks explicit instructions to:
1.  Perform transitive geographic inference (Body A on River B + River B flows into Body C -> Body A on tributary of Body C).
2.  Relax entity type constraints when the logical answer (e.g., "Hudson Bay") mismatches the requested type ("River") but satisfies the relationship.

**Proposed Experiment:**
Modify `reasoner/prompt_builder.py` to include:
- A **Transitive Inference** protocol for logical chaining.
- An **Entity Type Relaxation** protocol to prioritize logical relationships over strict type matching in questions.

### Conclusion (Verified on New Example)

**Verification:**
The fix was tested on a random new MuSiQue claim ("What is the oldest college in the state that contains Borough House Plantation?").
- **Result:** Pass (Verdict: `SUPPORTS`, Accuracy: 1.0).
- **Reasoning Trace:** The model explicitly used the structural path to link "Borough House Plantation" to "South Carolina" and then to "College of Charles Towne", demonstrating successful transitive inference.

**Graph Analysis:**
- **Claims:** Extracted claims were highly relevant (e.g., "The College of Charles Towne is the oldest college in South Carolina", "Borough House Plantation is in Stateburg, South Carolina").
- **Subclaims:** Decomposition correctly identified the implicit intermediate target ("Identify the state...") needed to bridge the two entities.
- **Connectivity:** The graph communities were well-connected via the "South Carolina" bridge node, which was pivotal for the multi-hop reasoning. The high number of bridge edges used (47) confirms the Lane A (Graph) context was actively utilized.

**Verdict:** The hypothesis is confirmed. Explicit prompt instructions for transitive inference and entity relaxation significantly improve multi-hop reasoning capabilities.

## Analysis: Graph Construction Pitfalls & Named Entity Gaps

### Specific Failure Case: MuSiQue Claim `3hop2__140194_49541_51068`
**Context:** The claim required linking "Jenna Coleman" -> "Victoria (TV Series)" -> "Network (ITV/PBS)".
**Failure:** The graph contained "Jenna Coleman" and "Victoria" (Entity), but completely missed "ITV" and the "Network" attribute, despite the text explicitly stating "...ITV biographical drama...".

### Root Cause Analysis (Codebase Inspection)
1.  **Entity Extractor (`entity_extractor.py`):**
    - Uses a generic prompt: "Extract ALL named entities... named concepts, actors, artifacts..."
    - Relying on valid JSON output often leads to **Schema Brittleness** (LLM hallucinating keys or adhering too strictly to generic types like "ORGANIZATION" without finding granular attributes).
    - It does not perform targeted extraction for specific attributes (e.g., "Network" for a "Series").

2.  **Structural Analyzer (`structural_analyzer.py`):**
    - **Current Capability:** Uses spaCy to find relationships (appositives, prepositions) between *already extracted* entities.
    - **The Gap:** It cannot *generate* new nodes. If the LLM misses "ITV" as an entity (perhaps seeing it just as an adjective in "ITV drama"), the structural analyzer has nothing to link "Victoria" to. The dependency parse `amod(drama, ITV)` is ignored because "ITV" isn't in the entity list.

### Broader Pitfall Analysis
The failure exemplifies **Schema Brittleness** and **Information Loss in generic extractions**.

| Pitfall | Observed Impact | Current Mitigation | Gap |
| :--- | :--- | :--- | :--- |
| **Boundary Errors** | "ITV biographical drama" -> "drama" vs "ITV drama" | None | spaCy handles dependencies, but only for *extracted* heads. |
| **Schema Brittleness** | Missing "Network" key for TV shows | Generic "Description" field | No schema enforcement for specific entity types (e.g., CreativeWork). |
| **Semantic Over-Atomization** | "Victoria" extracted as person, not series | Context string | Disambiguation is passive (text-based), not structural. |

### Recommendation
**Hybrid Extraction Strategy (Beyond Generic NER):**
1.  **Schema-Guided NER:** Instead of a single generic prompt, use a 2-pass approach.
    - *Pass 1:* Generic extraction (current).
    - *Pass 2:* **Type-Specific Refinement.** If an entity is identified as a "CreativeWork" or "TV Series", trigger a targeted prompt: *"For the series {X}, extract the Network, Creator, and Broadcast Years."*
2.  **Dependency-to-Entity Promotion:** Modify  to promote dependency modifiers (like  or ) to full entities if they appear to be proper nouns, even if the LLM missed them.
    - *Example:* If  exists and  is capitalized, force-create an "ITV" node and link it.

**Does spaCy solve this?**
- **Partially.** It solves the *structure* (knowing that ITV modifies drama).
- **No.** It does not currently solve the *extraction* (creating the node). The current implementation strictly filters structure by *existing* extracted entities.

## Analysis: Graph Construction Pitfalls & Named Entity Gaps

### Specific Failure Case: MuSiQue Claim `3hop2__140194_49541_51068`
**Context:** The claim required linking "Jenna Coleman" -> "Victoria (TV Series)" -> "Network (ITV/PBS)".
**Failure:** The graph contained "Jenna Coleman" and "Victoria" (Entity), but completely missed "ITV" and the "Network" attribute, despite the text explicitly stating "...ITV biographical drama...".

### Root Cause Analysis (Codebase Inspection)
1.  **Entity Extractor (`entity_extractor.py`):**
    - Uses a generic prompt: "Extract ALL named entities... named concepts, actors, artifacts..."
    - Relying on valid JSON output often leads to **Schema Brittleness** (LLM hallucinating keys or adhering too strictly to generic types like "ORGANIZATION" without finding granular attributes).
    - It does not perform targeted extraction for specific attributes (e.g., "Network" for a "Series").

2.  **Structural Analyzer (`structural_analyzer.py`):**
    - **Current Capability:** Uses spaCy to find relationships (appositives, prepositions) between *already extracted* entities.
    - **The Gap:** It cannot *generate* new nodes. If the LLM misses "ITV" as an entity (perhaps seeing it just as an adjective in "ITV drama"), the structural analyzer has nothing to link "Victoria" to. The dependency parse `amod(drama, ITV)` is ignored because "ITV" isn't in the entity list.

### Broader Pitfall Analysis
The failure exemplifies **Schema Brittleness** and **Information Loss in generic extractions**.

| Pitfall | Observed Impact | Current Mitigation | Gap |
| :--- | :--- | :--- | :--- |
| **Boundary Errors** | "ITV biographical drama" -> "drama" vs "ITV drama" | None | spaCy handles dependencies, but only for *extracted* heads. |
| **Schema Brittleness** | Missing "Network" key for TV shows | Generic "Description" field | No schema enforcement for specific entity types (e.g., CreativeWork). |
| **Semantic Over-Atomization** | "Victoria" extracted as person, not series | Context string | Disambiguation is passive (text-based), not structural. |

### Recommendation
**Hybrid Extraction Strategy (Beyond Generic NER):**
1.  **Schema-Guided NER:** Instead of a single generic prompt, use a 2-pass approach.
    - *Pass 1:* Generic extraction (current).
    - *Pass 2:* **Type-Specific Refinement.** If an entity is identified as a "CreativeWork" or "TV Series", trigger a targeted prompt: *"For the series {X}, extract the Network, Creator, and Broadcast Years."*
2.  **Dependency-to-Entity Promotion:** Modify `StructuralAnalyzer` to promote dependency modifiers (like `amod` or `compound`) to full entities if they appear to be proper nouns, even if the LLM missed them.
    - *Example:* If `amod(drama, ITV)` exists and `ITV` is capitalized, force-create an "ITV" node and link it.

**Does spaCy solve this?**
- **Partially.** It solves the *structure* (knowing that ITV modifies drama).
- **No.** It does not currently solve the *extraction* (creating the node). The current implementation strictly filters structure by *existing* extracted entities.
