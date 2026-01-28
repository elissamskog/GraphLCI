# CHUNKER: Strategic & Mathematical Synthesis for GALC

## 1. Taxonomic Overview
Chunking is the first stage of **Input Entropy Management**. Its goal is to maximize the **Context-to-Noise Ratio** within each segment to ensure the **PLANNER** can accurately route queries.

| Method | Logic Trigger | Complexity | Use Case |
| :--- | :--- | :--- | :--- |
| **Fixed-Size** | Token/Char Count ($N$) | $O(1)$ | Baseline / Low-resource |
| **Deterministic Semantic** | Structural Bounds ($\Sigma$) | $O(N)$ | Structured text (Wikipedia/Docs) |
| **Embedding-Based** | Vector Distance ($\Delta$) | $O(N \cdot D)$ | Thematically dense/fluid text |
| **Agentic/LLM** | Intent/Narrative ($\Psi$) | $O(N \cdot LLM)$ | High-precision logical parsing |

---

## 2. Deterministic Structural Logic
Deterministic methods rely on a **Recursive-Descent Hierarchy**. This prioritizes the document's inherent structural semantics over arbitrary token counts.

### Recursive Boundary Selection
The chunker attempts to split at the highest priority boundary ($\sigma$) that satisfies the size constraint $C \leq \text{target\_size}$:
1.  **Section Level:** `\n# | \n##` (Highest integrity).
2.  **Paragraph Level:** `\n\n+` (Standard semantic seam).
3.  **Sentence Level:** `[.!?]\s+` (Filtered for abbreviations like "Mr." or "Inc.").

### Contextual Anchoring (Math)
To prevent "orphaned" claims, we apply a sliding window overlap. Let $S$ be the total document tokens and $T$ be the target chunk size. The number of chunks $n$ is defined by:

$$n = \lceil \frac{S - O}{T - O} \rceil$$

Where:
* $T$: Target Token Count (e.g., 1200).
* $O$: Overlap Tokens (e.g., 100).



---

## 3. Stochastic & Vector-Based Methods
These methods utilize the **Semantic Gap** between text segments to determine breaks, rather than punctuation.

### Cosine Similarity Thresholding
Individual sentences $s_i$ are embedded into vectors $v_i$. A split is triggered at index $i$ if:

$$1 - \frac{v_i \cdot v_{i+1}}{\|v_i\| \|v_{i+1}\|} > \tau$$

Where $\tau$ is the **Similarity Threshold** (e.g., the 80th percentile of distance). This ensures each chunk is **Thematically Unified**, reducing "muddled" embeddings in the **PLANNER** phase.

---

## 4. System Interdependencies & Optimization

### The Named Entity & Encoder Link
The **CHUNKER** does not operate in isolation. Its efficiency is strictly coupled with the **EXTRACTOR** (Named Entity Recognition) and the **Encoder** model:
* **The "Muddled" Vector Problem:** If the chunker creates large, multi-topic chunks, the **Encoder** produces a diluted vector. This causes the **PLANNER** to fail its similarity threshold, defaulting to Lane C (Vector Only).
* **Extraction Recall vs. Fragmented Entities:** Smaller chunks improve vector precision but increase the risk of splitting a named entity from its qualifying context (e.g., "Arthur's Magazine" in Chunk A and its "1844" start date in Chunk B).

### Optimization Note
There exists a critical optimization balance between **Chunk Size ($T$)**, **Encoder Precision**, and **Extraction Latency**. 
* **Smaller $T$ (600 tokens):** Increases the number of nodes and total extraction time, but yields high-precision "Bridge Edges" in the **LIBRARIAN**.
* **Larger $T$ (1200+ tokens):** Reduces extraction calls but requires a **Gleaning Loop** (Phase 3) to prevent the LLM from missing entities due to "attention decay".

---

> **Note for Agent:** For current hardware (Mac Mini M4), prioritize **Deterministic Structural Splitting** at 600 tokens with 100-token overlap. This minimizes "fuzzy" embeddings and allows the **PLANNER** to successfully enter Lane A/B without the overhead of agentic chunking.