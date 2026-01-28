# RESEARCH: The Three-Lane Logic for GALC

## 1. Taxonomic & Epistemic Overview
No single retrieval strategy is optimal for all query types. Aggregation creates a robust hybrid system that allows for **Epistemic Triangulation** (comparing structural relationships against direct text excerpts).

## 2. Lane A (Graph-LCI): High-Precision Pathfinding
Lane A utilizes **Personalized PageRank (PPR)**. To prevent signal washout by high-frequency hubs, we implement **Node Specificity** ($s_i$) per the HippoRAG protocol.

### Personalized PageRank (PPR) with Node Specificity
Let $S$ be the set of seed nodes and $P_i$ be the set of passages containing node $i$. We define node specificity $s_i$ as:
$$s_i = \frac{1}{|P_i|}$$

The teleportation vector $v$ is modulated such that:
$$v_i \propto \mathbb{1}_{i \in S} \cdot s_i$$

### Transition Probability & Synonymy
Edges consist of extracted relations ($E$) and synonymy links ($E'$). Synonymy weights $w_{ij}$ are defined by cosine similarity with a critical threshold $\tau$:
$$w_{ij} = \begin{cases} \text{cosine}(e_i, e_j) & \text{if } \text{cosine}(e_i, e_j) > 0.8 \\ 0 & \text{otherwise} \end{cases}$$
The transition matrix $T$ is normalized such that $\sum_j T_{ij} = 1$.

## 3. Lane B (Global): Hierarchical Modularity
LANE_B utilizes the **Leiden Algorithm** to maximize Modularity ($Q$) while enforcing recursive partition constraints. Unlike Louvain, Leiden guarantees well-connected communities, preventing "hallucinated" global themes.

### Resolution Parameters
We apply a hierarchical resolution parameter $\gamma$ to enable multi-level "Map-Reduce" summarization:
- **Level 0 (Roots):** Low $\gamma$ (Coarse-grained clusters).
- **Level 2 (Leaves):** High $\gamma$ (Fine-grained clusters).

## 4. Reasoner Verification: Entailment Boundaries
Verification follows the MiniCheck protocol. To enable zero-shot transfer without dataset tuning, we apply a binary classification function $M(d, c)$ based on the model's logit $z$.

### Decision Boundary
The decision threshold $t$ is defined as the probability midpoint:
$$M(d, c) = \begin{cases} 1 (\text{Supported}) & \text{if } z > t \\ 0 (\text{Unsupported}) & \text{if } z \leq t \end{cases}$$
Where $t = (v_{min} + v_{max}) / 2$.