# Issue: Implement Deterministic Entailment Boundaries in REASONER

## Summary
The current `REASONER` module (v5) utilizes a heuristic-based verdict resolution strategy in [verdict_resolver.py](reasoner/verdict_resolver.py). While effective for initial prototyping, this approach relies on soft counts of inferred relationships and subjective confidence scores parsed from LLM outputs. To ensure mathematical rigor, zero-shot stability, and auditability, we must transition to the **Midpoint Thresholding** protocol defined in [RESEARCH.md](planner/RESEARCH.md).

## Problem Description
The existing `determine_verdict_type` and `calculate_confidence` functions in [verdict_resolver.py](reasoner/verdict_resolver.py) are highly dependent on the phrasing of LLM responses. This creates several risks:
- **Subjectivity**: Verdicts can shift based on small linguistic changes in the LLM's reasoning trace.
- **Auditability Gap**: Inability to mathematically prove why a claim was integrated as "Supported" vs "Unsupported".
- **Zero-Shot Fragility**: Standard LLMs are often overconfident; relying on raw text prevents utilizing the deterministic entailment boundaries of NLI backbones.

## Proposed Solution: The $M(d,c)$ Classification Function
Adopt the mathematically rigorous binary classification function for claim $c$ and grounded evidence $d$:

$$M(d,c) = \begin{cases} 1 \text{ (Supported)} & \text{if } z > t \\ 0 \text{ (Unsupported)} & \text{if } z \le t \end{cases}$$

Where:
- **$z$**: The raw scalar output (logit or probability) from the verification model.
- **$t$**: The **midpoint threshold**, calculated as $t = \frac{v_{\min} + v_{\max}}{2}$. 
    - For Softmax Probability ($z \in [0, 1]$): $t = 0.5$.
    - For Logit Space: The arithmetic mean of theoretical extrema.

## Implementation Steps

### 1. Raw Score Exposure [REASONER]
- Update the reasoning pipeline to capture the raw logit $z$ before any formatting or rounding occurs.
- If using an external NLI backbone (e.g., DeBERTa or MiniCheck), the `LLMClient` must be extended to return these scores alongside text labels.

### 2. Refactor Verdict Logic [REASONER]
- **Target**: [verdict_resolver.py](reasoner/verdict_resolver.py)
- Replace counting-based logic (`supporting > contradicting`) with categorical grounding via $M(d,c)$.
- Ensure a claim is only integrated as "Grounded Evidence" if $M(d,c) = 1$.

### 3. Enhanced Trace Logging [SCENE]
- Record the scalar $z$ in [ReasonerTrace](schemas/contracts.py) and [Finding](schemas/contracts.py) objects.
- This enables standardized auditing where any third party can verify the $z > t$ boundary.

## Scientific Justification
- **Zero-Shot Transfer**: Midpoint thresholding assumes no prior knowledge of the dataset distribution, essential for auditing systems where the prevalence of supported claims is unknown.
- **Calibration Independence**: Large NLI backbones maintain logical boundaries at the midpoint even when overconfident, avoiding the need for per-dataset temperature scaling.
- **Transparency**: Defining $t$ as a function of the model range rather than a tuned hyperparameter ensures the system is reproducible across different corpora.

## Graph Tuning Evaluation
Dense Graphs / Shallow Search: If the graph construction phase uses a low encoder threshold to merge many nodes (high synonymy), a shallow PPR walk (high α) is often sufficient to find related information.
Sparse Graphs / Deep Search: If construction is highly restrictive (high Pattern Separation), a deeper PPR walk (low α) is required to bridge the gaps between discrete concepts.