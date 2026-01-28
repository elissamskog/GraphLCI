# WikiFactClaims Per-Row v5 Evaluation Results

## Configuration

- **Pipeline:** v5
- **LLM:** qwen2.5:14b
- **Embedding:** nomic-embed-text
- **Groups evaluated:** 1
- **Total claims:** 3 (completed: 3, errors: 0)
- **Total index time:** 959.6s
- **Timestamp:** 2026-01-28T20:14:27.307673

## Summary Metrics

| Metric | Value |
|--------|-------|
| Verdict Accuracy | 0.667 |
| Evidence Doc Recall | 0.500 |
| Planner Scope Recall | 1.000 |
| Mean Confidence | 0.693 |

## Per-Verdict Breakdown

| Expected | Count | Correct | Accuracy |
|----------|-------|---------|----------|
| SUPPORTS | 1 | 0 | 0.000 |
| REFUTES | 1 | 1 | 1.000 |
| NOT_ENOUGH_INFO | 1 | 1 | 1.000 |

## Confusion Matrix

Rows = Expected, Columns = Predicted

| Expected \ Predicted | SUPPORTS | REFUTES | NOT_ENOUGH_INFO |
|---|---|---|---|
| SUPPORTS | 0 | 0 | 1 |
| REFUTES | 0 | 1 | 0 |
| NOT_ENOUGH_INFO | 0 | 0 | 1 |

## Per-Type Accuracy

| Question Type | Accuracy |
|---------------|----------|
| comparison | 0.667 |

## Per-Level Accuracy

| Level | Accuracy |
|-------|----------|
| medium | 0.667 |

## Per-Claim Results

| Group | Idx | Claim (60 chars) | Expected | Predicted | OK | Conf | Scope Recall |
|-------|-----|------------------|----------|-----------|-----|------|--------------|
| row-1 | 0 | Arthur's Magazine, an American literary periodical publis... | SUPPORTS | NOT_ENOUGH_INFO | ✗ | 0.70 | 1.00 |
| row-1 | 1 | First for Women magazine, published by Bauer Media Group,... | REFUTES | REFUTES | ✓ | 1.00 | 1.00 |
| row-1 | 2 | Arthur's Magazine had a higher circulation than First for... | NOT_ENOUGH_INFO | NOT_ENOUGH_INFO | ✓ | 0.38 | 1.00 |

## Diagnostic Trace

### Group row-1 (comparison / medium)

- **Index:** 97 nodes, 343 edges, 82 bridges, 17 communities
- **Extractor:** 28 entities, 69 claims, 21 LLM calls (753.4s)

**Claim 0:** INCONCLUSIVE→NOT_ENOUGH_INFO (exp: SUPPORTS) ✗  conf=0.70  scope_recall=1.00

- **Planner:** seeds=10 scope=24nodes/7docs lane=LANE_A_GRAPH_LCI ppr_mass=0.73 router_conf=1.00 search_strategy=COMPARISON
- **Reasoner:** docs=7 rels=1 ev=3 llm=111809ms total=111845ms
- **Confidence:** overall=0.70 ev_quality=1.00 agreement=0.00 coverage=1.00
- **Verdict answer:** Arthur's Magazine, an American literary periodical published in Philadelphia from 1844 to 1846, indeed began publication before First for Women, which started in 1989.
- **Verdict summary:** The provided structural evidence clearly indicates that Arthur's Magazine was published between 1844 and 1846, while First for Women began its publication in 1989.
- **Top seeds:**
  - claim_0036 sim=0.8679 doc=Arthur's Magazine — Arthur's Magazine was an American literary periodical.
  - claim_0035 sim=0.7945 doc=Arthur's Magazine — Arthur's Magazine was published in Philadelphia.
  - claim_0051 sim=0.7659 doc=First for Women — The magazine was started in 1989
  - claim_0003 sim=0.7458 doc=Radio City (Indian radio station) — Radio City was started in Mumbai.
  - claim_0050 sim=0.7348 doc=First for Women — First for Women is published by Bauer Media Group in the USA
  - claim_0049 sim=0.7255 doc=First for Women — First for Women is a woman's magazine
  - claim_0004 sim=0.7059 doc=Radio City (Indian radio station) — Radio City was first launched in Bengaluru.
  - claim_0042 sim=0.7033 doc=Arthur's Magazine — In May 1846, Arthur's Magazine was merged into Godey's Lady's Book.
  - claim_0038 sim=0.6724 doc=Arthur's Magazine — Edgar A. Poe contributed to Arthur's Magazine.
  - claim_0041 sim=0.6724 doc=Arthur's Magazine — Thomas G. Spear contributed to Arthur's Magazine.
- **Relationships:**
  - Arthur's Magazine → First for Women [PRECEDES] conf=0.9500
- **Evidence:**
  - doc=Arthur's Magazine — Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 1
  - doc=Arthur's Magazine — Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 1
  - doc=First for Women — The magazine was started in 1989.

**Claim 1:** PARTIALLY_REFUTED→REFUTES (exp: REFUTES) ✓  conf=1.00  scope_recall=1.00

- **Planner:** seeds=10 scope=28nodes/6docs lane=LANE_A_GRAPH_LCI ppr_mass=0.77 router_conf=0.95 search_strategy=COMPARISON
- **Reasoner:** docs=6 rels=1 ev=2 llm=134982ms total=134983ms
- **Confidence:** overall=1.00 ev_quality=1.00 agreement=1.00 coverage=1.00
- **Verdict answer:** The claim is refuted. First for Women magazine, published by Bauer Media Group, was actually started in 1989, which does not predate Arthur's Magazine (published from 1844 to 1846) by over a century.
- **Verdict summary:** The provided structural evidence and semantic facts contradict the claim that First for Women predates Arthur's Magazine by over a century, as First for Women was started in 1989.
- **Top seeds:**
  - claim_0051 sim=0.8013 doc=First for Women — The magazine was started in 1989
  - claim_0049 sim=0.7725 doc=First for Women — First for Women is a woman's magazine
  - claim_0050 sim=0.7499 doc=First for Women — First for Women is published by Bauer Media Group in the USA
  - claim_0036 sim=0.7460 doc=Arthur's Magazine — Arthur's Magazine was an American literary periodical.
  - claim_0035 sim=0.7280 doc=Arthur's Magazine — Arthur's Magazine was published in Philadelphia.
  - claim_0003 sim=0.7059 doc=Radio City (Indian radio station) — Radio City was started in Mumbai.
  - claim_0004 sim=0.6837 doc=Radio City (Indian radio station) — Radio City was first launched in Bengaluru.
  - claim_0053 sim=0.6720 doc=First for Women — In 2011 the circulation of First for Women was 1,310,696 copies
  - claim_0052 sim=0.6656 doc=First for Women — First for Women is based in Englewood Cliffs, New Jersey
  - claim_0009 sim=0.6621 doc=Radio City (Indian radio station) — PlanetRadiocity.com was launched in May 2008.
- **Relationships:**
  - First for Women → Arthur's Magazine [CONTRADICTS] conf=0.9500
- **Evidence:**
  - doc=First for Women — First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was st
  - doc=First for Women — The magazine was started in 1989.

**Claim 2:** NOT_FOUND→NOT_ENOUGH_INFO (exp: NOT_ENOUGH_INFO) ✓  conf=0.38  scope_recall=1.00

- **Planner:** seeds=10 scope=27nodes/4docs lane=LANE_A_GRAPH_LCI ppr_mass=0.81 router_conf=1.00 search_strategy=COMPARISON
- **Reasoner:** docs=4 rels=1 ev=0 llm=103431ms total=103455ms
- **Confidence:** overall=0.38 ev_quality=0.95 agreement=0.00 coverage=0.00
- **Verdict answer:** There is insufficient evidence to confirm or refute the claim that Arthur's Magazine had a higher circulation than First for Women during its peak years of publication.
- **Verdict summary:** The provided documents do not contain any information about the circulation figures of Arthur's Magazine, nor do they provide comparative data between the circulations of different magazines.
- **Top seeds:**
  - claim_0035 sim=0.7687 doc=Arthur's Magazine — Arthur's Magazine was published in Philadelphia.
  - claim_0036 sim=0.7550 doc=Arthur's Magazine — Arthur's Magazine was an American literary periodical.
  - claim_0053 sim=0.7302 doc=First for Women — In 2011 the circulation of First for Women was 1,310,696 copies
  - claim_0022 sim=0.7175 doc=Echosmith — 'Cool Kids' had over 1,200,000 sales in the US.
  - claim_0050 sim=0.6803 doc=First for Women — First for Women is published by Bauer Media Group in the USA
  - claim_0024 sim=0.6781 doc=Echosmith — 'Cool Kids' had 1.3 million downloads sold.
  - claim_0023 sim=0.6586 doc=Echosmith — 'Cool Kids' was Warner Bros. Records' fifth-biggest-selling-digital song of 2014
  - claim_0019 sim=0.6539 doc=Echosmith — 'Cool Kids' reached number 13 on Billboard Hot 100.
  - entity_0025 sim=0.6512 doc=Echosmith — 1.3 million downloads
  - claim_0020 sim=0.6427 doc=Echosmith — 'Cool Kids' was certified double platinum by RIAA.
- **Relationships:**
  - Arthur's Magazine → First for Women [UNRELATED] conf=0.9500
- **Evidence:**
  - doc= — Arthur's Magazine had a higher circulation than First for Women during its peak years of publication

