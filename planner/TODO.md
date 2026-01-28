# Planner Tasks & Implementation TODOs

## [ ] Retire Global Context Summaries (Lane B)
- [ ] Remove `GlobalContext` logic from `assemble_context` in `planner_controller.py`.
- [ ] Disable `SENSEMAKING` intent triggers that route to Lane B.
- [ ] Cleanup `scope_assembler.py` to remove community summary fetching logic.

## [/] Implement HippoRAG Node Specificity ($s_i$)
- [ ] Calculate passage frequency $|P_i|$ for all nodes.
- [ ] Update `ppr_expander.py` with the specificity-modulated teleportation vector.

## [ ] Aggregator Pattern (Lane A + C Parallelism)
- [ ] Update `PlannerOutput` to return `focused_context` and `targeted_context` for every query.