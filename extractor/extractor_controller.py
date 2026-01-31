"""
Extractor Controller
Location: extractor/extractor_controller.py
Owner: EXTRACTOR Instance

Orchestrates the LLM extraction pipeline: extract entities/claims,
glean missed items, resolve coreferences, embed nodes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from schemas.contracts import (
    ChunkExtractionDetail,
    CoreferenceGroup,
    EmbeddedNode,
    ExplorerOutput,
    ExtractedClaim,
    ExtractedEntity,
    ExtractorInput,
    ExtractorTrace,
    NodeSet,
    TextChunk,
)
from extractor.entity_extractor import extract_entities, extract_entities_batch, refine_entities_batch
from extractor.claim_extractor import extract_claims, extract_claims_batch
from extractor.gleaner import glean_missing
from extractor.coreference_resolver import resolve_coreferences
from extractor.node_embedder import embed_nodes, NodeDescriptor
from extractor.structural_analyzer import StructuralAnalyzer

logger = logging.getLogger("knowledgeaudit.extractor")


class ExtractorController:
    """v5 EXTRACTOR pipeline: extract, glean, coref, embed.

    Parameters
    ----------
    llm_client:
        Object satisfying the ``LLMClient`` protocol.  Model selection and
        temperature are the caller's responsibility when constructing this
        client; the controller does not modify the client's parameters.
    embedding_client:
        Object satisfying the ``EmbeddingClient`` protocol.  Embedding model
        selection is the caller's responsibility when constructing this client.
    config:
        Extractor-specific configuration dictionary.  See ``docs/EXTRACTOR.md``
        §9 for the full configuration reference.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_client: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm_client
        self._embed_client = embedding_client
        self._config = config or {}

        # Initialize structural analyzer for deterministic pre-processing
        structural_cfg = self._config.get("structural_analysis", {})
        use_spacy = structural_cfg.get("use_spacy", True)
        spacy_model = structural_cfg.get("spacy_model", "en_core_web_sm")
        self._structural_analyzer = StructuralAnalyzer(
            use_spacy=use_spacy,
            spacy_model=spacy_model,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _inject_structural_bridges(
        self,
        chunk: TextChunk,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedClaim]:
        """Extract structural bridge claims using deterministic analysis.

        Delegates to StructuralAnalyzer for dependency-based or heuristic-based
        structural relationship detection.

        Parameters
        ----------
        chunk:
            The text chunk being analyzed.
        entities:
            Entities extracted from this chunk.

        Returns
        -------
        List[ExtractedClaim]
            Structural bridge claims with high confidence.
        """
        return self._structural_analyzer.extract_structural_bridges(chunk, entities)

    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load schema from config (inline or file path)."""
        schema = self._config.get("schema")
        schema_path = self._config.get("schema_path")
        if schema is not None:
            return schema
        if schema_path and isinstance(schema_path, str):
            return self._load_schema_file(schema_path)
        return None

    @staticmethod
    def _load_schema_file(path: str) -> Optional[Dict[str, Any]]:
        """Load a schema from a JSON or YAML file."""
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore[import-untyped]
                return yaml.safe_load(text)
            except ImportError:
                return None
        return json.loads(text)

    def process(self, extractor_input: ExtractorInput) -> ExplorerOutput:
        """Run the extraction pipeline on pre-chunked text.

        Pipeline:
        1. Per chunk: extract entities, extract claims, glean missing
        2. Coreference resolution across all entities
        3. Build NodeDescriptors for entities and claims
        4. Embed all descriptors
        5. Build chunk_node_map
        6. Assemble NodeSet and return ExplorerOutput

        Parameters
        ----------
        extractor_input:
            Pre-chunked text with source doc IDs.

        Returns
        -------
        ExplorerOutput
            Node set with embedded nodes and chunk-node mapping.
        """
        try:
            t0 = time.monotonic()
            llm_calls = 0
            total_gleaning_iterations = 0

            all_chunks = extractor_input.chunks
            source_doc_ids = extractor_input.source_doc_ids
            schema = self._load_schema()

            # ---- Read config toggles (§9) ----
            extraction_cfg = self._config.get("extraction", {})
            phases_cfg = extraction_cfg.get("phases", {})
            phases_ner = phases_cfg.get("ner", True)
            phases_openie = phases_cfg.get("openIE", True)

            gleaning_cfg = self._config.get("gleaning", {})
            gleaning_enabled = gleaning_cfg.get("enabled", True)
            gleaning_max_iter = gleaning_cfg.get("max_iterations", 2)
            gleaning_skip_tokens = gleaning_cfg.get("skip_below_tokens", 800)

            postproc_cfg = self._config.get("postprocessing", {})
            coref_enabled = postproc_cfg.get("coreference_resolution", True)

            tracing_cfg = self._config.get("tracing", {})
            tracing_enabled = tracing_cfg.get("enabled", True)
            tracing_verbosity = tracing_cfg.get("verbosity", "full")
            # Step 1: Per-chunk extraction
            all_entities: List[ExtractedEntity] = []
            all_claims: List[ExtractedClaim] = []
            chunk_entities_map: Dict[str, List[ExtractedEntity]] = {}
            chunk_claims_map: Dict[str, List[ExtractedClaim]] = {}
            chunk_details: List[ChunkExtractionDetail] = []

            # Batch execution
            extraction_cfg = self._config.get("extraction", {})
            num_ctx = extraction_cfg.get("num_ctx", 4096)
            batch_size = 2 if getattr(self._llm, "model_path", "").startswith("mlx") else 1

            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i : i + batch_size]
                logger.info("DEBUG: Processing batch of %d chunks (start_idx=%d)", len(batch_chunks), i)
                
                try:
                    # 1. Batch Entities
                    batch_entities = []
                    if phases_ner:
                        batch_entities = extract_entities_batch(self._llm, batch_chunks, schema=schema, num_ctx=num_ctx)
                        llm_calls += 1 # 1 batch call
                    else:
                        batch_entities = [[] for _ in batch_chunks]

                    # 1b. Batch Refinement (Schema-Guided)
                    if phases_ner:
                        refined_batch = refine_entities_batch(self._llm, batch_chunks, batch_entities, num_ctx=num_ctx)
                        # Merge refined entities
                        did_refine = False
                        for k, extra in enumerate(refined_batch):
                            if extra:
                                batch_entities[k].extend(extra)
                                did_refine = True
                        if did_refine:
                            llm_calls += 1 # Count the refinement batch call

                    # 2. Batch Claims
                    batch_claims = []
                    if phases_openie:
                        batch_claims = extract_claims_batch(self._llm, batch_chunks, batch_entities, schema=schema, num_ctx=num_ctx)
                        llm_calls += 1 # 1 batch call
                    else:
                        batch_claims = [[] for _ in batch_chunks]

                    # 3. Sequential Gleaning, Structural Analysis & Assembly
                    for j, chunk in enumerate(batch_chunks):
                        entities = batch_entities[j]
                        claims = batch_claims[j]

                        gleaning_iters = 0
                        gleaning_new_ents = 0
                        gleaning_new_cls = 0

                        if gleaning_enabled:
                            gleaning = glean_missing(
                                self._llm, chunk, entities, claims,
                                max_iter=gleaning_max_iter,
                                skip_below_tokens=gleaning_skip_tokens,
                                num_ctx=num_ctx,
                            )
                            # Gleaning is still sequential calls
                            llm_calls += gleaning.iterations_performed
                            total_gleaning_iterations += gleaning.iterations_performed
                            entities = entities + gleaning.additional_entities
                            claims = claims + gleaning.additional_claims
                            gleaning_iters = gleaning.iterations_performed
                            gleaning_new_ents = len(gleaning.additional_entities)
                            gleaning_new_cls = len(gleaning.additional_claims)

                        # 3a. Structural bridge injection (per-chunk, deterministic)
                        structural_bridges = self._inject_structural_bridges(chunk, entities)
                        if structural_bridges:
                            claims = claims + structural_bridges
                            logger.debug(
                                "[%s] Structural bridges: %d",
                                chunk.chunk_id,
                                len(structural_bridges),
                            )

                        detail = ChunkExtractionDetail(
                            chunk_id=chunk.chunk_id,
                            source_doc_id=chunk.source_doc_id,
                            token_count=chunk.token_count,
                            text_preview=chunk.text[:300],
                            entities=list(entities),
                            claims=list(claims),
                            gleaning_iterations=gleaning_iters,
                            gleaning_new_entities=gleaning_new_ents,
                            gleaning_new_claims=gleaning_new_cls,
                        )

                        all_entities.extend(entities)
                        all_claims.extend(claims)
                        chunk_entities_map[chunk.chunk_id] = entities
                        chunk_claims_map[chunk.chunk_id] = claims
                        chunk_details.append(detail)

                except Exception as e:
                    logger.error("Failed to extract batch starting at %d: %s", i, e)
                    raise

            # --- Debug: Print extracted entities and claims (Summary Only) ---
            total_structural_bridges = sum(
                1 for c in all_claims if c.claim_id.endswith(tuple(f"_struct_{i:03d}" for i in range(1000)))
            )
            logger.info("=== EXTRACTOR SUMMARY ===")
            logger.info("  Entities Extracted: %d", len(all_entities))
            logger.info("  Claims Extracted:   %d", len(all_claims))
            logger.info("  Structural Bridges: %d", total_structural_bridges)
            
            if total_structural_bridges > 0:
                 # Only show bridge claims as they are critical for debugging graph connectivity
                for c in all_claims:
                    if "_struct_" in c.claim_id:
                        logger.debug("  BRIDGE: %s", c.claim_text)

            # Step 2: Coreference resolution
            if coref_enabled:
                coref_groups = resolve_coreferences(self._llm, all_entities)
                # Only count an LLM call if resolve_coreferences actually
                # called the LLM (it short-circuits for <= 3 unique forms).
                unique_forms = set(e.surface_form for e in all_entities)
                if len(unique_forms) > 3:
                    llm_calls += 1
            else:
                # Generate singleton groups when coref is disabled
                seen_forms: Dict[str, None] = {}
                for e in all_entities:
                    seen_forms.setdefault(e.surface_form, None)
                coref_groups = [
                    CoreferenceGroup(canonical_form=form, variants=[form])
                    for form in seen_forms
                ]

            # ENHANCEMENT (H4): A cross-document comparative claim generation
            # step could be inserted here.  After coreference resolution groups
            # entities, a second pass could compare entities of the same type
            # across documents (e.g., dates, monetary amounts, obligation
            # scopes) and generate synthetic comparative claims such as
            # "Document A's liability cap exceeds Document B's."
            # See docs/investigations/EXTRACTOR_INVESTIGATION.md §H4.

            # Step 3: Build NodeDescriptors
            descriptors: List[NodeDescriptor] = []

            # Entity descriptors — one per coreference group
            for i, group in enumerate(coref_groups):
                # Aggregate source chunks and best description for this group
                group_source_chunks: List[str] = []
                group_source_doc_ids: set = set()
                group_description: Optional[str] = None
                for entity in all_entities:
                    if entity.surface_form in group.variants:
                        if entity.chunk_id not in group_source_chunks:
                            group_source_chunks.append(entity.chunk_id)
                        # Find source doc from chunks
                        for ch in all_chunks:
                            if ch.chunk_id == entity.chunk_id:
                                group_source_doc_ids.add(ch.source_doc_id)
                        # Keep the longest description across variants
                        if entity.description and (
                            group_description is None
                            or len(entity.description) > len(group_description)
                        ):
                            group_description = entity.description

                doc_id = next(iter(group_source_doc_ids)) if group_source_doc_ids else ""
                meta: Dict[str, Any] = {
                    "variants": group.variants,
                    "entity_index": i,
                }
                if group_description:
                    meta["description"] = group_description
                descriptors.append(
                    NodeDescriptor(
                        node_id=f"entity_{i:04d}",
                        node_type="ENTITY",
                        canonical_text=group.canonical_form,
                        source_doc_id=doc_id,
                        source_chunks=group_source_chunks,
                        metadata=meta,
                    )
                )

            # Claim descriptors — one per unique claim text
            unique_claims: Dict[str, ExtractedClaim] = {}
            for claim in all_claims:
                key = claim.claim_text.strip().lower()
                if key not in unique_claims:
                    unique_claims[key] = claim
                else:
                    # Merge chunk_id into the existing unique claim if not already there
                    # Note: NodeDescriptor takes source_chunks as a list.
                    # We'll handle the source_chunks merging below by referencing all_claims if needed,
                    # but for now let's just keep the first instance and its chunk.
                    pass
            
            dedup_claims = list(unique_claims.values())

            entity_offset = len(coref_groups)
            for j, claim in enumerate(dedup_claims):
                # Find source doc
                claim_doc_id = ""
                for ch in all_chunks:
                    if ch.chunk_id == claim.chunk_id:
                        claim_doc_id = ch.source_doc_id
                        break

                descriptors.append(
                    NodeDescriptor(
                        node_id=f"claim_{j:04d}",
                        node_type="CLAIM",
                        canonical_text=claim.claim_text,
                        source_doc_id=claim_doc_id,
                        source_chunks=[claim.chunk_id],
                        metadata={
                            "claim_id": claim.claim_id,
                            "subject": claim.subject_entity,
                            "object": claim.object_entity,
                            "predicate": claim.predicate,
                            "confidence": claim.confidence,
                        },
                    )
                )

            # Step 4: Embed all descriptors
            embedded_nodes = embed_nodes(descriptors, self._embed_client)

            # Step 5: Build chunk_node_map
            chunk_node_map: Dict[str, List[str]] = {}
            for node in embedded_nodes:
                for chunk_id in node.source_chunks:
                    if chunk_id not in chunk_node_map:
                        chunk_node_map[chunk_id] = []
                    chunk_node_map[chunk_id].append(node.node_id)

            # Step 6: Assemble NodeSet and return
            avg_claims = (
                len(all_claims) / len(all_chunks) if all_chunks else 0.0
            )

            node_set = NodeSet(
                nodes=embedded_nodes,
                source_doc_ids=list(source_doc_ids),
                chunk_count=len(all_chunks),
                entity_count=len(coref_groups),
                claim_count=len(all_claims),
                extraction_config=self._config,
                coreference_groups=coref_groups,
                avg_claims_per_chunk=avg_claims,
            )

            elapsed = time.monotonic() - t0

            # Build trace respecting tracing config (§9)
            if tracing_enabled:
                trace_chunk_details: Optional[List[ChunkExtractionDetail]] = None
                if tracing_verbosity == "full":
                    trace_chunk_details = chunk_details
                # "summary" verbosity omits chunk_details

                trace: Optional[ExtractorTrace] = ExtractorTrace(
                    trace_id=str(uuid.uuid4()),
                    input_chunk_count=len(all_chunks),
                    entities_extracted=len(coref_groups),
                    claims_extracted=len(all_claims),
                    gleaning_iterations=total_gleaning_iterations,
                    llm_calls=llm_calls,
                    extraction_time_seconds=round(elapsed, 4),
                    chunk_details=trace_chunk_details,
                )
            else:
                trace = None

            return ExplorerOutput(
                node_set=node_set,
                chunk_node_map=chunk_node_map,
                trace=trace,
            )

        except Exception as e:
            # Return minimal valid output with error info and partial trace
            logger.critical("EXTRACTOR PIPELINE CRITICAL FAILURE: %s", str(e), exc_info=True)
            elapsed = time.monotonic() - t0
            error_trace = ExtractorTrace(
                trace_id=str(uuid.uuid4()),
                input_chunk_count=len(extractor_input.chunks),
                entities_extracted=0,
                claims_extracted=0,
                gleaning_iterations=total_gleaning_iterations,
                llm_calls=llm_calls,
                extraction_time_seconds=round(elapsed, 4),
            )
            empty_node_set = NodeSet(
                nodes=[],
                source_doc_ids=list(extractor_input.source_doc_ids),
                chunk_count=0,
                entity_count=0,
                claim_count=0,
                extraction_config={"error": str(e)},
            )
            return ExplorerOutput(node_set=empty_node_set, trace=error_trace)
