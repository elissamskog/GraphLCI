"""
Extractor v5 Pipeline Tests
Location: extractor/test_extractor_v5.py
Owner: EXTRACTOR Instance

Unit tests for the v5 EXTRACTOR pipeline modules using mock clients.
"""

from __future__ import annotations

import json
import sys
import os
import unittest

# Ensure project root is on path so schemas can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.contracts import (
    CoreferenceGroup,
    EmbeddedNode,
    ExtractedClaim,
    ExtractedEntity,
    ExtractorInput,
    TextChunk,
)
from extractor.entity_extractor import extract_entities
from extractor.claim_extractor import extract_claims
from extractor.gleaner import glean_missing, GleaningResult
from extractor.coreference_resolver import resolve_coreferences
from extractor.node_embedder import embed_nodes, NodeDescriptor
from extractor.extractor_controller import ExtractorController


# ============================================================================
# MOCK CLIENTS
# ============================================================================

class MockLLMClient:
    """Routes responses by substring matching on prompt."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return '{"error": "no mock response"}'


class MockEmbeddingClient:
    """Produces deterministic 384-dim vectors."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0

    def embed(self, texts: list) -> list:
        self.call_count += 1
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list:
        import hashlib, struct
        digest = hashlib.sha256(text.encode()).digest()
        floats = []
        for i in range(self.dimension):
            block = hashlib.md5(digest + struct.pack(">I", i)).digest()
            raw = struct.unpack(">I", block[:4])[0]
            floats.append((raw / 0xFFFFFFFF) * 2 - 1)
        mag = sum(f * f for f in floats) ** 0.5
        if mag > 0:
            floats = [f / mag for f in floats]
        return floats


# ============================================================================
# SHARED MOCK RESPONSES
# ============================================================================

ENTITY_RESPONSE = json.dumps([
    {"surface_form": "ACME Corp", "entity_type": "ORGANIZATION",
     "description": "The company required to comply with the Widget Act"},
    {"surface_form": "Widget Act", "entity_type": "ARTIFACT",
     "description": "Legislation imposing compliance requirements and penalties"},
    {"surface_form": "$5 million", "entity_type": "QUANTITY",
     "description": "The monetary penalty amount under the Widget Act"},
])

CLAIM_RESPONSE = json.dumps([
    {
        "claim": "ACME Corp is subject to the Widget Act.",
        "subject": "ACME Corp",
        "object": "Widget Act",
        "predicate": "is subject to",
        "description": "ACME Corp must comply with the Widget Act.",
        "confidence": 0.9,
    },
    {
        "claim": "The penalty is $5 million.",
        "subject": "Widget Act",
        "object": "$5 million",
        "predicate": "imposes penalty of",
        "description": "Widget Act imposes a penalty of $5 million.",
        "confidence": 0.85,
    },
])

GLEANING_COMPLETE_RESPONSE = "EXTRACTION COMPLETE"

GLEANING_FINDS_RESPONSE = json.dumps({
    "entities": [
        {"surface_form": "Board of Directors", "entity_type": "ORGANIZATION",
         "description": "Governance body overseeing ACME Corp"},
    ],
    "claims": [
        {
            "claim": "Board of Directors oversees ACME Corp.",
            "subject": "Board of Directors",
            "object": "ACME Corp",
            "predicate": "oversees",
            "description": "The Board of Directors provides oversight.",
            "confidence": 0.7,
        },
    ],
})

COREF_RESPONSE = json.dumps([
    {"canonical": "ACME Corporation", "variants": ["ACME Corp", "ACME Corporation", "Acme"]},
    {"canonical": "Widget Act", "variants": ["Widget Act"]},
    {"canonical": "$5 million", "variants": ["$5 million"]},
    {"canonical": "Board of Directors", "variants": ["Board of Directors", "Board"]},
])


# ============================================================================
# TEST CASES
# ============================================================================

class TestExtractEntities(unittest.TestCase):
    """Entity extraction via mock LLM."""

    def test_extract_entities(self):
        llm = MockLLMClient({"Extract all named entities": ENTITY_RESPONSE})
        chunk = TextChunk(
            chunk_id="doc_001_chunk_0000",
            source_doc_id="doc_001",
            text="ACME Corp must comply with the Widget Act. The penalty is $5 million.",
            token_count=14,
            position=0,
        )
        entities = extract_entities(llm, chunk)
        self.assertEqual(len(entities), 3)

        surface_forms = [e.surface_form for e in entities]
        self.assertIn("ACME Corp", surface_forms)
        self.assertIn("Widget Act", surface_forms)
        self.assertIn("$5 million", surface_forms)

        # Verify descriptions are populated
        for e in entities:
            self.assertIsNotNone(e.description, f"{e.surface_form} should have a description")
            self.assertIsInstance(e.description, str)

        # Verify char offsets are found
        for e in entities:
            if e.surface_form in chunk.text:
                self.assertIsNotNone(e.char_offset)
                self.assertGreaterEqual(e.char_offset, 0)


class TestExtractClaims(unittest.TestCase):
    """Claim extraction via mock LLM."""

    def test_extract_claims(self):
        llm = MockLLMClient({"Extract factual claims": CLAIM_RESPONSE})
        chunk = TextChunk(
            chunk_id="doc_001_chunk_0000",
            source_doc_id="doc_001",
            text="ACME Corp must comply with the Widget Act. The penalty is $5 million.",
            token_count=14,
            position=0,
        )
        entities = [
            ExtractedEntity(surface_form="ACME Corp", entity_type="ORG", chunk_id=chunk.chunk_id),
            ExtractedEntity(surface_form="Widget Act", entity_type="ARTIFACT", chunk_id=chunk.chunk_id),
        ]
        claims = extract_claims(llm, chunk, entities)
        self.assertEqual(len(claims), 2)

        self.assertEqual(claims[0].claim_id, "doc_001_chunk_0000_claim_000")
        self.assertEqual(claims[0].subject_entity, "ACME Corp")
        self.assertEqual(claims[0].object_entity, "Widget Act")
        self.assertAlmostEqual(claims[0].confidence, 0.9)

        self.assertEqual(claims[1].claim_id, "doc_001_chunk_0000_claim_001")
        self.assertAlmostEqual(claims[1].confidence, 0.85)


class TestGleanComplete(unittest.TestCase):
    """Gleaning returns complete immediately."""

    def test_glean_complete(self):
        llm = MockLLMClient({"Examine the text again": GLEANING_COMPLETE_RESPONSE})
        chunk = TextChunk(
            chunk_id="doc_001_chunk_0000",
            source_doc_id="doc_001",
            text=" ".join(["word"] * 1000),  # 1000 tokens, above 800 threshold
            token_count=1000,
            position=0,
        )
        entities = [ExtractedEntity(surface_form="X", entity_type="T", chunk_id=chunk.chunk_id)]
        claims = [ExtractedClaim(
            claim_id="c1", claim_text="X is Y", subject_entity="X",
            object_entity="Y", predicate="is", description="test",
            chunk_id=chunk.chunk_id, confidence=0.9,
        )]

        result = glean_missing(llm, chunk, entities, claims)
        self.assertEqual(len(result.additional_entities), 0)
        self.assertEqual(len(result.additional_claims), 0)
        self.assertEqual(result.confidence_complete, 1.0)
        self.assertEqual(result.iterations_performed, 1)


class TestGleanFindsItems(unittest.TestCase):
    """First gleaning pass finds items, second completes."""

    def test_glean_finds_items(self):
        call_count = [0]

        class TwoPassLLM:
            def generate(self, prompt):
                call_count[0] += 1
                if call_count[0] == 1:
                    return GLEANING_FINDS_RESPONSE
                return GLEANING_COMPLETE_RESPONSE

        llm = TwoPassLLM()
        chunk = TextChunk(
            chunk_id="doc_001_chunk_0000",
            source_doc_id="doc_001",
            text=" ".join(["word"] * 1000),
            token_count=1000,
            position=0,
        )
        entities = [ExtractedEntity(surface_form="X", entity_type="T", chunk_id=chunk.chunk_id)]
        claims = []

        result = glean_missing(llm, chunk, entities, claims)
        self.assertGreater(len(result.additional_entities), 0)
        self.assertGreater(len(result.additional_claims), 0)
        self.assertEqual(result.iterations_performed, 2)
        self.assertEqual(result.confidence_complete, 1.0)


class TestGleanSkipSmallChunk(unittest.TestCase):
    """Small chunks (< 800 tokens) are skipped."""

    def test_glean_skip_small_chunk(self):
        llm = MockLLMClient({})
        chunk = TextChunk(
            chunk_id="doc_001_chunk_0000",
            source_doc_id="doc_001",
            text="A short chunk.",
            token_count=3,
            position=0,
        )
        result = glean_missing(llm, chunk, [], [])
        self.assertEqual(result.iterations_performed, 0)
        self.assertEqual(result.confidence_complete, 1.0)
        self.assertEqual(llm.call_count, 0)


class TestResolveCoreferences(unittest.TestCase):
    """Coreference resolution via mock LLM."""

    def test_resolve_coreferences(self):
        llm = MockLLMClient({"Identify which entities": COREF_RESPONSE})
        entities = [
            ExtractedEntity(surface_form="ACME Corp", entity_type="ORG", chunk_id="c1"),
            ExtractedEntity(surface_form="Widget Act", entity_type="ART", chunk_id="c1"),
            ExtractedEntity(surface_form="$5 million", entity_type="QTY", chunk_id="c1"),
            ExtractedEntity(surface_form="Board of Directors", entity_type="ORG", chunk_id="c2"),
        ]
        groups = resolve_coreferences(llm, entities)
        self.assertGreater(len(groups), 0)

        # Find the ACME group
        acme_group = None
        for g in groups:
            if "ACME" in g.canonical_form:
                acme_group = g
                break
        self.assertIsNotNone(acme_group)
        self.assertIn("ACME Corp", acme_group.variants)

    def test_singleton_groups_for_few_entities(self):
        """<= 3 unique forms -> singleton groups, no LLM call."""
        llm = MockLLMClient({})
        entities = [
            ExtractedEntity(surface_form="A", entity_type="T", chunk_id="c1"),
            ExtractedEntity(surface_form="B", entity_type="T", chunk_id="c1"),
        ]
        groups = resolve_coreferences(llm, entities)
        self.assertEqual(len(groups), 2)
        self.assertEqual(llm.call_count, 0)


class TestEmbedNodes(unittest.TestCase):
    """Node embedding with MockEmbeddingClient."""

    def test_embed_nodes(self):
        client = MockEmbeddingClient(dimension=384)
        descriptors = [
            NodeDescriptor(
                node_id="entity_0000",
                node_type="ENTITY",
                canonical_text="ACME Corporation",
                source_doc_id="doc_001",
                source_chunks=["doc_001_chunk_0000"],
            ),
            NodeDescriptor(
                node_id="claim_0000",
                node_type="CLAIM",
                canonical_text="ACME Corp is subject to the Widget Act.",
                source_doc_id="doc_001",
                source_chunks=["doc_001_chunk_0000"],
            ),
        ]
        nodes = embed_nodes(descriptors, client)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(client.call_count, 1)

        for node in nodes:
            self.assertEqual(len(node.embedding), 384)
            self.assertIsInstance(node.embedding[0], float)

        self.assertEqual(nodes[0].node_id, "entity_0000")
        self.assertEqual(nodes[0].node_type, "ENTITY")
        self.assertEqual(nodes[1].node_id, "claim_0000")

    def test_embed_empty(self):
        client = MockEmbeddingClient()
        nodes = embed_nodes([], client)
        self.assertEqual(len(nodes), 0)
        self.assertEqual(client.call_count, 0)


class TestExtractorPipeline(unittest.TestCase):
    """End-to-end ExtractorController pipeline with pre-built chunks."""

    def test_extractor_pipeline(self):
        mock_responses = {
            "Extract all named entities": ENTITY_RESPONSE,
            "Extract factual claims": CLAIM_RESPONSE,
            "Examine the text again": GLEANING_COMPLETE_RESPONSE,
            "Identify which entities": COREF_RESPONSE,
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)

        controller = ExtractorController(llm, embed_client)

        # Pre-built chunks (as if from ChunkerController)
        chunks = [
            TextChunk(
                chunk_id="doc_001_chunk_0000",
                source_doc_id="doc_001",
                text="ACME Corp must comply with the Widget Act. The penalty is $5 million.",
                token_count=14,
                position=0,
            ),
            TextChunk(
                chunk_id="doc_002_chunk_0000",
                source_doc_id="doc_002",
                text="Board of Directors oversees ACME Corp operations and compliance.",
                token_count=9,
                position=0,
            ),
        ]

        extractor_input = ExtractorInput(
            chunks=chunks,
            source_doc_ids=["doc_001", "doc_002"],
        )
        output = controller.process(extractor_input)

        # Validate output structure
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.node_set)
        self.assertGreater(len(output.node_set.nodes), 0)
        self.assertIn("doc_001", output.node_set.source_doc_ids)
        self.assertIn("doc_002", output.node_set.source_doc_ids)
        self.assertGreater(output.node_set.chunk_count, 0)
        self.assertGreater(output.node_set.entity_count, 0)
        self.assertGreater(output.node_set.claim_count, 0)

        # Validate chunk_node_map
        self.assertIsNotNone(output.chunk_node_map)
        self.assertGreater(len(output.chunk_node_map), 0)

        # Validate embedded nodes have 384-dim vectors
        for node in output.node_set.nodes:
            self.assertEqual(len(node.embedding), 384)

        # Validate coreference groups present
        self.assertIsNotNone(output.node_set.coreference_groups)
        self.assertGreater(len(output.node_set.coreference_groups), 0)

        # Validate entity descriptions flow through to node metadata
        entity_nodes = [n for n in output.node_set.nodes if n.node_type == "ENTITY"]
        self.assertGreater(len(entity_nodes), 0)
        has_description = any(
            n.metadata.get("description") for n in entity_nodes
        )
        self.assertTrue(has_description, "At least one entity node should have a description in metadata")

        # Validate chunk_details carry entity descriptions
        self.assertIsNotNone(output.trace)
        self.assertIsNotNone(output.trace.chunk_details)
        for detail in output.trace.chunk_details:
            for entity in detail.entities:
                if entity.description is not None:
                    self.assertIsInstance(entity.description, str)

        # Validate LLM and embedding client were called
        self.assertGreater(llm.call_count, 0)
        self.assertGreater(embed_client.call_count, 0)


# ============================================================================
# NULL-FIELD ROBUSTNESS TESTS (reproduce NoneType.strip() crash)
# ============================================================================

class TestEntityExtractorNullFields(unittest.TestCase):
    """LLM may return explicit null values for fields — .strip() must not crash."""

    def test_entity_type_null(self):
        """entity_type: null should not crash, falls back to 'UNKNOWN'."""
        response = json.dumps([
            {"surface_form": "ACME Corp", "entity_type": None, "description": "A company"},
        ])
        llm = MockLLMClient({"Extract all named entities": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="ACME Corp does things.", token_count=5, position=0,
        )
        entities = extract_entities(llm, chunk)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_type, "UNKNOWN")

    def test_surface_form_null(self):
        """surface_form: null should be skipped, not crash."""
        response = json.dumps([
            {"surface_form": None, "entity_type": "ORG", "description": "A company"},
            {"surface_form": "Widget", "entity_type": "ART", "description": "A widget"},
        ])
        llm = MockLLMClient({"Extract all named entities": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Widget is important.", token_count=4, position=0,
        )
        entities = extract_entities(llm, chunk)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].surface_form, "Widget")

    def test_description_null(self):
        """description: null should not crash."""
        response = json.dumps([
            {"surface_form": "ACME", "entity_type": "ORG", "description": None},
        ])
        llm = MockLLMClient({"Extract all named entities": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="ACME exists.", token_count=3, position=0,
        )
        entities = extract_entities(llm, chunk)
        self.assertEqual(len(entities), 1)
        self.assertIsNone(entities[0].description)

    def test_all_fields_null(self):
        """All fields null — item should be skipped entirely."""
        response = json.dumps([
            {"surface_form": None, "entity_type": None, "description": None},
        ])
        llm = MockLLMClient({"Extract all named entities": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Nothing here.", token_count=2, position=0,
        )
        entities = extract_entities(llm, chunk)
        self.assertEqual(len(entities), 0)


class TestClaimExtractorNullFields(unittest.TestCase):
    """LLM may return explicit null values in claim fields."""

    def test_claim_fields_null(self):
        """Null object/predicate/description should not crash."""
        response = json.dumps([
            {
                "claim": "ACME must comply.",
                "subject": "ACME",
                "object": None,
                "predicate": None,
                "description": None,
                "confidence": 0.8,
            },
        ])
        llm = MockLLMClient({"Extract factual claims": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="ACME must comply.", token_count=4, position=0,
        )
        entities = [ExtractedEntity(surface_form="ACME", entity_type="ORG", chunk_id="c1")]
        claims = extract_claims(llm, chunk, entities)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].object_entity, "")
        self.assertEqual(claims[0].predicate, "")
        self.assertEqual(claims[0].description, "")

    def test_subject_null_skips_claim(self):
        """subject: null should skip the claim, not crash."""
        response = json.dumps([
            {
                "claim": "Something happens.",
                "subject": None,
                "object": "Y",
                "predicate": "does",
                "description": "test",
                "confidence": 0.5,
            },
        ])
        llm = MockLLMClient({"Extract factual claims": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Something happens.", token_count=3, position=0,
        )
        claims = extract_claims(llm, chunk, [])
        self.assertEqual(len(claims), 0)

    def test_claim_text_null_skips(self):
        """claim: null should skip the claim, not crash."""
        response = json.dumps([
            {
                "claim": None,
                "subject": "X",
                "object": "Y",
                "predicate": "does",
                "description": "test",
                "confidence": 0.5,
            },
        ])
        llm = MockLLMClient({"Extract factual claims": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Some text.", token_count=2, position=0,
        )
        claims = extract_claims(llm, chunk, [])
        self.assertEqual(len(claims), 0)


class TestGleanerNullFields(unittest.TestCase):
    """Gleaner must handle null fields from LLM responses."""

    def test_gleaned_entity_null_fields(self):
        """Null entity_type and description should not crash gleaner."""
        response = json.dumps({
            "entities": [
                {"surface_form": "NewEnt", "entity_type": None, "description": None},
            ],
            "claims": [],
        })
        llm = MockLLMClient({"Examine the text again": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text=" ".join(["word"] * 1000), token_count=1000, position=0,
        )
        result = glean_missing(llm, chunk, [], [], max_iter=1)
        self.assertEqual(len(result.additional_entities), 1)
        self.assertEqual(result.additional_entities[0].entity_type, "UNKNOWN")
        self.assertIsNone(result.additional_entities[0].description)

    def test_gleaned_claim_null_fields(self):
        """Null claim fields should not crash gleaner."""
        response = json.dumps({
            "entities": [],
            "claims": [
                {
                    "claim": "X does Y",
                    "subject": "X",
                    "object": None,
                    "predicate": None,
                    "description": None,
                    "confidence": None,
                },
            ],
        })
        llm = MockLLMClient({"Examine the text again": response})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text=" ".join(["word"] * 1000), token_count=1000, position=0,
        )
        result = glean_missing(llm, chunk, [], [], max_iter=1)
        self.assertEqual(len(result.additional_claims), 1)
        self.assertEqual(result.additional_claims[0].object_entity, "")
        self.assertEqual(result.additional_claims[0].predicate, "")
        self.assertEqual(result.additional_claims[0].description, "")


class TestCorefResolverNullFields(unittest.TestCase):
    """Coreference resolver must handle null fields."""

    def test_canonical_null_skipped(self):
        """canonical: null should skip the group, not crash."""
        response = json.dumps([
            {"canonical": None, "variants": ["A", "B"]},
            {"canonical": "Good", "variants": ["Good", "G"]},
        ])
        llm = MockLLMClient({"Identify which entities": response})
        entities = [
            ExtractedEntity(surface_form="A", entity_type="T", chunk_id="c1"),
            ExtractedEntity(surface_form="B", entity_type="T", chunk_id="c1"),
            ExtractedEntity(surface_form="Good", entity_type="T", chunk_id="c1"),
            ExtractedEntity(surface_form="G", entity_type="T", chunk_id="c1"),
        ]
        groups = resolve_coreferences(llm, entities)
        # The null-canonical group should be skipped; "Good" group parsed;
        # "A" and "B" appear as fallback singletons
        canonical_forms = [g.canonical_form for g in groups]
        self.assertIn("Good", canonical_forms)
        self.assertIn("A", canonical_forms)
        self.assertIn("B", canonical_forms)


class TestPipelineNullFieldsEndToEnd(unittest.TestCase):
    """End-to-end test: LLM returns null fields, pipeline must not crash."""

    def test_pipeline_survives_null_fields(self):
        """Pipeline produces valid output even when LLM returns null fields."""
        entity_resp = json.dumps([
            {"surface_form": "ACME", "entity_type": None, "description": None},
            {"surface_form": None, "entity_type": "ORG", "description": "skip me"},
            {"surface_form": "Widget", "entity_type": "ART", "description": "A widget"},
        ])
        claim_resp = json.dumps([
            {
                "claim": "ACME uses Widget",
                "subject": "ACME",
                "object": None,
                "predicate": None,
                "description": None,
                "confidence": None,
            },
            {
                "claim": None,
                "subject": "Widget",
                "object": "X",
                "predicate": "is",
                "description": "skip",
                "confidence": 0.5,
            },
        ])
        mock_responses = {
            "Extract all named entities": entity_resp,
            "Extract factual claims": claim_resp,
            "Examine the text again": "EXTRACTION COMPLETE",
            "Identify which entities": json.dumps([
                {"canonical": "ACME", "variants": ["ACME"]},
                {"canonical": "Widget", "variants": ["Widget"]},
            ]),
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)
        controller = ExtractorController(llm, embed_client)

        chunks = [
            TextChunk(
                chunk_id="c1", source_doc_id="d1",
                text="ACME uses Widget for things.",
                token_count=6, position=0,
            ),
        ]
        inp = ExtractorInput(chunks=chunks, source_doc_ids=["d1"])
        output = controller.process(inp)

        # Should not crash and should produce valid output
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.node_set)
        # 2 valid entities (null surface_form skipped)
        self.assertEqual(output.node_set.entity_count, 2)
        # 1 valid claim (null claim text skipped)
        self.assertEqual(output.node_set.claim_count, 1)
        # Nodes were embedded
        self.assertGreater(len(output.node_set.nodes), 0)


# ============================================================================
# CONFIG TOGGLE TESTS
# ============================================================================

class TestNERToggle(unittest.TestCase):
    """NER phase can be disabled via config."""

    def test_ner_disabled(self):
        """When ner=false, no entities are extracted but claims still run."""
        mock_responses = {
            "Extract all named entities": ENTITY_RESPONSE,
            "Extract factual claims": CLAIM_RESPONSE,
            "Examine the text again": GLEANING_COMPLETE_RESPONSE,
            "Identify which entities": COREF_RESPONSE,
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)

        config = {"extraction": {"phases": {"ner": False}}}
        controller = ExtractorController(llm, embed_client, config=config)

        chunks = [
            TextChunk(
                chunk_id="doc_001_chunk_0000",
                source_doc_id="doc_001",
                text="ACME Corp must comply with the Widget Act.",
                token_count=8,
                position=0,
            ),
        ]
        inp = ExtractorInput(chunks=chunks, source_doc_ids=["doc_001"])
        output = controller.process(inp)

        # No entity nodes (NER was disabled)
        self.assertEqual(output.node_set.entity_count, 0)
        # Claims should still be extracted
        self.assertGreater(output.node_set.claim_count, 0)


class TestOpenIEToggle(unittest.TestCase):
    """OpenIE phase can be disabled via config."""

    def test_openie_disabled(self):
        """When openIE=false, no claims are extracted but entities still run."""
        mock_responses = {
            "Extract all named entities": ENTITY_RESPONSE,
            "Extract factual claims": CLAIM_RESPONSE,
            "Examine the text again": GLEANING_COMPLETE_RESPONSE,
            "Identify which entities": COREF_RESPONSE,
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)

        config = {"extraction": {"phases": {"openIE": False}}}
        controller = ExtractorController(llm, embed_client, config=config)

        chunks = [
            TextChunk(
                chunk_id="doc_001_chunk_0000",
                source_doc_id="doc_001",
                text="ACME Corp must comply with the Widget Act. The penalty is $5 million.",
                token_count=14,
                position=0,
            ),
        ]
        inp = ExtractorInput(chunks=chunks, source_doc_ids=["doc_001"])
        output = controller.process(inp)

        # No claims extracted
        self.assertEqual(output.node_set.claim_count, 0)
        # Entities should still be extracted
        self.assertGreater(output.node_set.entity_count, 0)


class TestGleaningToggle(unittest.TestCase):
    """Gleaning can be disabled via config."""

    def test_gleaning_disabled(self):
        """When gleaning.enabled=false, zero gleaning LLM calls."""
        entity_resp = json.dumps([
            {"surface_form": "X", "entity_type": "T", "description": "An entity"},
        ])
        claim_resp = json.dumps([
            {"claim": "X does Y", "subject": "X", "object": "Y",
             "predicate": "does", "description": "test", "confidence": 0.9},
        ])
        mock_responses = {
            "Extract all named entities": entity_resp,
            "Extract factual claims": claim_resp,
            "Examine the text again": "SHOULD NOT BE CALLED",
            "Identify which entities": json.dumps([
                {"canonical": "X", "variants": ["X"]},
            ]),
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)

        config = {"gleaning": {"enabled": False}}
        controller = ExtractorController(llm, embed_client, config=config)

        chunks = [
            TextChunk(
                chunk_id="c1", source_doc_id="d1",
                text=" ".join(["word"] * 1000),
                token_count=1000, position=0,
            ),
        ]
        inp = ExtractorInput(chunks=chunks, source_doc_ids=["d1"])
        output = controller.process(inp)

        # Trace should show zero gleaning iterations
        self.assertIsNotNone(output.trace)
        self.assertEqual(output.trace.gleaning_iterations, 0)
        # LLM calls = 1 (NER) + 1 (OpenIE) = 2; no gleaning, no coref (<=3 forms)
        self.assertEqual(output.trace.llm_calls, 2)


class TestCorefToggle(unittest.TestCase):
    """Coreference resolution can be disabled via config."""

    def test_coref_disabled_produces_singletons(self):
        """When coreference_resolution=false, each unique entity form is a singleton group."""
        mock_responses = {
            "Extract all named entities": ENTITY_RESPONSE,
            "Extract factual claims": CLAIM_RESPONSE,
            "Examine the text again": GLEANING_COMPLETE_RESPONSE,
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)

        config = {"postprocessing": {"coreference_resolution": False}}
        controller = ExtractorController(llm, embed_client, config=config)

        chunks = [
            TextChunk(
                chunk_id="doc_001_chunk_0000",
                source_doc_id="doc_001",
                text="ACME Corp must comply with the Widget Act. The penalty is $5 million.",
                token_count=14,
                position=0,
            ),
        ]
        inp = ExtractorInput(chunks=chunks, source_doc_ids=["doc_001"])
        output = controller.process(inp)

        # Each coreference group should be a singleton
        for group in output.node_set.coreference_groups:
            self.assertEqual(len(group.variants), 1)
            self.assertEqual(group.canonical_form, group.variants[0])


class TestGleaningParams(unittest.TestCase):
    """Gleaning parameters are forwarded from config."""

    def test_custom_skip_below_tokens(self):
        """Custom skip_below_tokens from config is respected."""
        llm = MockLLMClient({})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Short text",
            token_count=500,  # Above default 800? No — below 800
            position=0,
        )
        # Default threshold 800 — chunk with 500 tokens is skipped
        result = glean_missing(llm, chunk, [], [], skip_below_tokens=800)
        self.assertEqual(result.iterations_performed, 0)
        self.assertEqual(llm.call_count, 0)

        # Lower threshold to 100 — chunk with 500 tokens now runs gleaning
        llm2 = MockLLMClient({"Examine the text again": GLEANING_COMPLETE_RESPONSE})
        result2 = glean_missing(llm2, chunk, [], [], skip_below_tokens=100)
        self.assertGreater(result2.iterations_performed, 0)
        self.assertEqual(llm2.call_count, 1)

    def test_custom_max_iterations(self):
        """Custom max_iterations limits the number of gleaning passes."""
        call_count = [0]

        class AlwaysFindsLLM:
            def generate(self, prompt):
                call_count[0] += 1
                return json.dumps({
                    "entities": [{"surface_form": f"E{call_count[0]}",
                                  "entity_type": "T",
                                  "description": "found entity"}],
                    "claims": [],
                })

        llm = AlwaysFindsLLM()
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text=" ".join(["word"] * 1000),
            token_count=1000, position=0,
        )
        result = glean_missing(llm, chunk, [], [], max_iter=3)
        self.assertEqual(result.iterations_performed, 3)
        self.assertEqual(call_count[0], 3)


class TestTracingConfig(unittest.TestCase):
    """Tracing config controls trace output."""

    def _make_controller(self, config):
        mock_responses = {
            "Extract all named entities": ENTITY_RESPONSE,
            "Extract factual claims": CLAIM_RESPONSE,
            "Examine the text again": GLEANING_COMPLETE_RESPONSE,
            "Identify which entities": COREF_RESPONSE,
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)
        return ExtractorController(llm, embed_client, config=config)

    def _make_input(self):
        chunks = [
            TextChunk(
                chunk_id="doc_001_chunk_0000",
                source_doc_id="doc_001",
                text="ACME Corp must comply with the Widget Act. The penalty is $5 million.",
                token_count=14,
                position=0,
            ),
        ]
        return ExtractorInput(chunks=chunks, source_doc_ids=["doc_001"])

    def test_tracing_disabled(self):
        """When tracing.enabled=false, trace is None."""
        controller = self._make_controller({"tracing": {"enabled": False}})
        output = controller.process(self._make_input())
        self.assertIsNone(output.trace)

    def test_tracing_summary_verbosity(self):
        """When verbosity='summary', chunk_details is omitted."""
        controller = self._make_controller(
            {"tracing": {"enabled": True, "verbosity": "summary"}}
        )
        output = controller.process(self._make_input())
        self.assertIsNotNone(output.trace)
        self.assertIsNone(output.trace.chunk_details)

    def test_tracing_full_verbosity(self):
        """When verbosity='full', chunk_details is included."""
        controller = self._make_controller(
            {"tracing": {"enabled": True, "verbosity": "full"}}
        )
        output = controller.process(self._make_input())
        self.assertIsNotNone(output.trace)
        self.assertIsNotNone(output.trace.chunk_details)
        self.assertGreater(len(output.trace.chunk_details), 0)


class TestSchemaFormat(unittest.TestCase):
    """Structured entity_types schema is formatted in prompts."""

    def test_entity_extractor_structured_schema(self):
        """entity_extractor formats entity_types list with indicators."""
        from extractor.entity_extractor import _build_prompt

        schema = {
            "entity_types": [
                {"name": "Clause", "description": "A provision in a contract",
                 "indicators": ["Section", "Article"]},
                {"name": "Party", "description": "A person or organization"},
            ]
        }
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Test text", token_count=2, position=0,
        )
        prompt = _build_prompt(chunk, schema)
        self.assertIn("Clause", prompt)
        self.assertIn("A provision in a contract", prompt)
        self.assertIn("indicators: Section, Article", prompt)
        self.assertIn("Party", prompt)
        self.assertIn("A person or organization", prompt)

    def test_entity_extractor_flat_schema_fallback(self):
        """Flat {k: v} schema still works for backward compatibility."""
        from extractor.entity_extractor import _build_prompt

        schema = {"PERSON": "A named individual", "ORG": "An organization"}
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Test text", token_count=2, position=0,
        )
        prompt = _build_prompt(chunk, schema)
        self.assertIn("PERSON", prompt)
        self.assertIn("A named individual", prompt)

    def test_claim_extractor_structured_schema(self):
        """claim_extractor formats entity_types list."""
        from extractor.claim_extractor import _build_prompt

        schema = {
            "entity_types": [
                {"name": "Clause", "description": "A provision"},
                {"name": "Obligation", "description": "A requirement"},
            ]
        }
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text="Test text", token_count=2, position=0,
        )
        entities = [
            ExtractedEntity(surface_form="X", entity_type="T", chunk_id="c1"),
        ]
        prompt = _build_prompt(chunk, entities, schema)
        self.assertIn("Clause", prompt)
        self.assertIn("Obligation", prompt)


class TestLLMCallCount(unittest.TestCase):
    """LLM call count is accurate when coref short-circuits."""

    def test_no_coref_llm_call_for_few_entities(self):
        """When <= 3 unique forms, coref doesn't count an LLM call."""
        entity_resp = json.dumps([
            {"surface_form": "A", "entity_type": "T", "description": "Entity A"},
            {"surface_form": "B", "entity_type": "T", "description": "Entity B"},
        ])
        claim_resp = json.dumps([
            {"claim": "A is B", "subject": "A", "object": "B",
             "predicate": "is", "description": "test", "confidence": 0.9},
        ])
        mock_responses = {
            "Extract all named entities": entity_resp,
            "Extract factual claims": claim_resp,
            "Examine the text again": GLEANING_COMPLETE_RESPONSE,
        }
        llm = MockLLMClient(mock_responses)
        embed_client = MockEmbeddingClient(dimension=384)

        controller = ExtractorController(llm, embed_client)

        chunks = [
            TextChunk(
                chunk_id="c1", source_doc_id="d1",
                text=" ".join(["word"] * 1000),
                token_count=1000, position=0,
            ),
        ]
        inp = ExtractorInput(chunks=chunks, source_doc_ids=["d1"])
        output = controller.process(inp)

        # 1 NER + 1 OpenIE + 1 gleaning (complete) + 0 coref (<=3 forms) = 3
        self.assertEqual(output.trace.llm_calls, 3)


class TestGleanedEntityDescription(unittest.TestCase):
    """Gleaned entities include description field."""

    def test_gleaned_entity_has_description(self):
        """Entities from gleaning include the description field."""
        gleaning_resp = json.dumps({
            "entities": [
                {"surface_form": "NewEntity", "entity_type": "T",
                 "description": "A newly discovered entity"},
            ],
            "claims": [],
        })
        llm = MockLLMClient({"Examine the text again": gleaning_resp})
        chunk = TextChunk(
            chunk_id="c1", source_doc_id="d1",
            text=" ".join(["word"] * 1000),
            token_count=1000, position=0,
        )
        result = glean_missing(llm, chunk, [], [], max_iter=1)
        self.assertEqual(len(result.additional_entities), 1)
        self.assertEqual(result.additional_entities[0].description, "A newly discovered entity")


class TestSchemaPathConfig(unittest.TestCase):
    """Controller supports schema_path config key."""

    def test_schema_path_loads_json(self):
        """schema_path pointing to a JSON file loads the schema."""
        import tempfile

        schema = {"entity_types": [{"name": "Test", "description": "A test type"}]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(schema, f)
            f.flush()
            path = f.name

        try:
            controller = ExtractorController(
                MockLLMClient({}), MockEmbeddingClient(),
                config={"schema_path": path}
            )
            loaded = controller._load_schema()
            self.assertIsNotNone(loaded)
            self.assertIn("entity_types", loaded)
            self.assertEqual(loaded["entity_types"][0]["name"], "Test")
        finally:
            os.unlink(path)

    def test_inline_schema_takes_precedence(self):
        """Inline 'schema' key takes precedence over 'schema_path'."""
        inline = {"entity_types": [{"name": "Inline", "description": "Inline schema"}]}
        controller = ExtractorController(
            MockLLMClient({}), MockEmbeddingClient(),
            config={"schema": inline, "schema_path": "/nonexistent/path.json"}
        )
        loaded = controller._load_schema()
        self.assertEqual(loaded["entity_types"][0]["name"], "Inline")


if __name__ == "__main__":
    unittest.main()
