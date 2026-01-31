import unittest
from typing import List, Dict, Any
from schemas.contracts import (
    GraphEdge, GraphNode, GraphState, NodeType, DomainType, EdgeType,
    RouterDecision, SearchStrategyClassification, SearchStrategyType,
    LaneType, IdentifiedTargets, ScopeObject, ScopedDocument,
    SupportingDocument, DependencyType, RawDocument, ScopeRequest,
    DocumentRole
)
from librarian.resolver.ppr_traversal import PPRTraversal, PPRInput
from librarian.resolver.scope_resolver import ScopeResolver, ResolverInput
from librarian.assembler.context_builder import ContextBuilder, ContextBuildInput

class MockLLM:
    def generate(self, prompt: str) -> str:
        # Minimal mock response that looks like the interpretation JSON
        return """
{
  "relevant_edges": [],
  "supporting_documents": [],
  "domain_pair": {
    "primary_domain": "LEGAL",
    "secondary_domain": null,
    "cross_domain_logic": null
  },
  "completeness_confidence": 1.0,
  "warnings": []
}
"""

class TestNoiseReduction(unittest.TestCase):
    def setUp(self):
        # Setup a simple graph: A -> B -> C
        # A is the seed. Relevance should flow to B (forward) and a bit to C.
        # Flow from B back to D (unrelated) should be dampened.
        self.nodes = [
            GraphNode("A", "doc_A", NodeType.DOCUMENT, DomainType.LEGAL, "Seed Document", []),
            GraphNode("B", "doc_B", NodeType.DOCUMENT, DomainType.LEGAL, "Forward Target", []),
            GraphNode("C", "doc_C", NodeType.DOCUMENT, DomainType.LEGAL, "Grandchild Target", []),
            GraphNode("D", "doc_D", NodeType.DOCUMENT, DomainType.LEGAL, "Noisy Ancestor", []),
        ]
        self.edges = [
            GraphEdge("e1", "A", "B", EdgeType.AMENDS, 1.0, "A amends B"),
            GraphEdge("e2", "B", "C", EdgeType.GOVERNS, 1.0, "B governs C"),
            GraphEdge("e3", "D", "A", EdgeType.DEFINES_TERM, 1.0, "D defines term in A"), # D -> A
        ]
        self.graph_state = GraphState(self.nodes, self.edges)

    def test_asymmetric_weighting(self):
        # Test 1: Symmetric (factors = 1.0)
        trav_sym = PPRTraversal()
        inp_sym = PPRInput(
            seed_nodes=["A"],
            graph_state=self.graph_state,
            forward_weight_factor=1.0,
            backward_weight_factor=1.0,
            min_score_threshold=0.0
        )
        out_sym = trav_sym.traverse(inp_sym)
        scores_sym = dict(out_sym.ranked_nodes)
        
        # Test 2: Asymmetric (backward = 0.5)
        trav_asym = PPRTraversal()
        inp_asym = PPRInput(
            seed_nodes=["A"],
            graph_state=self.graph_state,
            forward_weight_factor=1.0,
            backward_weight_factor=0.1, # Stronger effect for test
            min_score_threshold=0.0
        )
        out_asym = trav_asym.traverse(inp_asym)
        scores_asym = dict(out_asym.ranked_nodes)

        print(f"\nSymmetric scores: {scores_sym}")
        print(f"Asymmetric scores: {scores_asym}")

        # In asymmetric case, D should have significantly lower score than B
        # because the path A -> D is backward, whereas A -> B is forward.
        self.assertTrue(scores_asym["B"] > scores_asym["D"])
        
        # The ratio of D's score relative to B's should be smaller in asymmetric case
        ratio_sym = scores_sym["D"] / scores_sym["B"]
        ratio_asym = scores_asym["D"] / scores_asym["B"]
        self.assertTrue(ratio_asym < ratio_sym)

    def test_top_k_ceiling(self):
        # Setup documents with different scores
        primary = [
            ScopedDocument("A", 0.9, DocumentRole.SOURCE),
            ScopedDocument("B", 0.8, DocumentRole.TARGET),
        ]
        supporting = [
            SupportingDocument("C", DependencyType.REFERENCE, None, 0.7),
            SupportingDocument("D", DependencyType.REFERENCE, None, 0.6),
            SupportingDocument("E", DependencyType.REFERENCE, None, 0.5),
        ]
        
        from schemas.contracts import EdgeContext, EdgeType
        edge_context = [
            EdgeContext("ec1", EdgeType.AMENDS, "A", "B", "", None),
            EdgeContext("ec2", EdgeType.AMENDS, "B", "C", "", None),
            EdgeContext("ec3", EdgeType.AMENDS, "C", "D", "", None),
            EdgeContext("ec4", EdgeType.AMENDS, "D", "E", "", None),
        ]

        scope = ScopeObject(
            query_id="q1",
            primary_documents=primary,
            supporting_documents=supporting,
            edge_context=edge_context,
            domain_pair=None,
            scope_metadata=None,
            warnings=[]
        )
        
        raw_docs = [
            RawDocument(id, "Content of " + id, {}) 
            for id in ["A", "B", "C", "D", "E"]
        ]
        
        # Mock decontextualized content
        decontext = {d.doc_id: d.content for d in raw_docs}
        
        # 1. Without Top-K (all should be included if budget allows)
        cb_no_k = ContextBuilder({"max_top_k": None, "max_graph_hops": 10})
        inp = ContextBuildInput(scope, raw_docs, decontext, 1000)
        out_no_k = cb_no_k.build(inp)
        self.assertEqual(len(out_no_k.lci_payload.documents), 5)
        
        # 2. With Top-K = 3
        cb_k3 = ContextBuilder({"max_top_k": 3, "max_graph_hops": 10})
        out_k3 = cb_k3.build(inp)
        self.assertEqual(len(out_k3.lci_payload.documents), 3)
        
        # Verify they are the top 3 by score
        doc_ids = [d.doc_id for d in out_k3.lci_payload.documents]
        self.assertIn("A", doc_ids)
        self.assertIn("B", doc_ids)
        self.assertIn("C", doc_ids)
        self.assertNotIn("D", doc_ids)
        self.assertNotIn("E", doc_ids)

if __name__ == "__main__":
    unittest.main()
