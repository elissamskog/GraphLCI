
import json
import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

class GraphDebugger:
    def __init__(self, checkpoint_path: str, trace_path: Optional[str] = None):
        print(f"[*] Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            self.checkpoint = json.load(f)
        
        self.trace = None
        if trace_path and os.path.exists(trace_path):
            print(f"[*] Loading trace: {trace_path}")
            with open(trace_path, "r") as f:
                self.trace = json.load(f)
        
        self.G = self._build_networkx_graph()
        self.auditor = GraphAuditor(self.G, self.trace)
        print(f"[*] Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def _build_networkx_graph(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        graph_data = self.checkpoint.get("graph", {})
        nodes_data = graph_data.get("nodes", {})
        edges_data = graph_data.get("edges", {})
        
        for nid, ninfo in nodes_data.items():
            G.add_node(nid, **ninfo)
            
        edge_list = edges_data.values() if isinstance(edges_data, dict) else edges_data
        for edge in edge_list:
            G.add_edge(
                edge["source"], 
                edge["target"], 
                edge_id=edge["edge_id"],
                edge_type=edge["edge_type"],
                is_cross_document=edge.get("is_cross_document", False),
                weight=edge.get("weight", 1.0)
            )
        return G

    # --- Section 1: Topological Metric Audit ---

    def run_topological_audit(self):
        print("\n" + "="*50)
        print("TOPOLOGICAL METRIC AUDIT")
        print("="*50)
        
        # 1. Degree Distribution
        degrees = [d for n, d in self.G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        print(f"[*] Average Degree: {avg_degree:.2f}")
        
        # Hub Detection
        sorted_degrees = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        print("\n[*] Top 10 Hubs (Semantic Sinks?):")
        for nid, d in sorted_degrees[:10]:
            node = self.G.nodes[nid]
            print(f"  - {nid} (deg={d}): {node.get('canonical_text')} [{node.get('node_type')}]")
            
        # 2. Connected Components (treating as undirected for connectivity)
        undir_G = self.G.to_undirected()
        components = list(nx.connected_components(undir_G))
        print(f"\n[*] Connected Components: {len(components)}")
        if len(components) > 1:
            print("  [!] WARNING: Fragmented graph detected. Isolation islands present.")
            comp_sizes = [len(c) for c in components]
            print(f"  [!] Component sizes: {sorted(comp_sizes, reverse=True)[:5]}...")

        # 3. Bridge Edge Ratio
        total_edges = self.G.number_of_edges()
        cross_doc_edges = sum(1 for _, _, data in self.G.edges(data=True) if data.get("is_cross_document"))
        bridge_ratio = cross_doc_edges / total_edges if total_edges > 0 else 0
        print(f"\n[*] Bridge Edge Ratio (is_cross_document): {bridge_ratio:.4f}")
        if bridge_ratio < 0.10:
            print("  [!] WARNING: Low cross-document connectivity. Multi-hop traversal may fail.")

        # Degree Distribution Plot
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), alpha=0.7, color='steelblue', edgecolor='black')
        plt.title("Node Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Node Count")
        plt.grid(axis='y', alpha=0.3)
        plot_path = "output/temp/degree_distribution.png"
        plt.savefig(plot_path)
        print(f"\n[*] Degree distribution plot saved to: {plot_path}")

    # --- Section 2: Interactive Subgraph Visualization ---

    def visualize_subgraph(self, center_node_id: str, radius: int = 2, output_html: str = "output/temp/subgraph.html"):
        print(f"\n[*] Generating {radius}-hop ego-graph for: {center_node_id}")
        
        if center_node_id not in self.G:
            # Try to find by canonical text
            found = False
            for nid, data in self.G.nodes(data=True):
                if data.get("canonical_text") == center_node_id:
                    center_node_id = nid
                    found = True
                    break
            if not found:
                print(f"  [!] Node '{center_node_id}' not found in graph.")
                return

        # Create ego graph
        undir_G = self.G.to_undirected()
        ego_nodes = nx.single_source_shortest_path_length(undir_G, center_node_id, cutoff=radius).keys()
        sub_G = self.G.subgraph(ego_nodes)
        
        self._save_pyvis_html(sub_G, output_html, title=f"{radius}-hop Subgraph: {center_node_id}")

    def export_dashboard_data(self, output_json: str = "output/dashboard_data.json"):
        print(f"[*] Exporting dashboard data to: {output_json}")
        
        nodes = []
        for nid, data in self.G.nodes(data=True):
            nodes.append({
                "id": nid,
                "label": data.get("canonical_text", nid),
                "type": data.get("node_type", "UNKNOWN"),
                "doc": data.get("source_doc_id", "Unknown"),
                "description": data.get("metadata", {}).get("description", "")
            })
            
        links = []
        for u, v, data in self.G.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "type": data.get("edge_type", "unknown"),
                "is_cross_document": data.get("is_cross_document", False),
                "weight": data.get("weight", 1.0)
            })
            
        # Extract claim-specific PPR scores and metadata from trace
        claims_data = {}
        if self.trace:
            for group in self.trace.get("groups", []):
                for ct in group.get("claim_traces", []):
                    cid = ct.get("claim_id")
                    ppr_map = {}
                    planner_trace = ct.get("planner_trace", {})
                    for sd in planner_trace.get("scope_node_details", []):
                        ppr_map[sd["node_id"]] = sd["ppr_score"]
                    
                    claims_data[cid] = {
                        "text": ct.get("claim_text"),
                        "ppr_scores": ppr_map,
                        "verdict_predicted": ct.get("predicted_mapped"),
                        "verdict_expected": ct.get("expected"),
                        "correct": ct.get("correct"),
                        "lane": ct.get("lane"),
                        "reasoning": ct.get("reasoner_trace", {}).get("reasoning_trace", "")
                    }

        dashboard_payload = {
            "nodes": nodes,
            "links": links,
            "claims": claims_data,
            "metadata": {
                "timestamp": self.checkpoint.get("timestamp"),
                "graph_stats": {
                    "nodes": self.G.number_of_nodes(),
                    "edges": self.G.number_of_edges()
                }
            }
        }
        
        with open(output_json, "w") as f:
            json.dump(dashboard_payload, f, indent=2)
        print(f"[*] Dashboard data exported successfully.")

    def _save_pyvis_html(self, sub_G: nx.Graph, output_path: str, title: str, ppr_scores: Optional[Dict[str, float]] = None):
        if not PYVIS_AVAILABLE:
            print("  [!] PyVis not installed. Skipping visualization.")
            return

        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=True)
        net.force_atlas_2based()
        
        for nid, data in sub_G.nodes(data=True):
            ntype = data.get("node_type", "UNKNOWN")
            label = data.get("canonical_text", nid)
            doc_id = data.get("source_doc_id", "Unknown")
            
            # Color coding
            color = "#007bff" if ntype == "ENTITY" else "#28a745" # Blue for Entity, Green for Claim
            if ppr_scores and nid in ppr_scores:
                # Gradient based on mass? Just highlight for now
                mass = ppr_scores[nid]
                size = 20 + (mass * 500) # Scale by mass
            else:
                size = 20
                
            hover_text = f"ID: {nid}\nType: {ntype}\nDoc: {doc_id}\nText: {label}"
            if ppr_scores and nid in ppr_scores:
                hover_text += f"\nPPR Score: {ppr_scores[nid]:.6f}"
                
            net.add_node(nid, label=label, title=hover_text, color=color, size=size)
            
        for u, v, data in sub_G.edges(data=True):
            etype = data.get("edge_type", "unknown")
            is_cross = data.get("is_cross_document", False)
            color = "#ffc107" if is_cross else "#6c757d" # Yellow for Cross-Doc, Grey for Internal
            
            net.add_edge(u, v, title=f"Type: {etype}\nCross-Doc: {is_cross}", color=color)
            
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          }
        }
        """)
        
        net.save_graph(output_path)
        print(f"[*] Visualization saved to: {output_path}")

    # --- Section 3: Path Activation Tracing ---

    def trace_ppr_activation(self, claim_id: str, threshold: float = 0.005, output_html: str = "output/temp/ppr_trace.html"):
        if not self.trace:
            print("  [!] No trace file loaded. Path activation tracing requires trace JSON.")
            return
            
        print(f"\n[*] Tracing PPR activation for claim: {claim_id}")
        
        # Find the claim trace
        claim_data = None
        for group in self.trace.get("groups", []):
            for ct in group.get("claim_traces", []):
                if ct.get("claim_id") == claim_id:
                    claim_data = ct
                    break
            if claim_data: break
            
        if not claim_data:
            print(f"  [!] Claim ID '{claim_id}' not found in trace.")
            return
            
        # Get PPR scores from scope_node_details
        ppr_scores = {}
        planner_trace = claim_data.get("planner_trace", {})
        for sd in planner_trace.get("scope_node_details", []):
            ppr_scores[sd["node_id"]] = sd["ppr_score"]
            
        # Filter nodes by threshold
        active_nodes = [nid for nid, score in ppr_scores.items() if score >= threshold]
        
        if not active_nodes:
            print(f"  [!] No nodes found above PPR threshold {threshold}.")
            return
            
        print(f"[*] Found {len(active_nodes)} active nodes above threshold {threshold}")
        
        # Build subgraph of active nodes + their immediate context edges
        sub_G = self.G.subgraph(active_nodes)
        
        self._save_pyvis_html(sub_G, output_html, title=f"PPR Activation Trace: {claim_id}", ppr_scores=ppr_scores)

class GraphAuditor:
    def __init__(self, G: nx.MultiDiGraph, trace: Optional[Dict] = None):
        self.G = G
        self.trace = trace

    # --- Section 1: Global Entropy & Distribution Metrics ---

    def calculate_global_metrics(self) -> Dict[str, Any]:
        """Calculates structural balance and relationship diversity."""
        # 1. Relationship Entropy
        edge_types = [data.get("edge_type", "unknown") for _, _, data in self.G.edges(data=True)]
        counts = Counter(edge_types)
        total = sum(counts.values())
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values() if count > 0) # Avoid log(0)
        
        # 2. Source-to-Node Ratio
        nodes_by_doc = Counter([data.get("doc", data.get("source_doc_id", "unknown")) for _, data in self.G.nodes(data=True)])
        doc_count = len(nodes_by_doc)
        node_density = {doc: count for doc, count in nodes_by_doc.items()}
        avg_density = np.mean(list(node_density.values())) if node_density else 0
        
        # 3. Average Clustering Coefficient
        undir_G = self.G.to_undirected()
        # MultiGraph clustering requires flattening or manual handling in NX
        simple_undir = nx.Graph(undir_G)
        avg_clustering = nx.average_clustering(simple_undir)

        return {
            "relationship_entropy": entropy,
            "predicate_distribution": dict(counts),
            "avg_fact_density": avg_density,
            "fact_density_per_doc": node_density,
            "avg_clustering_coefficient": avg_clustering
        }

    # --- Section 2: Node-Specific Saliency Audit ---

    def describe_node(self, node_id_or_text: str, claim_id: Optional[str] = None) -> Dict[str, Any]:
        """Outputs detailed metrics for a specific node for agent consumption."""
        nid = node_id_or_text
        if nid not in self.G:
            # Try lookup by text
            for n, data in self.G.nodes(data=True):
                if data.get("canonical_text") == nid:
                    nid = n
                    break
        
        if nid not in self.G:
            return {"error": f"Node {node_id_or_text} not found."}

        node_data = self.G.nodes[nid]
        
        # In-degree / Out-degree
        in_deg = self.G.in_degree(nid)
        out_deg = self.G.out_degree(nid)
        
        # PPR Specificity Index
        ppr_specificity = None
        if self.trace and claim_id:
            ppr_scores = self._get_ppr_scores(claim_id)
            if ppr_scores:
                scores = list(ppr_scores.values())
                if len(scores) > 1: # Need at least 2 scores for std dev
                    mean_ppr = np.mean(scores)
                    std_ppr = np.std(scores)
                    node_ppr = ppr_scores.get(nid, 0)
                    ppr_specificity = (node_ppr - mean_ppr) / std_ppr if std_ppr > 0 else 0
                elif len(scores) == 1: # Only one score, cannot calculate std dev, but can show relative
                    node_ppr = ppr_scores.get(nid, 0)
                    ppr_specificity = node_ppr # If only one, it's the "most specific"
                else:
                    ppr_specificity = 0 # No scores
            else:
                ppr_specificity = 0 # No scores
        
        # Co-occurrence neighborhood
        neighbors = []
        for v in self.G.successors(nid):
            # Aggregate weights across multi-edges
            weight = sum(d.get("weight", 1.0) for d in self.G.get_edge_data(nid, v).values())
            neighbors.append((v, self.G.nodes[v].get("canonical_text"), weight))
        sorted_neighbors = sorted(neighbors, key=lambda x: x[2], reverse=True)[:5]

        # Document Lineage
        lineage = set([node_data.get("source_doc_id") or node_data.get("doc")])
        for u, _, data in self.G.in_edges(nid, data=True):
            if data.get("doc_id"): lineage.add(data.get("doc_id"))

        return {
            "node_id": nid,
            "label": node_data.get("canonical_text"),
            "type": node_data.get("node_type"),
            "degrees": {"in": in_deg, "out": out_deg},
            "is_sink": in_deg > out_deg * 2 if out_deg > 0 else in_deg > 5,
            "ppr_specificity_zscore": ppr_specificity,
            "top_neighbors": sorted_neighbors,
            "document_lineage": list(filter(None, lineage))
        }

    # --- Section 3: Essential CLI Debugging Utilities ---

    def get_top_hubs(self, n: int = 10) -> List[Dict]:
        """Sink Detector: identifies top nodes by degree and PageRank."""
        # Simple PageRank over directed graph
        try:
            pr = nx.pagerank(self.G, weight="weight")
        except Exception as e:
            print(f"Warning: Could not calculate PageRank: {e}. Defaulting to 0 for all nodes.")
            pr = {node: 0 for node in self.G.nodes()} # Fallback if graph is weirdly disjoint for simple PR

        hubs = []
        for nid, deg in sorted(self.G.degree(), key=lambda x: x[1], reverse=True):
            hubs.append({
                "id": nid,
                "label": self.G.nodes[nid].get("canonical_text"),
                "degree": deg,
                "pagerank": pr.get(nid, 0)
            })
        return hubs[:n]

    def check_connectivity(self, source_text: str, target_text: str) -> Dict[str, Any]:
        """Bridge Verifier: finds shortest path between two semantic entities."""
        def find_nid(text):
            for n, data in self.G.nodes(data=True):
                if data.get("canonical_text") == text: return n
            return None

        s_id = find_nid(source_text)
        t_id = find_nid(target_text)

        if not s_id or not t_id:
            return {"error": "Source or target entity text not found."}

        try:
            path = nx.shortest_path(self.G, s_id, t_id)
            path_labels = [self.G.nodes[nid].get("canonical_text") for nid in path]
            return {"path": path_labels, "hops": len(path) - 1}
        except nx.NetworkXNoPath:
            return {"error": "No Path Found", "suggestion": "Check synonymy thresholds or bridge edges."}
        except Exception as e:
            return {"error": f"An unexpected error occurred during path finding: {e}"}

    def sim_walk_decay(self, alpha: float, path_length: int) -> float:
        """Damping Scrubber: calculates reachability probability P = alpha^L."""
        # User formula P=alpha^L for simulation of reachability
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha (damping factor) must be between 0 and 1.")
        if path_length < 0:
            raise ValueError("Path length must be non-negative.")
        prob = alpha ** path_length
        return prob

    def _get_ppr_scores(self, claim_id: str) -> Dict[str, float]:
        if not self.trace: return {}
        for group in self.trace.get("groups", []):
            for ct in group.get("claim_traces", []):
                if ct.get("claim_id") == claim_id:
                    return {sd["node_id"]: sd["ppr_score"] for sd in ct.get("planner_trace", {}).get("scope_node_details", [])}
        return {}

def main():
    parser = argparse.ArgumentParser(description="v5 Graph Debugger")
    parser.add_argument("--checkpoint", type=str, default="output/multihop_librarian_checkpoint.json", help="Path to librarian checkpoint")
    parser.add_argument("--trace", type=str, default="output/multihop_output.trace.json", help="Path to evaluation trace JSON")
    parser.add_argument("--audit", action="store_true", help="Run topological metric audit")
    parser.add_argument("--viz-node", type=str, help="Generate ego-graph for a specific node ID or canonical text")
    parser.add_argument("--radius", type=int, default=2, help="Radius for ego-graph")
    parser.add_argument("--trace-claim", type=str, help="Trace PPR activation for a specific Claim ID from the trace JSON")
    parser.add_argument("--threshold", type=float, default=0.005, help="PPR mass threshold for activation tracing")
    parser.add_argument("--export-dashboard", action="store_true", help="Export flattened JSON for JS-based dashboard")
    parser.add_argument("--dashboard-file", type=str, default="output/dashboard_data.json", help="Path for dashboard JSON")
    
    # New auditor arguments
    parser.add_argument("--global-metrics", action="store_true", help="Calculate and print global graph metrics (entropy, clustering, etc.)")
    parser.add_argument("--describe-node", type=str, help="Get detailed description for a specific node (by ID or canonical text)")
    parser.add_argument("--describe-node-claim", type=str, help="Optional claim ID for --describe-node to calculate PPR specificity")
    parser.add_argument("--check-path", nargs=2, metavar=('SOURCE_TEXT', 'TARGET_TEXT'), help="Check for shortest path between two nodes by canonical text")
    parser.add_argument("--sim-decay", nargs=2, type=float, metavar=('ALPHA', 'LENGTH'), help="Simulate walk decay: P = ALPHA^LENGTH")

    args = parser.parse_args()
    
    debugger = GraphDebugger(args.checkpoint, args.trace)
    
    if args.audit:
        debugger.run_topological_audit()
        
    if args.viz_node:
        debugger.visualize_subgraph(args.viz_node, radius=args.radius)
        
    if args.trace_claim:
        debugger.trace_ppr_activation(args.trace_claim, threshold=args.threshold)

    if args.export_dashboard:
        debugger.export_dashboard_data(args.dashboard_file)

    if args.global_metrics:
        print("\n" + "="*50)
        print("GLOBAL GRAPH METRICS")
        print("="*50)
        metrics = debugger.auditor.calculate_global_metrics()
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"[*] {k}: (omitted dict, {len(v)} keys)")
            else:
                print(f"[*] {k}: {v:.4f}" if isinstance(v, float) else f"[*] {k}: {v}")
        
    if args.describe_node:
        print("\n" + "="*50)
        print(f"NODE SALIENCY AUDIT: {args.describe_node}")
        print("="*50)
        res = debugger.auditor.describe_node(args.describe_node, claim_id=args.describe_node_claim)
        if "error" in res:
            print(f"[!] {res['error']}")
        else:
            for k, v in res.items():
                print(f"[*] {k}: {v}")

    if args.check_path:
        print("\n" + "="*50)
        print(f"BRIDGE VERIFIER: {args.check_path[0]} -> {args.check_path[1]}")
        print("="*50)
        res = debugger.auditor.check_connectivity(args.check_path[0], args.check_path[1])
        if "error" in res:
            print(f"[!] {res['error']}")
            if "suggestion" in res: print(f"[?] Suggestion: {res['suggestion']}")
        else:
            print(f"[*] Hops: {res['hops']}")
            print(f"[*] Path: {' -> '.join(res['path'])}")

    if args.sim_decay:
        alpha, length = args.sim_decay
        prob = debugger.auditor.sim_walk_decay(alpha, int(length))
        print(f"\n[*] Damping Scrubber: Reachability P({int(length)} hops, {alpha} alpha) = {prob:.6f}")
        
    if not (args.audit or args.viz_node or args.trace_claim or args.export_dashboard or 
            args.global_metrics or args.describe_node or args.check_path or args.sim_decay):
        parser.print_help()

if __name__ == "__main__":
    main()
