# Xctopus: Geometric Architecture & Atomic Knowledge (KN)

This document describes the core architectural shift of Xctopus from a passive data repository to a dynamic, geometric knowledge organism.

## 1. Core Concept: Beyond RAG to Vector Geometry
Traditional RAG (Retrieval-Augmented Generation) indexes text for search. Xctopus, instead, implements **Semantic Anchors**—physical-like objects in a high-dimensional vector space.

*   **Knowledge Atoms (KN)**: These are the primary units of knowledge (JSON files) that represent consolidated "truths".
*   **Geometric Properties**:
    *   **Mass (Gravity)**: Calculated from the number of variants. Larger nodes have a stronger "gravitational pull" on incoming evidence.
    *   **Centroid**: The mathematical center of the node's influence.
    *   **Variance**: Defines the "radius" or "fuzziness" of the knowledge.
    *   **Rigor Inertia**: High-trust nodes (GOLD/SILVER) have structural resistance to change, preventing "new data noise" from corrupting established facts.

## 2. The Learning Mechanism: Non-Linear Evolution
Xctopus does **not** use traditional backpropagation or training epochs. Instead, intelligence emerges from **topological recalculations**:

### A. Repulsion & Contrast
When a new node (e.g., TP53) is introduced near an existing one (e.g., BRCA1), the system recalculates their boundaries. They "push" against each other, forcing each node to become more specific (e.g., BRCA1 moves away from generic "Tumor Suppressor" terms to focus on gene-specific domains like "RING").

### B. Centroid Displacement
Learning occurs when new evidence shifts the mathematical center of a node. 
*   **Confirmation**: Increases density and gravity.
*   **Contradiction**: Expands variance or triggers node splitting (creation of sub-nodes).

### C. Bayesian Refinement
The Orchestrator uses Bayesian filters to handle new data without a global "re-train":
*   **Consistence**: Reinforces the current KN.
*   **Conflict**: Generates alerts and increases variance, protecting the GOLD standard.
*   **Novelty**: Creates "Hypothesis Buffers" (Candidates).

## 3. Knowledge Activation (Friction & Synthesis)
Knowledge in Xctopus is activated by **Friction**—the collision of static Knowledge Atoms with dynamic incoming Evidence.

*   **In-Context Synthesis**: The LLM performs real-time synthesis between the JSON atómico (the "book") and the new evidence (the "question"). It doesn't use "stale" weights; it thinks via the Attention mechanism.
*   **Semantic Symbiosis**: The LLM is anchored to a node (it knows *where* it is), but it requires evidence to provide an interpretation and fill in technical blanks.

## 4. Phase Management
*   **Phase 1: Anatomy (Seed Phase)**: Implantation of "Memory Anchors" (Bootstrap). Static, surgical, and robust.
*   **Phase 2: Physiology (Dynamic Phase)**: Learning via displacement, conflict resolution, and continual evidence ingestion. This is where the "Movement" and the clinical interpretation happen.

---
*Note: This architecture allows for Hospital-Grade stability while maintaining the flexibility of Continual Learning.*
