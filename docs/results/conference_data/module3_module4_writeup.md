# LightGAP — Modules 3 & 4: Diagnostic Assessment & Dynamic Graph Remediation

This document presents the implementation architecture, illustrative assessment items, and dynamic runtime graph mutation mechanics of **Module 3 (Diagnostic Quiz Engine)** and **Module 4 (Adaptive Memory Decay & Runtime Graph Remediation)**.

---

## 1. Methodological & Evaluation Scope Statement

> [!IMPORTANT]
> **Scoping Discipline**: While Modules 1 and 2 underwent rigorous statistical benchmarking against 5-fold cross-validation and an immutable held-out gold set (`gold_pairs_v1`, 105 pairs) with strict schema-level anti-leakage guards, **Modules 3 and 4 are presented strictly as implemented, validated software systems with concrete illustrative walkthroughs**, rather than claims of measured human behavioral outcomes.
> 
> The core theoretical contribution of LightGAP is the algorithmic discovery of prerequisite dependency DAGs and budget-constrained optimal learning paths (ROC-AUC 0.9327, precision 0.8421, 90.0% reversal rejection, and mathematical ILP guarantees). Modules 3 and 4 demonstrate that the discovered graph is actionable in a closed-loop interactive learning environment:
> 1. Nodes serve as diagnostic probe points equipped with misconception-tagged distractors (Module 3).
> 2. Learner errors dynamically mutate the active topological graph at runtime to inject remediation detours and enforce spaced retention without requiring full pipeline recomputation (Module 4).
> 
> Both modules are fully implemented in the LightGAP backend, exposed via authenticated FastAPI endpoints (`/quiz` and `/remediation`), and verified by 115 passing unit and integration tests (including `test_quiz_engine.py`, `test_decay_model.py`, and `test_graph_rewriter.py`). They are not, however, benchmarked on human cohort learning gains, which remains reserved for longitudinal educational field studies.

---

## 2. Module 3: Diagnostic Assessment with Misconception-Tagged Distractors

Module 3 attaches targeted diagnostic questions to concept nodes in the induced DAG. Unlike generic multiple-choice items, each question is constructed with a 3-role distractor schema designed to isolate *where* a learner's conceptual understanding fails:
- **Option A (Correct):** Confirms mastery of the target concept and allows forward progression.
- **Option B (Prerequisite Distractor):** Explicitly maps to a known upstream prerequisite failure (misconception vector $M_k$). Selecting Option B triggers Module 4's graph rewriter to reroute the learner to remediation.
- **Option C (Conceptual Distractor):** Captures within-concept confusion or category errors (e.g., conflating distinct paradigms or confusing correlation with causation), allowing localized remedial explanation without graph mutation.

Below are three live quiz items generated across our experimental corpora:

### 2.1 Domain 1: Data Mining (Worked Benchmark Domain)

* **Target Concept Node:** `Data mining`
* **Stem:** 
  > *"A company has a database containing 10 million customer transaction records. They want to identify hidden patterns, such as which products are frequently purchased together, to optimize inventory. Which of the following best describes the primary technical approach they should employ?"*
* **Option A (Correct):** 
  > *"Applying algorithms from machine learning and statistics to scan the massive dataset for non-trivial patterns and correlations."*
* **Option B (Prerequisite Distractor):** 
  > *"Writing a simple SQL query to retrieve all transactions where the total amount exceeds $100, as this directly filters the relevant data."*
* **Option C (Conceptual Distractor):** 
  > *"Manually reviewing a random sample of 100 transactions to guess the purchasing trends, as this is the most accurate way to understand customer behavior."*
* **Misconception Tag:** `Confusing data retrieval with pattern discovery`
* **Diagnostic Function:** Option B directly targets the failure to recognize that relational database query filtering is a prerequisite foundational data retrieval step, not predictive pattern discovery. Selecting Option B signals that the learner lacks the prerequisite concept of exploratory data analysis vs. standard CRUD operations.

---

### 2.2 Domain 2: Machine Learning (Curricular Corpus)

* **Target Concept Node:** `Machine Learning Articles`
* **Stem:** 
  > *"A data scientist is writing a technical article explaining the difference between supervised and unsupervised learning. Which of the following statements correctly distinguishes the two based on the nature of the training data?"*
* **Option A (Correct):** 
  > *"Supervised learning uses labeled data where the input is paired with the correct output, whereas unsupervised learning uses unlabeled data to find hidden structures or patterns without predefined output labels."*
* **Option B (Prerequisite Distractor):** 
  > *"Supervised learning requires a large amount of data to be computationally efficient, while unsupervised learning can operate effectively with very small datasets because it does not need to calculate error gradients."*
* **Option C (Conceptual Distractor):** 
  > *"Supervised learning is used to predict continuous values (regression), while unsupervised learning is used to predict categorical classes (classification), regardless of whether labels are present."*
* **Misconception Tag:** `Confusing algorithm types (regression/classification) with learning paradigms (supervised/unsupervised)`
* **Diagnostic Function:** Option B confuses computational optimization prerequisites with paradigm definitions, while Option C conflates problem task targets (continuous vs. discrete prediction) with supervision signals.

---

### 2.3 Domain 3: Photosynthesis (Scientific Corpus)

* **Target Concept Node:** `photosynthesis`
* **Stem:** 
  > *"During the light-dependent reactions of photosynthesis, what is the primary role of the electron transport chain (ETC) in the thylakoid membrane?"*
* **Option A (Correct):** 
  > *"To create a proton gradient across the membrane that drives ATP synthase to produce ATP."*
* **Option B (Prerequisite Distractor):** 
  > *"To directly convert light energy into chemical energy stored in glucose molecules."*
* **Option C (Conceptual Distractor):** 
  > *"To reduce NADP+ to NADPH by accepting electrons from water, which is the sole source of ATP production."*
* **Misconception Tag:** `Confusing the ETC's role in chemiosmosis with direct substrate-level phosphorylation or glucose synthesis`
* **Diagnostic Function:** Option B reveals an upstream gap in basic cellular energy storage prerequisites (glucose synthesis occurs downstream in the Calvin cycle, not the thylakoid ETC). Option C isolates internal biochemical confusion between electron carriers and phosphorylation mechanisms.

---

## 3. Module 4: Dynamic Graph Remediation & Memory Decay

Module 4 executes runtime graph rewriting when a learner triggers a prerequisite misconception, converting diagnostic feedback into topological intervention.

### 3.1 Remediation Cycle Mechanics (`GraphRewriter`)

The graph rewriter operates in real time on the active session's in-memory `GraphModel` without re-running S0–S6. The lifecycle proceeds as follows:

```
[Initial Graph]
  derivatives ───────────────────────────────► calculus ──────────► integrals
  (status: in_progress)                       (status: in_progress)

                  │
                  ▼  Learner selects Option B on target 'calculus'
                  │  Misconception: M_derivative_calculus
                  ▼

[Remediation Injected]
  derivatives ──► RM_derivative_calculus ───► calculus ──────────► integrals
                  (status: in_progress)       (status: LOCKED)    (status: LOCKED)
                  [Dynamic Node]              [Burden Edge]

                  │
                  ▼  Learner passes remediation criteria
                  │  (2 consecutive correct reviews)
                  ▼

[Resolved & Restored]
  derivatives ───────────────────────────────► calculus ──────────► integrals
  (status: mastered)                          (status: in_progress)
```

#### Step-by-Step Architectural Walkthrough:

1. **Initial Precedence State:**
   - Upstream node: `derivatives` (confidence $0.85$, status: `in_progress`).
   - Target node: `calculus` (status: `in_progress`).
   - Downstream node: `integrals` (status: `locked` by topological precedence).
   - Active edge: `derivatives` $\to$ `calculus` (type: `PREREQUISITE`, weight: $0.85$).

2. **Trigger Event:**
   - The learner attempts the diagnostic quiz for `calculus` and selects Option B, triggering misconception tag `M_derivative_calculus`.

3. **Runtime Graph Mutation (`inject_remediation`):**
   - **Snapshot:** The rewriter stashes all incoming prerequisite edges to `calculus` in memory: `[('derivatives', 'calculus', 0.85)]`.
   - **Node Injection:** A temporary remediation concept node `RM_derivative_calculus` is instantiated with:
     ```python
     ConceptNode(
         id="RM_derivative_calculus",
         label="Remediate M_derivative_calculus",
         status=NodeStatus.IN_PROGRESS,
         is_dynamic_remediation=True,
         triggering_misconception="M_derivative_calculus"
     )
     ```
   - **Edge Rerouting:**
     - Original edge `derivatives` $\to$ `calculus` is severed.
     - Upstream edge is redirected: `derivatives` $\to$ `RM_derivative_calculus` (`type: PREREQUISITE`).
     - A burden dependency is injected: `RM_derivative_calculus` $\to$ `calculus` (`type: REMEDIATION_LINK`, confidence: $1.0$).
   - **Target Lock:** `calculus.status` is set to `NodeStatus.LOCKED`. The learner cannot proceed to `calculus` (or downstream `integrals`) until the remediation node is cleared.
   - **Client Sync:** The mutated graph is serialized into the standard JSON response; the frontend React Flow renderer displays `RM_derivative_calculus` as an amber detour with an alert badge, with `calculus` visually locked.

4. **Resolution (`resolve_remediation`):**
   - The learner reviews targeted micro-content and achieves `resolution_successes = 2` consecutive correct answers on the remediation concept.
   - `resolve_remediation("RM_derivative_calculus")` triggers:
     - Incident edges to and from `RM_derivative_calculus` are removed.
     - Node `RM_derivative_calculus` is deleted from the graph.
     - The stashed edge `derivatives` $\to$ `calculus` is restored.
     - `calculus.status` is unlocked to `NodeStatus.IN_PROGRESS`.
   - The learner cleanly returns to their primary curriculum path.

### 3.2 Spaced Memory Decay Model (`DecayModel`)

To maintain long-term retention across complex multi-node roadmaps, Module 4 integrates exponential memory decay with parameterized scheduling (SM-2 and FSRS):
- **Exponential Retention:** For any mastered concept $v$ with stability $S_v$ (in seconds), retention probability decays over elapsed time $\Delta t$:
  $$R(\Delta t) = \exp\left(-\frac{\Delta t}{S_v}\right)$$
- **Crossing-Point Detection:** When $R(\Delta t)$ drops below the decay threshold $\tau_{\text{decay}} = 0.5$, the time until decay is analytically derived as:
  $$\Delta t_{\text{decay}} = S_v \cdot \ln(1 / \tau_{\text{decay}})$$
- When this threshold is crossed, the concept node transitions to `NodeStatus.DECAYED`, alerting the learner to perform a quick refresher review before continuing deeper down the dependency graph.
