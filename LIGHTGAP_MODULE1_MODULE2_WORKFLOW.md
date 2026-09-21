# LightGAP: Complete Mathematical & Algorithmic Workflow (Modules 1 & 2)

> **Subject Input Example**: `"Machine Learning"`  
> **Reference**: *LightGAP: A Lightweight, Graph-Based Adaptive Pathway System for Prerequisite-Aware Intelligent Tutoring (Section 4.1 & 4.2)*

---

## Executive Summary

This document formalizes the exact mathematical and computational pipeline connecting raw concept text to a topologically valid, cycle-free, and budget-optimized prerequisite Directed Acyclic Graph (DAG).

```
"Machine Learning"
       ↓
[Phase 0: Concept Set Extraction]
       ↓ (N = 6 concepts)
[Module 1: Prerequisite Determination Engine]
  1. Concept Feature Matrix H (N × 384) via all-MiniLM-L6-v2
  2. Asymmetric Directional Feature Matrix X (M × 1536)
  3. Stage-1: Baseline Logistic Regression (P_LR)
  4. Stage-2: 2-Layer Directed Graph Convolutional Network (P_GNN)
  5. Stage-3: Small-Language-Model (SLM) Counterfactual Perplexity Probe (P_final)
       ↓ Scored Candidate Pairs P(u → v)
[Module 2: Graph Construction & Constraint Optimization Engine]
  1. Threshold Filter τ_edge (≥ 0.65)
  2. Tarjan's Strongly Connected Components (SCC) Cycle Elimination
  3. Transitive Reduction (Prune redundant shortcut hops)
  4. Precedence-Constrained Knapsack Optimization (ILP under time budget)
       ↓
Final Verified Prerequisite DAG G = (V, E)
```

---

# Phase 0: Curriculum Concept Decomposition ($V$)

Given the user query `"Machine Learning"`, the system decomposes the domain into discrete concept nodes $V = \{c_1, c_2, \dots, c_N\}$. For mathematical clarity, we trace $N = 6$ foundational concepts:

| ID | Concept Label | Text Definition & Learning Outcome ($t_i$) |
|---|---|---|
| $c_1$ | `linear_algebra` | Matrices, vector spaces, dot products, eigenvalues. Learning outcome: perform matrix transformations. |
| $c_2$ | `probability_stats` | Random variables, distributions, expectation, variance. Learning outcome: quantify uncertainty. |
| $c_3$ | `gradient_descent` | Objective functions, partial derivatives, learning rates, convergence. Learning outcome: minimize cost functions. |
| $c_4$ | `linear_regression` | Closed-form normal equations and iterative fitting of linear hyperplanes. Learning outcome: predict continuous scalars. |
| $c_5$ | `logistic_regression` | Sigmoid activation, cross-entropy loss, decision boundaries. Learning outcome: binary classification. |
| $c_6$ | `evaluation_metrics` | Train/test splits, precision, recall, ROC-AUC, overfitting. Learning outcome: non-circular validation. |

---

# Module 1: Prerequisite Determination Engine

The objective of Module 1 is to predict directional dependency probabilities $P(u \to v) \in [0, 1]$ for candidate concept pairs without manual labeling.

---

### Step 1.1: Concept Feature Matrix ($H \in \mathbb{R}^{N \times 384}$)

Each concept's textual description $t_i$ is encoded using `all-MiniLM-L6-v2` (an 80 MB sentence transformer running in $<0.5\text{s}$ on CPU):

$$h_i = \text{MiniLM}(t_i) \in \mathbb{R}^{384}, \quad \text{for } i \in \{1, \dots, 6\}$$

Stacking the normalized vectors row-wise gives the **Concept Node Feature Matrix**:

$$H = \begin{bmatrix}
— & h_{\text{linear\_algebra}} & — \\
— & h_{\text{probability\_stats}} & — \\
— & h_{\text{gradient\_descent}} & — \\
— & h_{\text{linear\_regression}} & — \\
— & h_{\text{logistic\_regression}} & — \\
— & h_{\text{evaluation\_metrics}} & —
\end{bmatrix} \in \mathbb{R}^{6 \times 384}$$

#### Why Cosine Similarity Fails as a Prerequisite Metric:
$$\cos(h_1, h_4) = \frac{h_1 \cdot h_4}{\|h_1\| \|h_4\|} = \cos(h_4, h_1) \approx 0.78$$
Cosine similarity is strictly symmetric: $\cos(h_u, h_v) = \cos(h_v, h_u)$. It confirms topical relatedness, but is mathematically blind to dependency direction ($u \to v$ vs. $v \to u$).

---

### Step 1.2: Candidate Pair Universe ($M = 30$ pairs)

We form all directed ordered pairs $(u, v)$ such that $u \neq v$:

$$M = N(N - 1) = 6 \times 5 = 30 \text{ candidate pairs}$$

Every relationship is tested in both forward and backward orientations:
* $(c_1 \to c_4)$: Does Linear Algebra precede Linear Regression?
* $(c_4 \to c_1)$: Does Linear Regression precede Linear Algebra?

---

### Step 1.3: Asymmetric Directional Feature Matrix ($X \in \mathbb{R}^{30 \times 1536}$)

To capture directionality, Section 4.1 defines the 1536-dimensional directional feature vector:

$$f_{\text{dir}}(u, v) = \Big[\, \underbrace{h_u}_{384} \;\parallel\; \underbrace{h_v}_{384} \;\parallel\; \underbrace{(h_u - h_v)}_{384} \;\parallel\; \underbrace{(h_u \odot h_v)}_{384} \,\Big] \in \mathbb{R}^{1536}$$

Where:
* $h_u$: Prerequisite semantic representation
* $h_v$: Target semantic representation
* $(h_u - h_v)$: **Directional gradient**. Swapping $u$ and $v$ negates this block:
  $$(h_u - h_v) = -(h_v - h_u)$$
* $(h_u \odot h_v)$: Element-wise Hadamard interaction

Stacking all 30 pair vectors forms the **Pairwise Feature Matrix**:
$$X \in \mathbb{R}^{30 \times 1536}$$

---

### Step 1.4: Stage 1 — Baseline Logistic Regression

The feature matrix $X$ is evaluated through calibrated logistic regression weights $W_{\text{LR}} \in \mathbb{R}^{1536}$ and bias $b_{\text{LR}} \in \mathbb{R}$:

$$P_{\text{LR}}(u \to v) = \sigma(X W_{\text{LR}} + b_{\text{LR}}) = \frac{1}{1 + \exp\left(-\left(X W_{\text{LR}} + b_{\text{LR}}\right)\right)}$$

#### Sample Stage-1 Inferred Outputs:
* $(c_1 \to c_4)$ (Linear Algebra $\to$ Linear Regression): **$0.89$** *(High confidence)*
* $(c_4 \to c_1)$ (Linear Regression $\to$ Linear Algebra): **$0.11$** *(Low confidence / Discarded)*
* $(c_4 \to c_5)$ (Linear Regression $\to$ Logistic Regression): **$0.85$** *(High confidence)*
* $(c_3 \to c_4)$ (Gradient Descent $\to$ Linear Regression): **$0.55$** *(⚠️ Ambiguous: $P \in [0.4, 0.7]$)*

---

### Step 1.5: Stage 2 — 2-Layer Directed GCN (DirGCN)

To leverage network topology rather than isolated pairs, we build an initial adjacency matrix $A \in \{0, 1\}^{6 \times 6}$ from the top-$k$ ($k=2$) outgoing edges per node from Stage 1.

DirGCN aggregates in-neighbors, out-neighbors, and self-loops using three separate learned parameter matrices ($W_{\text{in}}, W_{\text{out}}, W_{\text{self}}$):

$$H^{(l+1)} = \text{ReLU}\left( \tilde{D}_{\text{in}}^{-1} A^T H^{(l)} W_{\text{in}} + \tilde{D}_{\text{out}}^{-1} A H^{(l)} W_{\text{out}} + H^{(l)} W_{\text{self}} \right)$$

* $\tilde{D}_{\text{in}}^{-1} A^T H^{(l)} W_{\text{in}}$: Aggregates prerequisites pointing **into** node $v$
* $\tilde{D}_{\text{out}}^{-1} A H^{(l)} W_{\text{out}}$: Aggregates downstream concepts pointing **out from** node $u$
* $H^{(l)} W_{\text{self}}$: Retains the concept's intrinsic semantic content

After 2 layers, refined link probability is predicted via an MLP over the concatenated representations:
$$P_{\text{GNN}}(u \to v) = \sigma\left( \text{MLP}([H^{(2)}_u \parallel H^{(2)}_v]) \right)$$
* For pair $(c_3 \to c_4)$, $P_{\text{GNN}}(c_3 \to c_4) = 0.59$ (remains in the ambiguous band $[0.4, 0.7]$).

---

### Step 1.6: Stage 3 — SLM Counterfactual Perplexity Probe

To bound compute, the Small Language Model (SLM) probe is invoked **only for edges in the ambiguous band** $P \in [0.4, 0.7]$ (avoiding expensive queries on clear edges).

For $u = \text{"Gradient Descent"}, v = \text{"Linear Regression"}$:

1. **Standard Prompt for $v$**:
   > *"Explain the optimization of Linear Regression."*  
   > Measured perplexity: $\text{PPL}_{\text{standard}} = 12.4$

2. **Counterfactual Prompt (explain $v$ without mentioning $u$)**:
   > *"Explain the optimization of Linear Regression without mentioning Gradient Descent."*  
   > Measured perplexity: $\text{PPL}_{\text{without\_u}} = 22.8$

3. **Normalized Perplexity Increase Metric $D(u \to v)$**:
   $$D(u \to v) = \frac{\text{PPL}_{\text{without\_u}} - \text{PPL}_{\text{standard}}}{\text{PPL}_{\text{standard}}} = \frac{22.8 - 12.4}{12.4} = +0.838$$

4. **Asymmetric Verification $D(v \to u)$**:
   * Explaining Gradient Descent without referencing Linear Regression yields $\text{PPL}$ delta of $+0.12$.
   * Asymmetric difference:
     $$D_{\text{asym}}(u, v) = D(u \to v) - D(v \to u) = 0.838 - 0.12 = +0.718$$

5. **Logit Blending ($\alpha = 0.6$)**:
   $$\text{logit}(P_{\text{final}}) = \text{logit}(P_{\text{GNN}}) + \alpha \cdot D_{\text{asym}}(u, v)$$
   $$P_{\text{final}}(c_3 \to c_4) = \mathbf{0.84} \quad (\text{Ambiguity conclusively resolved!})$$

---

# Module 2: Graph Construction & Constraint Optimization

Module 1 outputs a scored candidate edge list:
```
(c1, c3): 0.82  -> Linear Algebra -> Gradient Descent
(c1, c4): 0.91  -> Linear Algebra -> Linear Regression
(c3, c4): 0.84  -> Gradient Descent -> Linear Regression
(c2, c5): 0.86  -> Probability -> Logistic Regression
(c4, c5): 0.88  -> Linear Regression -> Logistic Regression
(c5, c6): 0.89  -> Logistic Regression -> Evaluation Metrics
(c6, c3): 0.67  -> Evaluation Metrics -> Gradient Descent  (Noisy cycle!)
```

---

### Step 2.1: Threshold Filtering ($\tau_{\text{edge}} = 0.65$)
All candidate pairs with score below $\tau_{\text{edge}}$ are pruned:
$$E_{\text{cand}} = \{ (u, v) \mid P(u \to v) \ge 0.65 \}$$

---

### Step 2.2: Tarjan's SCC Cycle Detection & Pruning
A valid curriculum cannot contain circular prerequisite logic.

Suppose the candidate set contains:
$$c_3 \to c_4 \to c_5 \to c_6 \to c_3$$

1. **Tarjan's DFS Algorithm** identifies $\{c_3, c_4, c_5, c_6\}$ as a Strongly Connected Component (cycle).
2. **Pruning Rule**: Locate the edge in the cycle with minimum confidence:
   * $(c_3 \to c_4): 0.84$
   * $(c_4 \to c_5): 0.88$
   * $(c_5 \to c_6): 0.89$
   * $(c_6 \to c_3): 0.67$ $\leftarrow$ **Minimum confidence edge**
3. **Action**: Edge $(c_6 \to c_3)$ is deleted.
4. **Guarantee**: Graph is now strictly acyclic ($\text{is\_DAG} = \text{True}$).

---

### Step 2.3: Transitive Reduction
Consider the subgraph over $c_1$, $c_3$, and $c_4$:
* Direct edge: $c_1 \to c_4$ (Linear Algebra $\to$ Linear Regression)
* Multi-hop path: $c_1 \to c_3 \to c_4$ (Linear Algebra $\to$ Gradient Descent $\to$ Linear Regression)

Because $c_1 \to c_4$ is implied by the 2-hop sequence, the direct edge is transitively redundant:
$$\text{If } \exists \text{ path } u \leadsto v \text{ of length } \ge 2, \quad \text{delete direct edge } (u, v)$$

* **Action**: Edge $(c_1 \to c_4)$ is pruned.
* **Benefit**: Removes visual clutter and enforces proper pedagogical sequencing.

---

### Step 2.4: Time-Budget Constraint Optimization (ILP)
When a learner specifies a time budget $T_{\text{budget}}$ (e.g., $60\text{ minutes}$):

Each concept has an estimated study time $T(v)$ and pedagogical importance $I(v)$:
* $c_1$: $20\text{ min}$, Importance: $0.8$
* $c_2$: $20\text{ min}$, Importance: $0.8$
* $c_3$: $25\text{ min}$, Importance: $0.9$
* $c_4$: $25\text{ min}$, Importance: $0.9$
* $c_5$: $30\text{ min}$, Importance: $0.95$

We formulate the **Precedence-Constrained Knapsack Problem**:
$$\max \sum_{v \in V} x_v \cdot I(v)$$
$$\text{subject to:}$$
1. **Time Budget Constraint**: $\sum_{v \in V} x_v \cdot T(v) \le T_{\text{budget}}$
2. **Precedence Constraint**: $x_v \le x_u \quad \forall (u, v) \in E_{\text{DAG}}$  
   *(A student cannot be assigned target $v$ without taking prerequisite $u$)*
3. **Integrity**: $x_v \in \{0, 1\}$

Solved exactly via PuLP / SciPy MILP solver in $<0.05\text{s}$ on CPU.

---

## Final Output of Modules 1 & 2

The pipeline outputs a mathematically validated, cycle-free prerequisite DAG:

```text
[Linear Algebra] ───────► [Optimization & Gradient Descent]
                                     │
                                     ▼
                            [Linear Regression]
                                     │
[Probability & Stats] ───────────────┼────────► [Logistic Regression]
                                                     │
                                                     ▼
                                            [Evaluation & Metrics]
```

This clean DAG is serialized into the **Section 6 Dynamic JSON Contract** and sent to **Module 3** (content attachment & diagnostic quiz generation) and **Module 4** (runtime graph mutation upon misconception).
