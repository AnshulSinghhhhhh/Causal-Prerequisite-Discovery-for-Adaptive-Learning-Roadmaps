# LightGAP — UI Design Brief

Read `00_ANTIGRAVITY_BUILD_BRIEF.md` and `01_ARCHITECTURE_SPEC.md` first — the data this UI renders comes from the `depth`/`cluster_id`-bearing schema and the `tree_paths` closure table defined there, not from the old `level === 0 | 1 | 2` contract.

## Why the current UI looks generic, and what that has to do with layout, not just style

The screenshot that started this redesign shows a flat three-column layout with locked cards reading "Core Mechanisms & Data Structures" — that's not a visual-design failure, it's `frontend/src/components/RoadmapGraph.tsx` computing node position from `level === 0/1/2` with manual `x`/`y` column arithmetic. A tree of arbitrary depth (which S5 now produces) cannot be drawn by that code no matter how it's restyled. So this brief has two layers that must not be conflated:

1. **Layout — computed, not designed.** Node positions come from a real graph-layout algorithm operating on `tree_paths` + `dag_edges`, not from hand-picked coordinates or an LLM's opinion of where things go.
2. **Visual and interaction design — where 21st.dev, Google Stitch, and the impeccable skill do the actual work**, once they have real structure to render.

## Layout: replace the column math

Swap the manual positioning for a real layout library — `dagre` (fast, tree-friendly, what React Flow's own examples use) or `elkjs` (handles denser DAG cross-edges more gracefully) — run against:

- **The hierarchy** (`tree_paths`, one root, arbitrary depth, branches from S5's clusters) as the primary layout skeleton.
- **The DAG prerequisite edges** (`dag_edges`) drawn as a distinct overlay — different stroke style from hierarchy edges, since they mean a different thing (dependency, not containment) and the current `EdgeType.HIERARCHY` / `EdgeType.PREREQUISITE` separation in the backend exists precisely so the frontend can tell them apart.

Depth is unbounded from the layout engine's perspective — don't reintroduce a level cap anywhere in the new renderer, including in prop types or default column-width assumptions.

## Content contract this UI must bind to

| Data | Source | What it drives |
|---|---|---|
| `concepts.depth` | S5 longest-path layering | vertical (or radial) position, arbitrary range |
| `clusters.label` | S5, LLM-named | branch headers |
| `dag_edges` (kept vs `dropped_reason`) | S4 | visible dependency lines; optionally a debug/research view showing dropped edges and why, for the authors' own use during evaluation |
| `NodeStatus` (locked/available/in_progress/completed) | S6 planner + S7 completion cascade | node state styling, unlock affordance |
| `concepts.source_doc_id` / `source_span` | S0 | a "why is this here" provenance affordance — this is a research tool, showing where a concept came from is a feature, not clutter |
| quiz + resource payload | S7 | the existing side-panel pattern, now bound to real per-node content instead of the 8-video keyword ladder |

## Design direction — ground it in what this actually is

This is a research tool for visualizing discovered concept dependency structure, not a consumer learning app. Lean into that vernacular rather than a generic SaaS-dashboard look: think citation/dependency graphs, subway-map-style route lines for the active learning path through the DAG, mastery as a filled/unfilled state rather than a percentage bar on every card. Avoid the rounded-card-with-soft-shadow kit applied uniformly regardless of node kind — a root, a branch, and a leaf are structurally different things and should read as different things, not as the same card at three sizes.

Numbered markers (1. / 2. / 3.) on branches are only appropriate if the branches truly are sequential — with S5-derived clusters they often won't be (siblings under the same parent are frequently learnable in parallel, only cross-branch prerequisite edges impose real order). Don't default to numbering; let the DAG edges themselves show the actual required order, since that's the more honest and more interesting thing to display.

## Where to use the existing tools

- **Google Stitch** — use for the initial screen-level flows: the main graph canvas, the node detail panel, the "generate a roadmap" entry flow, the diagnostic quiz panel. Treat its output as the layout-and-flow sketch, not the final pixels — the actual node positions still come from dagre/elkjs against real data, not from Stitch's mockup coordinates.
- **21st.dev** — use for production component implementation once the flow is settled: node cards (root/branch/leaf as distinct components, not one component with a size prop), the locked/available/completed state treatments, the side panel tabs (Study Packet / Diagnostic Lab, matching what's already in the existing frontend).
- **The impeccable skill** — use for the micro-interactions: unlocking a node, completing a quiz and watching downstream nodes light up, expanding a branch. One or two orchestrated moments (an unlock animation, a path-highlight on hover) land better than motion on every hover state — keep the rest of the interface quiet so those moments read as intentional.

## Two acceptance checks specific to this UI (in addition to the P4 gate in the build brief)

1. Render two structurally different goal concepts (see `05_TESTING_RESULTS_PROTOCOL.md` for which two) and confirm the resulting trees are visibly different in depth and branch count — not two different color schemes applied to the same three-column shape.
2. Turn off all styling temporarily and confirm the raw layout is still legible as a tree with a DAG overlay — if it only looks structured because of card colors and borders, the layout itself hasn't actually changed.
