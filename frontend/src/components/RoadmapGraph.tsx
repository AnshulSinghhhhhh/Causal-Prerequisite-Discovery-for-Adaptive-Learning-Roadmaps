import React, { useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Node,
  MarkerType,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';

import type { GraphPayload, NodePayload } from '../api/client';

/** Color + style for each node status (Section 6 contract statuses + 4-state lifecycle). */
const STATUS_CONFIG: Record<
  string,
  { border: string; fill: string; text: string; stripe: string; badgeBg: string; icon: string; label: string }
> = {
  completed: {
    border: '#10b981',
    fill: '#ffffff',
    text: '#065f46',
    stripe: '#10b981',
    badgeBg: '#dcfce7',
    icon: '✓',
    label: 'Mastered',
  },
  in_progress: {
    border: '#3b82f6',
    fill: '#ffffff',
    text: '#1e40af',
    stripe: '#3b82f6',
    badgeBg: '#dbeafe',
    icon: '◐',
    label: 'In Progress',
  },
  available: {
    border: '#f59e0b',
    fill: '#ffffff',
    text: '#92400e',
    stripe: '#f59e0b',
    badgeBg: '#fef3c7',
    icon: '🔓',
    label: 'Ready to Learn',
  },
  locked: {
    border: '#e2e8f0',
    fill: '#f8fafc',
    text: '#64748b',
    stripe: '#cbd5e1',
    badgeBg: '#f1f5f9',
    icon: '🔒',
    label: 'Locked',
  },
};

const NODE_DIMENSIONS = {
  root: { width: 300, height: 76 },
  branch: { width: 250, height: 68 },
  leaf: { width: 260, height: 84 },
};

/**
 * Dagre Arbitrary-Depth Hierarchical + DAG Overlay Layout.
 * Per UI spec (04_UI_DESIGN_BRIEF.md):
 * - Primary skeleton: hierarchy edges (tree_paths / parent_id)
 * - Directed overlay: dag prerequisite edges
 * - Depth is unbounded: handles arbitrary depth from S5 longest-path layering.
 */
function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
  graphPayload: GraphPayload
): Node[] {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: 'TB',
    nodesep: 48,
    ranksep: 72,
    marginx: 40,
    marginy: 40,
  });

  const nodeTypeLookup = new Map<string, string>();
  graphPayload.nodes.forEach((n) => {
    nodeTypeLookup.set(n.id, n.node_type || (n.depth === 0 ? 'root' : n.children?.length ? 'branch' : 'leaf'));
  });

  // Add nodes to Dagre with distinct dimension per node kind
  nodes.forEach((node) => {
    const kind = (nodeTypeLookup.get(node.id) || 'leaf') as keyof typeof NODE_DIMENSIONS;
    const dims = NODE_DIMENSIONS[kind] || NODE_DIMENSIONS.leaf;
    dagreGraph.setNode(node.id, { width: dims.width, height: dims.height });
  });

  // Add hierarchy edges with higher weight so tree containment guides layout
  graphPayload.edges.forEach((e) => {
    if (dagreGraph.hasNode(e.source) && dagreGraph.hasNode(e.target)) {
      if (e.type === 'hierarchy') {
        dagreGraph.setEdge(e.source, e.target, { weight: 3 });
      } else {
        dagreGraph.setEdge(e.source, e.target, { weight: 1 });
      }
    }
  });

  // If node has parent_id not in edges, add implicit hierarchy edge for dagre
  graphPayload.nodes.forEach((n) => {
    if (n.parent_id && dagreGraph.hasNode(n.parent_id) && dagreGraph.hasNode(n.id)) {
      if (!dagreGraph.hasEdge(n.parent_id, n.id)) {
        dagreGraph.setEdge(n.parent_id, n.id, { weight: 3 });
      }
    }
  });

  try {
    dagre.layout(dagreGraph);
  } catch (err) {
    console.warn('[RoadmapGraph] Dagre layout fallback on cycle:', err);
  }

  return nodes.map((node) => {
    const nodeWithPos = dagreGraph.node(node.id);
    const kind = (nodeTypeLookup.get(node.id) || 'leaf') as keyof typeof NODE_DIMENSIONS;
    const dims = NODE_DIMENSIONS[kind] || NODE_DIMENSIONS.leaf;
    return {
      ...node,
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      position: {
        x: nodeWithPos ? nodeWithPos.x - dims.width / 2 : 0,
        y: nodeWithPos ? nodeWithPos.y - dims.height / 2 : 0,
      },
    };
  });
}

export interface RoadmapGraphProps {
  graph: GraphPayload;
  selectedNodeId?: string | null;
  onNodeClick?: (nodeId: string) => void;
}

export function RoadmapGraph({
  graph,
  selectedNodeId,
  onNodeClick,
}: RoadmapGraphProps) {
  // Collapsed branch/cluster state
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());

  const toggleCollapse = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  // Build visible nodes & edges based on collapse state
  const { visibleNodes, visibleEdges } = useMemo(() => {
    const nodeMap = new Map<string, NodePayload>();
    graph.nodes.forEach((n) => nodeMap.set(n.id, n));

    // Determine hidden descendants of collapsed nodes
    const hiddenNodeIds = new Set<string>();
    const findDescendants = (parentId: string) => {
      const parent = nodeMap.get(parentId);
      if (parent && parent.children) {
        parent.children.forEach((childId) => {
          hiddenNodeIds.add(childId);
          findDescendants(childId);
        });
      }
    };

    collapsedIds.forEach((cid) => {
      findDescendants(cid);
    });

    // Map payload nodes to ReactFlow nodes
    const flowNodes: Node[] = graph.nodes
      .filter((n) => !hiddenNodeIds.has(n.id))
      .map((n) => {
        const isRemediation = n.is_dynamic_remediation;
        const isSelected = selectedNodeId === n.id;
        const nodeType = n.node_type || (n.depth === 0 ? 'root' : (n.children && n.children.length > 0) ? 'branch' : 'leaf');
        const isCollapsed = collapsedIds.has(n.id);
        const hasChildren = (n.children && n.children.length > 0);

        let cfg = STATUS_CONFIG[n.status] ?? STATUS_CONFIG.locked;
        if (isRemediation) {
          cfg = {
            border: '#ea580c',
            fill: '#ffffff',
            text: '#9a3412',
            stripe: '#ea580c',
            badgeBg: '#ffedd5',
            icon: '⚡',
            label: 'Remediation',
          };
        }

        const cleanLabel = n.label || n.id.replace(/_/g, ' ');
        const progressPct = Math.round((n.progress || 0) * 100);

        // 1. Root Node Style (Curriculum Master Hub)
        if (nodeType === 'root') {
          return {
            id: n.id,
            data: {
              node_type: 'root',
              label: (
                <div style={{ userSelect: 'none', padding: '6px 8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#38bdf8', background: 'rgba(56, 189, 248, 0.15)', padding: '2px 8px', borderRadius: 12 }}>
                      🌟 Master Goal
                    </span>
                    <span style={{ fontSize: 11, background: '#1e293b', color: '#94a3b8', padding: '2px 8px', borderRadius: 10, fontWeight: 600 }}>
                      {progressPct}% Mastered
                    </span>
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: '#ffffff', lineHeight: 1.35, letterSpacing: '-0.01em' }}>
                    {cleanLabel}
                  </div>
                  <div style={{ width: '100%', height: 5, background: '#1e293b', borderRadius: 3, marginTop: 10, overflow: 'hidden' }}>
                    <div style={{ width: `${progressPct}%`, height: '100%', background: 'linear-gradient(90deg, #38bdf8, #818cf8)', transition: 'width 0.4s ease' }} />
                  </div>
                </div>
              ),
            },
            position: { x: 0, y: 0 },
            style: {
              background: 'linear-gradient(135deg, #0f172a, #1e293b)',
              color: '#ffffff',
              border: isSelected ? '2px solid #38bdf8' : '1px solid #334155',
              borderRadius: 14,
              padding: '12px 16px',
              minWidth: 260,
              maxWidth: 320,
              boxShadow: isSelected
                ? '0 0 0 4px rgba(56, 189, 248, 0.35), 0 12px 28px rgba(0,0,0,0.3)'
                : '0 8px 20px rgba(0,0,0,0.18)',
              cursor: 'pointer',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            },
          };
        }

        // 2. Branch Node Style (Cluster / Community Header Card)
        if (nodeType === 'branch') {
          return {
            id: n.id,
            data: {
              node_type: 'branch',
              label: (
                <div style={{ userSelect: 'none', padding: '2px 4px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span
                      style={{
                        fontSize: 9.5,
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        padding: '2px 7px',
                        borderRadius: 10,
                        background: '#e0e7ff',
                        color: '#3730a3',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 4,
                      }}
                    >
                      <span>🗂️</span> Cluster
                    </span>
                    {hasChildren && (
                      <button
                        onClick={(e) => toggleCollapse(n.id, e)}
                        title={isCollapsed ? 'Expand cluster subtopics' : 'Collapse cluster subtopics'}
                        style={{
                          background: '#f1f5f9',
                          border: '1px solid #cbd5e1',
                          borderRadius: '50%',
                          width: 22,
                          height: 22,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 10,
                          cursor: 'pointer',
                          color: '#475569',
                          fontWeight: 700,
                        }}
                      >
                        {isCollapsed ? '▶' : '▼'}
                      </button>
                    )}
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', lineHeight: 1.25 }}>
                    {cleanLabel}
                  </div>
                  {n.children && n.children.length > 0 && (
                    <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 4 }}>
                      {n.children.length} concepts
                    </div>
                  )}
                </div>
              ),
            },
            position: { x: 0, y: 0 },
            style: {
              border: '1.5px solid #c7d2fe',
              borderLeft: '5px solid #6366f1',
              background: '#fcfdff',
              borderRadius: 12,
              padding: '10px 14px',
              minWidth: 230,
              maxWidth: 270,
              boxShadow: isSelected
                ? '0 0 0 3px rgba(99, 102, 241, 0.4), 0 8px 18px rgba(0,0,0,0.08)'
                : '0 2px 8px rgba(0,0,0,0.04)',
              cursor: 'pointer',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              opacity: n.status === 'locked' ? 0.8 : 1,
            },
          };
        }

        // 3. Leaf Node Style (Atomic Concept Unit with Depth Indicator)
        const depthTag = n.depth !== undefined ? `Depth ${n.depth}` : '';

        return {
          id: n.id,
          data: {
            node_type: 'leaf',
            label: (
              <div style={{ userSelect: 'none', padding: '2px 2px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
                  <span
                    style={{
                      fontSize: 9,
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      padding: '2px 7px',
                      borderRadius: 6,
                      background: cfg.badgeBg,
                      color: cfg.text,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 3,
                    }}
                  >
                    <span>{cfg.icon}</span> {cfg.label}
                  </span>
                  {depthTag && (
                    <span style={{ fontSize: 9.5, fontWeight: 700, color: '#94a3b8', background: '#f1f5f9', padding: '1px 6px', borderRadius: 4, fontFamily: 'monospace' }}>
                      {depthTag}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: n.status === 'locked' ? '#64748b' : '#0f172a', lineHeight: 1.35 }}>
                  {cleanLabel}
                </div>
              </div>
            ),
          },
          position: { x: 0, y: 0 },
          style: {
            border: `1.5px solid ${cfg.border}`,
            borderLeft: `4px solid ${cfg.stripe}`,
            background: cfg.fill,
            borderRadius: 10,
            padding: '10px 14px',
            minWidth: 230,
            maxWidth: 270,
            boxShadow: isSelected
              ? '0 0 0 3px rgba(37, 99, 235, 0.35), 0 8px 16px rgba(0,0,0,0.08)'
              : isRemediation
              ? '0 0 0 2px rgba(234, 88, 12, 0.3), 0 4px 10px rgba(0,0,0,0.06)'
              : '0 2px 5px rgba(0,0,0,0.03)',
            cursor: 'pointer',
            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            opacity: n.status === 'locked' ? 0.7 : 1,
          },
        };
      });

    // Map payload edges to ReactFlow edges with distinct visual treatments for hierarchy vs prerequisite
    const flowEdges: Edge[] = graph.edges
      .filter((e) => !hiddenNodeIds.has(e.source) && !hiddenNodeIds.has(e.target))
      .map((e, i) => {
        const isRemediation = e.type === 'remediation_link';
        const isHierarchy = e.type === 'hierarchy';

        if (isHierarchy) {
          return {
            id: `h:${e.source}->${e.target}:${i}`,
            source: e.source,
            target: e.target,
            type: 'smoothstep',
            animated: false,
            style: { stroke: '#cbd5e1', strokeWidth: 1.5, strokeDasharray: '3 3' },
          };
        }

        if (isRemediation) {
          return {
            id: `r:${e.source}->${e.target}:${i}`,
            source: e.source,
            target: e.target,
            type: 'smoothstep',
            animated: true,
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: '#ea580c',
            },
            label: '⚡ repair',
            style: { stroke: '#ea580c', strokeWidth: 2.2, strokeDasharray: '4 4' },
          };
        }

        // Standard prerequisite edge: directed dependency line
        return {
          id: `p:${e.source}->${e.target}:${i}`,
          source: e.source,
          target: e.target,
          type: 'smoothstep',
          animated: e.confidence > 0.85,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#2563eb',
          },
          style: { stroke: '#3b82f6', strokeWidth: 2.0 },
        };
      });

    // Apply Dagre arbitrary-depth hierarchical layout
    const layoutedNodes = applyDagreLayout(flowNodes, flowEdges, graph);

    return { visibleNodes: layoutedNodes, visibleEdges: flowEdges };
  }, [graph, selectedNodeId, collapsedIds]);

  return (
    <div style={{ width: '100%', height: '100%', minHeight: 480, background: '#f8fafc' }}>
      <ReactFlow
        nodes={visibleNodes}
        edges={visibleEdges}
        fitView
        minZoom={0.2}
        maxZoom={1.8}
        onNodeClick={(_, node) => onNodeClick?.(node.id)}
      >
        <Background gap={18} size={1} color="#e2e8f0" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

export default RoadmapGraph;