import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import { BudgetControls } from './components/BudgetControls';
import { RoadmapGraph } from './components/RoadmapGraph';
import { NotebookLMPanel } from './components/NotebookLMPanel';
import type { GraphPayload } from './api/client';
import { client } from './api/client';

function App() {
  const [sessionId, setSessionId] = useState('');
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const loadGraph = async (sid: string) => {
    const g = await client.getGraph(sid);
    setGraph(g);
    if (g.nodes.length > 0 && !selectedNode) {
      const active = g.nodes.find((n) => n.status === 'in_progress') || g.nodes.find((n) => n.status === 'available') || g.nodes[0];
      setSelectedNode(active.id);
    }
  };

  const onSessionCreated = async (sid: string) => {
    setSessionId(sid);
    setSelectedNode(null);
    await loadGraph(sid);
  };

  const onNodeClick = (nodeId: string) => {
    setSelectedNode(nodeId);
  };

  const handleLearnNext = async () => {
    if (!sessionId) return;
    try {
      const res = await client.getNextSuggestion(sessionId);
      if (res.next_node_id) {
        setSelectedNode(res.next_node_id);
      }
    } catch (err) {
      console.error('Failed to get next suggestion', err);
    }
  };

  // Compute overall progress stats
  const completedCount = graph ? graph.nodes.filter((n) => n.status === 'completed' && (!n.node_type || n.node_type === 'leaf')).length : 0;
  const totalLeaves = graph ? graph.nodes.filter((n) => !n.node_type || n.node_type === 'leaf').length : 0;
  const progressPct = totalLeaves > 0 ? Math.round((completedCount / totalLeaves) * 100) : 0;

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif', display: 'flex', flexDirection: 'column', height: '100vh', background: '#f8fafc', color: '#0f172a' }}>
      <header style={{ borderBottom: '1px solid #e2e8f0', padding: '14px 24px 10px', background: '#ffffff', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: 'linear-gradient(135deg, #2563eb, #7c3aed)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ffffff', fontSize: 18, boxShadow: '0 2px 6px rgba(37,99,235,0.3)' }}>
              🧠
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#0f172a', letterSpacing: '-0.01em' }}>
                  LightGAP — Adaptive Knowledge Pathway
                </h1>
                <span style={{ fontSize: 10, fontWeight: 700, background: '#eff6ff', color: '#2563eb', padding: '2px 8px', borderRadius: 12, border: '1px solid #bfdbfe' }}>
                  Intelligent Tutoring System
                </span>
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                Directed Prerequisite Discovery (DirGCN) • Diagnostic Assessment • Dynamic Graph Remediation (SLM)
              </div>
            </div>
          </div>

          {/* Quick learning actions & progress */}
          {graph && graph.nodes.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#f8fafc', padding: '6px 14px', borderRadius: 10, fontSize: 12, border: '1px solid #e2e8f0' }}>
                <span style={{ fontWeight: 600, color: '#475569' }}>Curriculum Progress:</span>
                <div style={{ width: 80, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${progressPct}%`, height: '100%', background: '#10b981', transition: 'width 0.3s ease' }} />
                </div>
                <span style={{ fontWeight: 700, color: '#059669' }}>{completedCount}/{totalLeaves}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>({progressPct}%)</span>
              </div>
              <button
                onClick={handleLearnNext}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '7px 16px',
                  background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: 10,
                  fontSize: 12.5,
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 2px 6px rgba(37,99,235,0.25)',
                }}
              >
                <span>🎯</span> Next Concept to Learn
              </button>
            </div>
          )}
        </div>
        <BudgetControls onSessionCreated={onSessionCreated} />
      </header>

      <main style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <section style={{ flex: 1, position: 'relative', background: '#f8fafc' }}>
          {graph && graph.nodes.length > 0 ? (
            <RoadmapGraph
              graph={graph}
              selectedNodeId={selectedNode}
              onNodeClick={onNodeClick}
            />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', color: '#64748b', padding: 24 }}>
              <div style={{ width: 64, height: 64, borderRadius: 20, background: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 32, marginBottom: 16, border: '1px solid #bfdbfe' }}>
                🗺️
              </div>
              <h3 style={{ margin: '0 0 8px', color: '#0f172a', fontSize: 18 }}>Adaptive Knowledge Pathway Ready</h3>
              <p style={{ margin: 0, fontSize: 13.5, maxWidth: 500, textAlign: 'center', color: '#64748b', lineHeight: 1.5 }}>
                Enter any domain above (e.g. <strong>Machine Learning</strong>, <strong>Transformers & LLMs</strong>) to synthesize a multi-stage sequential learning roadmap with diagnostic assessments.
              </p>
            </div>
          )}
        </section>

        <aside style={{ width: 440, overflowY: 'auto', borderLeft: '1px solid #e2e8f0', background: '#ffffff' }}>
          {sessionId && selectedNode ? (
            <NotebookLMPanel
              sessionId={sessionId}
              nodeId={selectedNode}
              onGraphMutated={() => loadGraph(sessionId)}
              onNodeCompleted={() => loadGraph(sessionId)}
            />
          ) : (
            <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
              <div style={{ fontSize: 36, marginBottom: 12 }}>🎓</div>
              <h3 style={{ margin: '0 0 8px', color: '#0f172a', fontSize: 16 }}>LightGAP Study & Diagnostic Lab</h3>
              <p style={{ fontSize: 13, lineHeight: 1.5, margin: 0, color: '#64748b' }}>
                Select any concept node in the roadmap to view curated video lectures, inspect GitHub code repositories, and complete diagnostic tests.
              </p>
              <div style={{ marginTop: 24, textAlign: 'left', background: '#f8fafc', padding: 18, borderRadius: 10, fontSize: 12, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 700, color: '#334155', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.04em', fontSize: 11 }}>
                  Learning State Legend:
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: '#10b981', fontWeight: 700, fontSize: 14 }}>✓</span> <strong>Mastered</strong> — Diagnostic assessment completed successfully
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: '#2563eb', fontWeight: 700, fontSize: 14 }}>◐</span> <strong>In Progress</strong> — Currently active learning unit
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: '#d97706', fontWeight: 700, fontSize: 14 }}>🔓</span> <strong>Ready to Learn</strong> — Prerequisites satisfied and unlocked
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: '#64748b', fontWeight: 700, fontSize: 14 }}>🔒</span> <strong>Locked</strong> — Complete upstream prerequisite nodes first
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ color: '#ea580c', fontWeight: 700, fontSize: 14 }}>⚡</span> <strong>Dynamic Remediation</strong> — Injected upon misconception diagnosis
                </div>
              </div>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);