import React, { useState } from 'react';
import { client } from '../api/client';

export interface BudgetControlsProps {
  onSessionCreated?: (sessionId: string) => void;
}

/**
 * Form to configure a learner session: goal concept + time budget (minutes),
 * mirroring the ``StudentState`` in the JSON contract.
 */
export function BudgetControls({ onSessionCreated }: BudgetControlsProps) {
  const [goal, setGoal] = useState('Machine Learning');
  const [sessionId, setSessionId] = useState('');
  const [busy, setBusy] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');

  const generate = async () => {
    if (!goal.trim()) return;
    setBusy(true);
    setStatusMsg('Synthesizing prerequisite mind map…');
    try {
      const res = await client.createSession(goal.trim(), 0, undefined, true);
      setSessionId(res.session_id);
      onSessionCreated?.(res.session_id);
      setStatusMsg('');
    } catch (err) {
      console.error('session create failed', err);
      setStatusMsg('Failed to generate roadmap.');
    } finally {
      setBusy(false);
    }
  };

  const loadDemo = async () => {
    setBusy(true);
    setStatusMsg('Loading demo…');
    try {
      const res = await client.createSession('Linear Algebra', 0, undefined, false);
      setSessionId(res.session_id);
      const nodeIds = [
        'matrix_multiplication',
        'determinant',
        'inverse_matrix',
        'linear_system',
        'eigenvalues',
      ];
      const pairs: [string, string][] = [
        ['matrix_multiplication', 'determinant'],
        ['matrix_multiplication', 'inverse_matrix'],
        ['determinant', 'inverse_matrix'],
        ['inverse_matrix', 'linear_system'],
        ['determinant', 'eigenvalues'],
        ['matrix_multiplication', 'eigenvalues'],
      ];
      const scores = [0.88, 0.92, 0.85, 0.80, 0.89, 0.78];
      await client.buildGraph(res.session_id, nodeIds, pairs, scores);
      onSessionCreated?.(res.session_id);
      setStatusMsg('');
    } catch (err) {
      console.error('demo load failed', err);
      setStatusMsg('Demo failed.');
    } finally {
      setBusy(false);
    }
  };

  const setQuickGoal = (t: string) => {
    setGoal(t);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: '10px 0 6px' }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 280, maxWidth: 540 }}>
          <div style={{ position: 'relative', flex: 1, display: 'flex', alignItems: 'center' }}>
            <span style={{ position: 'absolute', left: 12, color: '#94a3b8', fontSize: 14 }}>🔍</span>
            <input
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && generate()}
              placeholder="Enter any domain (e.g. Machine Learning, Transformers, Linear Algebra...)"
              style={{
                width: '100%',
                padding: '9px 14px 9px 34px',
                borderRadius: 10,
                border: '1.5px solid #cbd5e1',
                fontSize: 13.5,
                outline: 'none',
                background: '#ffffff',
                boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
                transition: 'border-color 0.15s ease',
              }}
            />
          </div>
          <button
            onClick={generate}
            disabled={busy || !goal.trim()}
            style={{
              padding: '9px 18px',
              background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
              color: 'white',
              border: 'none',
              borderRadius: 10,
              fontWeight: 600,
              cursor: 'pointer',
              fontSize: 13.5,
              whiteSpace: 'nowrap',
              boxShadow: '0 2px 5px rgba(37, 99, 235, 0.25)',
              transition: 'opacity 0.15s ease',
            }}
          >
            {busy ? 'Synthesizing…' : '🚀 Generate Mind Map'}
          </button>
        </div>

        <button
          onClick={loadDemo}
          disabled={busy}
          style={{
            padding: '8px 14px',
            background: '#f8fafc',
            color: '#475569',
            border: '1px solid #cbd5e1',
            borderRadius: 8,
            cursor: 'pointer',
            fontSize: 12.5,
            fontWeight: 500,
          }}
        >
          Linear Algebra Demo
        </button>

        {statusMsg && <span style={{ fontSize: 12, color: '#2563eb', fontWeight: 600 }}>{statusMsg}</span>}
        {sessionId && !statusMsg && (
          <span style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'monospace' }}>Session: {sessionId.slice(0, 8)}</span>
        )}
      </div>

      {/* Quick Topic Presets */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8' }}>Try:</span>
        {[
          'Machine Learning',
          'Transformers & LLMs',
          'Linear Algebra',
          'Reinforcement Learning',
          'Computer Vision',
        ].map((topic) => (
          <button
            key={topic}
            onClick={() => setQuickGoal(topic)}
            style={{
              background: '#f1f5f9',
              border: 'none',
              borderRadius: 12,
              padding: '2px 10px',
              fontSize: 11,
              color: '#475569',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            {topic}
          </button>
        ))}
      </div>
    </div>
  );
}

export default BudgetControls;