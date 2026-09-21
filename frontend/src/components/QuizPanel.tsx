import React, { useState } from 'react';
import type { QuizQuestionPayload, QuizResponsePayload } from '../api/client';
import { client } from '../api/client';

export interface QuizPanelProps {
  sessionId: string;
  nodeId: string;
  questions: QuizQuestionPayload[];
  onResult?: (result: QuizResponsePayload) => void;
}

/**
 * Renders a node's three diagnostic questions. Selecting Option B (index 1, the
 * prerequisite distractor) sends a grade request whose response carries the
 * attributed misconception id so Module 4 can rewrite the graph server-side.
 */
export function QuizPanel({ sessionId, nodeId, questions, onResult }: QuizPanelProps) {
  const [selected, setSelected] = useState<Record<string, number>>({});
  const [results, setResults] = useState<Record<string, QuizResponsePayload>>({});

  const submitAnswer = async (q: QuizQuestionPayload, index: number) => {
    setSelected((s) => ({ ...s, [q.stem]: index }));
    try {
      const result = await client.gradeQuiz(sessionId, nodeId, q.stem, index);
      setResults((r) => ({ ...r, [q.stem]: result }));
      onResult?.(result);
    } catch (err) {
      console.error('grade failed', err);
    }
  };

  return (
    <div className="quiz-panel" style={{ padding: 12, borderTop: '1px solid #e2e8f0' }}>
      <h3>Diagnostic Quiz — {nodeId}</h3>
      {questions.map((q) => {
        const chosen = selected[q.stem];
        const result = results[q.stem];
        return (
          <div key={q.stem} style={{ marginBottom: 12 }}>
            <p style={{ fontWeight: 600 }}>{q.stem}</p>
            {q.options.map((opt, i) => (
              <label key={i} style={{ display: 'block', margin: '4px 0' }}>
                <input
                  type="radio"
                  name={q.stem}
                  checked={chosen === i}
                  onChange={() => submitAnswer(q, i)}
                  disabled={chosen !== undefined}
                />
                <span style={{ marginLeft: 6 }}>{opt}</span>
                {result && result.selected_index === i && (
                  <span style={{ marginLeft: 8, color: result.is_correct ? 'green' : 'red' }}>
                    {result.is_correct ? '✓' : result.attributed_misconception ? `✗ (${result.attributed_misconception})` : '✗'}
                  </span>
                )}
              </label>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export default QuizPanel;