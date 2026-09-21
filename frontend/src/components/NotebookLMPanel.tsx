import React, { useState, useEffect } from 'react';
import type { StudyPackPayload, QuizResponsePayload } from '../api/client';
import { client } from '../api/client';

export interface NotebookLMPanelProps {
  sessionId: string;
  nodeId: string;
  onGraphMutated?: () => void;
  onNodeCompleted?: (nodeId: string) => void;
}

export function NotebookLMPanel({
  sessionId,
  nodeId,
  onGraphMutated,
  onNodeCompleted,
}: NotebookLMPanelProps) {
  const [activeTab, setActiveTab] = useState<'study' | 'quiz'>('study');
  const [pack, setPack] = useState<StudyPackPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Active embedded video ID
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);

  // Quiz state
  const [selectedOpt, setSelectedOpt] = useState<number | null>(null);
  const [gradeResult, setGradeResult] = useState<QuizResponsePayload | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [mastered, setMastered] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!sessionId || !nodeId) return;
      setLoading(true);
      setError(null);
      setSelectedOpt(null);
      setGradeResult(null);
      setMastered(false);
      try {
        const res = await client.getStudyPack(sessionId, nodeId);
        if (!cancelled) {
          setPack(res);
          if (res.status === 'completed') {
            setMastered(true);
          }
          if (res.youtube && res.youtube.length > 0 && res.youtube[0].video_id) {
            setActiveVideoId(res.youtube[0].video_id);
          } else {
            setActiveVideoId(null);
          }
        }
      } catch (err: any) {
        if (!cancelled) setError(err.message || 'Failed to load study pack');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [sessionId, nodeId]);

  const handleGrade = async (stem: string, index: number) => {
    setSelectedOpt(index);
    setSubmitting(true);
    try {
      const res = await client.gradeQuiz(sessionId, nodeId, stem, index);
      setGradeResult(res);

      if (res.is_correct) {
        setMastered(true);
        await client.completeNode(sessionId, nodeId);
        onNodeCompleted?.(nodeId);
        onGraphMutated?.();
      } else if (res.action === 'remediate') {
        // Module 4 dynamic graph rewrite
        onGraphMutated?.();
      }
    } catch (err: any) {
      console.error('Quiz grading error', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleManualComplete = async () => {
    try {
      await client.completeNode(sessionId, nodeId);
      setMastered(true);
      if (pack) {
        setPack({ ...pack, status: 'completed' });
      }
      onNodeCompleted?.(nodeId);
      onGraphMutated?.();
    } catch (err) {
      console.error('Failed to complete node', err);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>
        <div style={{ fontSize: 32, marginBottom: 10 }}>⚡</div>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#334155' }}>
          Synthesizing Study Packet…
        </div>
        <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
          Retrieving multimodal content and diagnostic probes for <strong>{nodeId}</strong>
        </p>
      </div>
    );
  }

  if (error || !pack) {
    return (
      <div style={{ padding: 20, color: '#dc2626', background: '#fef2f2', margin: 16, borderRadius: 10, border: '1px solid #fecaca' }}>
        <div style={{ fontWeight: 600, fontSize: 13 }}>Could not load resources</div>
        <div style={{ fontSize: 12, marginTop: 4 }}>{error}</div>
      </div>
    );
  }

  const isLocked = pack.status === 'locked' && !pack.is_remediation;
  const isCompleted = pack.status === 'completed' || mastered;
  const isRemediation = pack.is_remediation;

  const currentQuiz = pack.quiz && pack.quiz.length > 0 ? pack.quiz[0] : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#ffffff' }}>
      {/* Header */}
      <div style={{ padding: '18px 20px 14px', borderBottom: '1px solid #f1f5f9' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              padding: '3px 10px',
              borderRadius: 20,
              background: isCompleted ? '#dcfce7' : isRemediation ? '#fee2e2' : isLocked ? '#f1f5f9' : '#dbeafe',
              color: isCompleted ? '#15803d' : isRemediation ? '#b91c1c' : isLocked ? '#64748b' : '#1d4ed8',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <span>{isCompleted ? '✓' : isRemediation ? '⚡' : isLocked ? '🔒' : '●'}</span>
            {isCompleted ? 'Mastered' : isRemediation ? 'Remediation Required' : isLocked ? 'Locked Concept' : 'Ready to Learn'}
          </span>
          <span style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', background: '#f8fafc', padding: '2px 8px', borderRadius: 6, border: '1px solid #e2e8f0' }}>
            LightGAP Mastery Hub
          </span>
        </div>
        <h2 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: '#0f172a', lineHeight: 1.35 }}>{pack.label}</h2>
        <p style={{ margin: '6px 0 0', fontSize: 13, color: '#64748b', lineHeight: 1.5 }}>{pack.summary}</p>

        {/* Lock Banner */}
        {isLocked && pack.unmet_prerequisites.length > 0 && (
          <div style={{ marginTop: 12, padding: '10px 12px', background: '#fffbeb', border: '1px solid #fef3c7', borderRadius: 8, fontSize: 12, color: '#92400e', lineHeight: 1.4 }}>
            🔒 <strong>Prerequisites required first:</strong> {pack.unmet_prerequisites.join(', ')}. Pass upstream diagnostic tests to unlock.
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #e2e8f0', background: '#f8fafc', padding: '0 16px' }}>
        <button
          onClick={() => setActiveTab('study')}
          style={{
            padding: '12px 16px',
            border: 'none',
            background: 'transparent',
            borderBottom: activeTab === 'study' ? '2px solid #2563eb' : '2px solid transparent',
            fontWeight: activeTab === 'study' ? 600 : 500,
            color: activeTab === 'study' ? '#2563eb' : '#64748b',
            cursor: 'pointer',
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <span>📚</span> Study Packet & Sources
        </button>
        <button
          onClick={() => setActiveTab('quiz')}
          style={{
            padding: '12px 16px',
            border: 'none',
            background: 'transparent',
            borderBottom: activeTab === 'quiz' ? '2px solid #2563eb' : '2px solid transparent',
            fontWeight: activeTab === 'quiz' ? 600 : 500,
            color: activeTab === 'quiz' ? '#2563eb' : '#64748b',
            cursor: 'pointer',
            fontSize: 13,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <span>🎯</span> Diagnostic Lab
        </button>
      </div>

      {/* Content Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '18px 20px' }}>
        {activeTab === 'study' && (
          <div>
            {/* Embedded YouTube Video Player */}
            {activeVideoId && (
              <div style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <h4 style={{ margin: 0, fontSize: 12, fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ color: '#ef4444' }}>▶</span> Video Lecture
                  </h4>
                  <a
                    href={`https://www.youtube.com/watch?v=${activeVideoId}`}
                    target="_blank"
                    rel="noreferrer"
                    style={{ fontSize: 11, color: '#2563eb', textDecoration: 'none', fontWeight: 600 }}
                  >
                    Open in YouTube ↗
                  </a>
                </div>
                <div
                  style={{
                    position: 'relative',
                    paddingBottom: '56.25%',
                    height: 0,
                    overflow: 'hidden',
                    borderRadius: 10,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                    background: '#000000',
                  }}
                >
                  <iframe
                    src={`https://www.youtube-nocookie.com/embed/${activeVideoId}?rel=0`}
                    title="Curated Video Lecture"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      border: 'none',
                    }}
                  />
                </div>
              </div>
            )}

            {/* YouTube Lectures List */}
            {pack.youtube && pack.youtube.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 12, fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: '#ef4444' }}>▶</span> Curated Lectures & Tutorials
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {pack.youtube.map((yt, idx) => {
                    const videoUrl = yt.url || (yt.video_id ? `https://www.youtube.com/watch?v=${yt.video_id}` : `https://www.youtube.com/results?search_query=${encodeURIComponent(yt.query)}`);
                    const isPlaying = activeVideoId === yt.video_id;

                    return (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '10px 14px',
                          borderRadius: 8,
                          border: isPlaying ? '1.5px solid #3b82f6' : '1px solid #e2e8f0',
                          background: isPlaying ? '#eff6ff' : '#f8fafc',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ flex: 1, marginRight: 10 }}>
                          <div style={{ fontSize: 13, fontWeight: 600, color: '#0f172a' }}>{yt.title}</div>
                          <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>📺 {yt.channel}</div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          {yt.video_id && (
                            <button
                              onClick={() => setActiveVideoId(yt.video_id || null)}
                              style={{
                                padding: '5px 10px',
                                fontSize: 11,
                                fontWeight: 600,
                                background: isPlaying ? '#2563eb' : '#ffffff',
                                color: isPlaying ? '#ffffff' : '#1e293b',
                                border: '1px solid #cbd5e1',
                                borderRadius: 6,
                                cursor: 'pointer',
                              }}
                            >
                              {isPlaying ? 'Playing' : '▶ Play'}
                            </button>
                          )}
                          <a
                            href={videoUrl}
                            target="_blank"
                            rel="noreferrer"
                            style={{
                              padding: '5px 10px',
                              fontSize: 11,
                              fontWeight: 600,
                              background: '#ffffff',
                              color: '#2563eb',
                              border: '1px solid #cbd5e1',
                              borderRadius: 6,
                              textDecoration: 'none',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            Watch ↗
                          </a>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Key Insights */}
            <div style={{ marginBottom: 20 }}>
              <h4 style={{ margin: '0 0 8px', fontSize: 12, fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                💡 Key Insights
              </h4>
              <div style={{ background: '#f8fafc', borderRadius: 8, padding: '12px 14px', border: '1px solid #e2e8f0' }}>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#334155', lineHeight: 1.6 }}>
                  {pack.key_takeaways.map((point, idx) => (
                    <li key={idx} style={{ marginBottom: 4 }}>{point}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Books & Papers */}
            <div style={{ marginBottom: 24 }}>
              <h4 style={{ margin: '0 0 10px', fontSize: 12, fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>📖</span> Recommended Literature & Papers
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {pack.books && pack.books.map((b, idx) => (
                  <div key={idx} style={{ padding: '10px 14px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#0f172a' }}>{b.title}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>By {b.author} {b.chapters ? `• ${b.chapters}` : ''}</div>
                  </div>
                ))}
                {pack.papers_or_docs && pack.papers_or_docs.map((p, idx) => (
                  <div key={idx} style={{ padding: '10px 14px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#0f172a' }}>{p.title}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Source: {p.source}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick action */}
            {!isCompleted && !isLocked && (
              <button
                onClick={handleManualComplete}
                style={{
                  width: '100%',
                  padding: '12px 18px',
                  background: 'linear-gradient(135deg, #16a34a, #15803d)',
                  color: 'white',
                  border: 'none',
                  borderRadius: 8,
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontSize: 13,
                  boxShadow: '0 2px 6px rgba(22, 163, 74, 0.25)',
                }}
              >
                ✓ Mark Concept as Mastered
              </button>
            )}
          </div>
        )}

        {activeTab === 'quiz' && (
          <div>
            {isLocked ? (
              <div style={{ textAlign: 'center', padding: 24, color: '#64748b' }}>
                <div style={{ fontSize: 32, marginBottom: 8 }}>🔒</div>
                <h4>Assessment Locked</h4>
                <p style={{ fontSize: 13 }}>You must master the prerequisite nodes before taking the diagnostic test for {pack.label}.</p>
              </div>
            ) : currentQuiz ? (
              <div>
                <div style={{ marginBottom: 12 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>Diagnostic Question</span>
                  <p style={{ fontWeight: 600, fontSize: 14, color: '#0f172a', margin: '6px 0 16px' }}>{currentQuiz.stem}</p>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {currentQuiz.options.map((optText, optIdx) => {
                    const isSelected = selectedOpt === optIdx;
                    let borderColor = '#e2e8f0';
                    let bgColor = '#ffffff';

                    if (gradeResult) {
                      if (optIdx === currentQuiz.correct_index) {
                        borderColor = '#16a34a';
                        bgColor = '#f0fdf4';
                      } else if (isSelected && !gradeResult.is_correct) {
                        borderColor = '#dc2626';
                        bgColor = '#fef2f2';
                      }
                    } else if (isSelected) {
                      borderColor = '#2563eb';
                      bgColor = '#eff6ff';
                    }

                    return (
                      <button
                        key={optIdx}
                        onClick={() => handleGrade(currentQuiz.stem, optIdx)}
                        disabled={submitting || (gradeResult !== null && gradeResult.is_correct)}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          padding: 12,
                          textAlign: 'left',
                          border: `1.5px solid ${borderColor}`,
                          borderRadius: 8,
                          background: bgColor,
                          cursor: 'pointer',
                          fontFamily: 'inherit',
                          fontSize: 13,
                          lineHeight: 1.4,
                        }}
                      >
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 22,
                            height: 22,
                            borderRadius: '50%',
                            background: isSelected ? '#2563eb' : '#f1f5f9',
                            color: isSelected ? '#ffffff' : '#475569',
                            fontWeight: 700,
                            marginRight: 10,
                            flexShrink: 0,
                            fontSize: 11,
                          }}
                        >
                          {String.fromCharCode(65 + optIdx)}
                        </span>
                        <span style={{ color: '#1e293b' }}>{optText}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Grade Outcome Alert */}
                {gradeResult && (
                  <div
                    style={{
                      marginTop: 16,
                      padding: 12,
                      borderRadius: 8,
                      background: gradeResult.is_correct ? '#f0fdf4' : '#fef2f2',
                      border: `1px solid ${gradeResult.is_correct ? '#bbf7d0' : '#fecaca'}`,
                      fontSize: 13,
                    }}
                  >
                    {gradeResult.is_correct ? (
                      <div style={{ color: '#15803d' }}>
                        <strong>🎉 Correct! Concept Mastered.</strong>
                        <p style={{ margin: '4px 0 0' }}>The mind map has updated and unlocked the next dependent concepts!</p>
                      </div>
                    ) : (
                      <div style={{ color: '#b91c1c' }}>
                        <strong>❌ Misconception Detected!</strong>
                        {gradeResult.attributed_misconception && (
                          <p style={{ margin: '4px 0 0' }}>
                            <strong>Diagnosed Misconception:</strong> {gradeResult.attributed_misconception}
                          </p>
                        )}
                        <p style={{ margin: '6px 0 0', fontSize: 12, color: '#7f1d1d' }}>
                          ⚡ <em>Module 4 dynamic rewrite has activated: a targeted remediation node has been injected upstream into your roadmap. Master it to repair this gap.</em>
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#64748b' }}>No diagnostic test configured for this node.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export const StudyHubPanel = NotebookLMPanel;
export default NotebookLMPanel;
