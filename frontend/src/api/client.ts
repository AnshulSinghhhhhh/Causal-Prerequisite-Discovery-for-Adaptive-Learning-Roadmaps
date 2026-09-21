/**
 * Typed client matching backend/app/models/schema.py.
 *
 * Mirror of the canonical JSON contract (Section 10). Types here are the
 * TypeScript reflection of the Pydantic models; they are kept in lockstep so a
 * schema drift between backend and frontend surfaces as a compile error rather
 * than an integration-time bug (see tests/backend/test_schema_contract.py for
 * the backend side of this guarantee).
 */

export type NodeStatus = 'completed' | 'in_progress' | 'available' | 'locked';
export type NodeType = 'root' | 'branch' | 'leaf';
export type PrerequisiteType = 'required' | 'recommended' | 'optional';
export type EdgeType = 'prerequisite' | 'remediation_link' | 'hierarchy';

export interface StudentState {
  goal_concept: string;
  time_budget_minutes: number;
}

export interface NodePayload {
  id: string;
  label?: string;
  status: NodeStatus;
  node_type?: NodeType;
  parent_id?: string | null;
  children?: string[];
  depth?: number;
  cluster_id?: string | null;
  progress?: number;
  is_dynamic_remediation: boolean;
  triggering_misconception: string | null;
}

export interface EdgePayload {
  source: string;
  target: string;
  type: EdgeType;
  confidence: number;
  prerequisite_type?: PrerequisiteType;
}

export interface GraphPayload {
  student_state: StudentState;
  nodes: NodePayload[];
  edges: EdgePayload[];
}

export interface QuizQuestionPayload {
  stem: string;
  options: string[];
  correct_index: number;
  target_node: string;
}

export interface QuizResponsePayload {
  target_node: string;
  selected_index: number;
  is_correct: boolean;
  attributed_misconception: string | null;
  match_score: number | null;
  action?: string;
}

export interface YouTubeResource {
  title: string;
  channel: string;
  query: string;
  video_id?: string;
  url?: string;
}

export interface GitHubResource {
  repo: string;
  description: string;
  url?: string;
}

export interface BookResource {
  title: string;
  author: string;
  chapters?: string;
}

export interface PaperResource {
  title: string;
  source: string;
}

export interface QuizItem {
  stem: string;
  options: string[];
  correct_index: number;
  misconception_tag?: string;
  misconception_explanation?: string;
}

export interface StudyPackPayload {
  node_id: string;
  label: string;
  status: NodeStatus;
  is_remediation: boolean;
  triggering_misconception: string | null;
  summary: string;
  key_takeaways: string[];
  youtube: YouTubeResource[];
  github: GitHubResource[];
  books: BookResource[];
  papers_or_docs: PaperResource[];
  quiz: QuizItem[];
  prerequisites: string[];
  unmet_prerequisites: string[];
}

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`LightGAP API ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const client = {
  /** Create a session and get its initial graph. */
  createSession(goalConcept: string, timeBudgetMinutes: number = 0, sessionId?: string, generateRoadmap: boolean = true) {
    return request<{ session_id: string; graph: GraphPayload }>('/graph/session', {
      method: 'POST',
      body: JSON.stringify({
        goal_concept: goalConcept,
        time_budget_minutes: timeBudgetMinutes,
        session_id: sessionId,
        generate_roadmap: generateRoadmap,
      }),
    });
  },

  /** Get NotebookLM study pack for a node. */
  getStudyPack(sessionId: string, nodeId: string) {
    return request<StudyPackPayload>(`/content/${sessionId}/${nodeId}`);
  },

  /** Mark a node as completed and unlock next nodes. */
  completeNode(sessionId: string, nodeId: string) {
    return request<GraphPayload>(`/graph/${sessionId}/complete-node`, {
      method: 'POST',
      body: JSON.stringify({ node_id: nodeId }),
    });
  },

  /** Fetch the current live graph for a session. */
  getGraph(sessionId: string) {
    return request<GraphPayload>(`/graph/${sessionId}`);
  },

  /** Assemble a DAG from a scored edge list and attach it to the session. */
  buildGraph(sessionId: string, nodeIds: string[], pairs: [string, string][], scores: number[]) {
    return request<GraphPayload>(`/graph/${sessionId}/build`, {
      method: 'POST',
      body: JSON.stringify({ node_ids: nodeIds, pairs, scores }),
    });
  },

  /** Generate the three diagnostic questions for a node. */
  generateQuiz(sessionId: string, nodeId: string, conceptText?: string, prerequisites: string[] = []) {
    return request<QuizQuestionPayload[]>('/quiz/generate', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        node_id: nodeId,
        concept_text: conceptText ?? nodeId,
        prerequisites,
      }),
    });
  },

  /** Grade a selection; triggers remediation via Module 4. */
  gradeQuiz(sessionId: string, nodeId: string, stem: string, selectedIndex: number) {
    return request<QuizResponsePayload>('/quiz/grade', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        node_id: nodeId,
        stem,
        selected_index: selectedIndex,
      }),
    });
  },

  /** Inject a remediation node upstream of target. */
  injectRemediation(sessionId: string, targetNode: string, misconceptionId: string) {
    return request<{ injected_node: string | null; target: string; locked: boolean }>(
      '/remediation/inject',
      {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          target_node: targetNode,
          misconception_id: misconceptionId,
        }),
      },
    );
  },

  /** Resolve a remediation node and restore the original edge structure. */
  resolveRemediation(sessionId: string, remediationNodeId: string) {
    return request<GraphPayload>('/remediation/resolve', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        remediation_node_id: remediationNodeId,
      }),
    });
  },

  /** Check memory decay across the graph. */
  decayCheck(sessionId: string, now?: number) {
    return request<{ decayed_nodes: string[]; graph: GraphPayload }>('/remediation/decay-check', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, now: now ?? null }),
    });
  },

  /** Find the next recommended node in the sequential learning flow. */
  getNextSuggestion(sessionId: string) {
    return request<{ next_node_id: string | null; label: string | null; level: number }>(
      `/graph/${sessionId}/next-suggestion`
    );
  },

  /** Check if a node can be studied or if prerequisites are locked. */
  canStartNode(sessionId: string, nodeId: string) {
    return request<{
      can_start: boolean;
      node_id: string;
      status: string;
      unmet_prerequisites: Array<{ id: string; label: string; status: string }>;
    }>(`/graph/${sessionId}/can-start/${nodeId}`);
  },
};