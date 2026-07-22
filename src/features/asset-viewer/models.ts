export interface Episode {
  name: string;
  path: string;
}

export interface Topic {
  name: string;
  type: string;
}

export interface ViewportState {
  topic: string | null;
  frame: number;
}

export interface AgentLogEntry {
  timestamp: string;
  message: string;
}

export interface LabelRunnerState {
  selectedEpisode: string | null;
  currentFrame: number;
  maxFrame: number;
  isPlaying: boolean;
  speed: number; // 0.5, 1, 2
  viewportTopics: (string | null)[];
  agentLogs: AgentLogEntry[];
  isAgentRunning: boolean;
}


