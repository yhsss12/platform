export type FrequencyTopicStandard = {
  topic: string;
  label: string;
  group?: string;
  min_hz?: number;
  default_min_hz: number;
};

export type TaskFrequencyConfig = {
  /** @deprecated 频率检测已固定启用，仅兼容旧数据 */
  enabled?: boolean;
  script_path?: string;
  topics: FrequencyTopicStandard[];
  camera_freq?: number;
  other_freq?: number;
};
