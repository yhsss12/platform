#!/usr/bin/env node
/**
 * 前端轨迹选择器 mock 验证（不跑真实仿真）。
 * 验证 episode 数值排序与中文标签生成。
 */
function getEpisodeIndex(item, fallbackIndex) {
  if (typeof item.episodeIndex === 'number' && item.episodeIndex > 0) {
    return item.episodeIndex;
  }
  const source = item.fileName || item.uri || '';
  const match = source.match(/episode[_-](\d+)/i);
  if (match) return Number(match[1]);
  return fallbackIndex + 1;
}

function getTrajectoryLabel(item, index) {
  if (item.label === '代表性回放') return '代表性回放';
  const round = getEpisodeIndex(item, index);
  return `第 ${round} 轮轨迹`;
}

function normalize(items) {
  return items
    .map((item, index) => ({
      ...item,
      episodeIndex: getEpisodeIndex(item, index),
      label: getTrajectoryLabel(item, index),
    }))
    .sort((a, b) => a.episodeIndex - b.episodeIndex);
}

function mockItems(count) {
  const shuffled = [];
  for (let i = count; i >= 1; i -= 1) {
    shuffled.push({
      episodeIndex: i,
      uri: `/api/workspace/cable-threading/jobs/mock/video?episode=${i}`,
      fileName: `episode_${String(i).padStart(3, '0')}.mp4`,
    });
  }
  return shuffled;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

for (const count of [10, 30, 100]) {
  const normalized = normalize(mockItems(count));
  assert(normalized.length === count, `${count} 轮 mock 数量不对`);
  for (let i = 0; i < count; i += 1) {
    assert(normalized[i].episodeIndex === i + 1, `${count} 轮排序错误 at ${i}`);
    assert(normalized[i].label === `第 ${i + 1} 轮轨迹`, `${count} 轮标签错误 at ${i}`);
  }
  console.log(`OK mock ${count} 轮：排序与中文标签正确`);
}

console.log('10/100 轮 UI 兼容逻辑验证通过（仅 mock，未跑真实评测）');
