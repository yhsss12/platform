-- 识别「项目已绑定团队，但 project_members 中的用户不在 team_users」的历史/异常行。
-- 执行前请备份；处理策略：将用户补录进 team_users，或从 project_members 删除，视业务而定

SELECT pm.project_id,
       p.name  AS project_name,
       p.team_id,
       pm.user_id
FROM project_members pm
JOIN projects p ON p.id = pm.project_id
WHERE p.team_id IS NOT NULL
  AND TRIM(p.team_id) <> ''
  AND NOT EXISTS (
    SELECT 1
    FROM team_users tu
    WHERE tu.team_id = p.team_id
      AND tu.user_id = pm.user_id
  );

-- 可选：仅统计数量
-- SELECT COUNT(*) FROM (
--   ... 同上 ...
-- ) q;
