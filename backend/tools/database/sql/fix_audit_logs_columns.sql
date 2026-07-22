-- 手工修复：无法跑 Alembic 时，在 PostgreSQL 上补齐 audit_logs 列（与 app.models.audit_log.AuditLog 一致）。
-- 推荐：cd backend && alembic upgrade head（见 alembic/versions/005_audit_logs_align_columns.py）
-- 以下语句使用 IF NOT EXISTS，可重复执行。

ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS role VARCHAR(50);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS project_name VARCHAR(200);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS project_id VARCHAR(64);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS resource_type VARCHAR(100);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS resource_id VARCHAR(100);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS resource_name VARCHAR(255);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS detail_json JSONB;
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS user_agent TEXT;
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS ip VARCHAR(64);
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS result VARCHAR(20) NOT NULL DEFAULT 'SUCCESS';
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS action_type VARCHAR(100) NOT NULL DEFAULT 'UNKNOWN';
ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS action_label VARCHAR(200) NOT NULL DEFAULT '-';

CREATE INDEX IF NOT EXISTS ix_audit_logs_project_id ON audit_logs (project_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_action_type ON audit_logs (action_type);
CREATE INDEX IF NOT EXISTS ix_audit_logs_result ON audit_logs (result);

-- 若表为旧版（仍保留 action / detail 列），在新增 action_type、action_label 后可手工回填一次：
-- UPDATE audit_logs SET action_type = action WHERE action IS NOT NULL;
-- UPDATE audit_logs SET action_label = LEFT(COALESCE(NULLIF(TRIM(COALESCE(detail, '')), ''), NULLIF(TRIM(COALESCE(action, '')), ''), '-'), 200);
