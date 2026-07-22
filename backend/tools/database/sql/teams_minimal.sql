-- 团队相关最小 DDL（手工/应急参考）。
-- 规范路径：在同一 PostgreSQL 上执行 `alembic upgrade head`（revision 009_teams_minimal）
-- 应用启动时仍会 create_all 新表，但已存在库的列/约束以 Alembic 为准

CREATE TABLE IF NOT EXISTS teams (
    id VARCHAR(128) NOT NULL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    code VARCHAR(64) NOT NULL,
    description TEXT,
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_by VARCHAR(128)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_teams_code ON teams (code);
CREATE INDEX IF NOT EXISTS idx_teams_status ON teams (status);
CREATE INDEX IF NOT EXISTS idx_teams_updated ON teams (updated_at);

CREATE TABLE IF NOT EXISTS team_admins (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(128) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_by VARCHAR(128),
    CONSTRAINT uq_team_admins_team_user UNIQUE (team_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_team_admins_team_id ON team_admins (team_id);
CREATE INDEX IF NOT EXISTS idx_team_admins_user ON team_admins (user_id);

-- 已存在的 projects 表补列（若列已存在请忽略错误）
ALTER TABLE projects ADD COLUMN IF NOT EXISTS team_id VARCHAR(128) NULL;
CREATE INDEX IF NOT EXISTS idx_projects_team_id ON projects (team_id);

CREATE TABLE IF NOT EXISTS team_users (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(128) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_by VARCHAR(128),
    CONSTRAINT uq_team_users_team_user UNIQUE (team_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_team_users_team_id ON team_users (team_id);
CREATE INDEX IF NOT EXISTS idx_team_users_user ON team_users (user_id);
