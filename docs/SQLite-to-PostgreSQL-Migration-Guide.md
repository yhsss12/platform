# SQLite 迁移到 PostgreSQL 完整指南

适用于：Flask + SQLAlchemy + Flask-Migrate + React 前端

---

## 0. 前置确认

请先确认并告诉我：
- **操作系统**：Windows / macOS / Linux（本指南会给出三种系统的命令）
- **当前 SQLite 文件路径**：例如 `./instance/your.db` 或 `backend/dev.db`
- **是否必须保留现有数据**：是 → 走 4 的数据迁移；否 → 可只做建表 + 新库

---

## 1. PostgreSQL 安装

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
# 查看版本与集群
psql --version
pg_lsclusters
# 若集群在非默认端口（如 5433），后续连接时用 -p 5433
```

### macOS (Homebrew)

```bash
brew install postgresql@15
brew services start postgresql@15
# 可选：加入 PATH
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
```

### Windows

- 从 [PostgreSQL 官网](https://www.postgresql.org/download/windows/) 下载安装包，运行安装程序。
- 安装时记住设置的 **postgres 用户密码**，并勾选 “Stack Builder” 若需要额外组件。
- 安装完成后在开始菜单打开 “SQL Shell (psql)” 或使用 `psql` 命令。

---

## 2. 数据库和用户创建

在已安装 PostgreSQL 的机器上执行（以下以 Linux 为例；Windows/macOS 用图形工具或 psql 同理）。

### 2.1 以超级用户进入 psql

**Linux：**

```bash
sudo -u postgres psql
# 若集群在 5433 端口：
sudo -u postgres psql -p 5433
```

**macOS：** 当前系统用户通常可直接连，无需密码：

```bash
psql postgres
```

**Windows：** 打开 “SQL Shell (psql)”，按提示输入 postgres 密码。

### 2.2 创建用户和数据库并授权

在 psql 中执行（请按需替换 `your_db_name`、`your_user`、`your_password`）：

```sql
-- 创建用户
CREATE USER your_user WITH PASSWORD 'your_password';

-- 创建数据库（属主设为该用户，便于权限管理）
CREATE DATABASE your_db_name OWNER your_user;

-- 授权（允许连接、建表、使用 schema）
GRANT ALL PRIVILEGES ON DATABASE your_db_name TO your_user;
\c your_db_name
GRANT ALL ON SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO your_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO your_user;

-- 退出
\q
```

**示例（与你提供的信息一致）：**

```sql
CREATE USER admin WITH PASSWORD 'admin123';
CREATE DATABASE eai_ide OWNER admin;
GRANT ALL PRIVILEGES ON DATABASE eai_ide TO admin;
\c eai_ide
GRANT ALL ON SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO admin;
\q
```

---

## 3. Flask 后端配置

### 3.1 安装 Python 依赖

```bash
pip install psycopg2-binary  # 同步驱动，Flask-SQLAlchemy 常用
# 若使用异步 SQLAlchemy（asyncpg），则：
# pip install asyncpg
```

建议写入 `requirements.txt`：

```
Flask
Flask-SQLAlchemy
Flask-Migrate
psycopg2-binary
```

### 3.2 修改数据库 URI 配置

**方式 A：在 `config.py` 或环境里用环境变量（推荐）**

```python
# config.py
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret')
    # 优先使用环境变量，便于开发/生产切换
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///instance/your.db'  # 默认 SQLite 回退
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
```

**方式 B：直接写死 PostgreSQL（仅用于本地试验）**

```python
SQLALCHEMY_DATABASE_URI = (
    'postgresql://admin:admin123@localhost:5432/eai_ide'
)
# 若 PostgreSQL 在 5433 端口，改为 5433
```

PostgreSQL URI 格式：

```
postgresql://用户名:密码@主机:端口/数据库名
```

### 3.3 安全管理密码（环境变量 + .env）

- 在项目根或后端目录创建 `.env`（并加入 `.gitignore`），例如：

```bash
# .env（不要提交到 Git）
DATABASE_URL=postgresql://admin:admin123@localhost:5432/eai_ide
SECRET_KEY=your-secret-key
```

- 在 Flask 里用 `os.environ.get('DATABASE_URL')` 或 `python-dotenv` 加载。
- 生产环境用系统环境变量或密钥管理服务，不要把 `.env` 部署到服务器。

### 3.4 若使用非默认端口

本机若 `pg_lsclusters` 显示端口为 **5433**，则：

```
DATABASE_URL=postgresql://admin:admin123@localhost:5433/eai_ide
```

---

## 4. 数据迁移方案

### 4.1 方案 A：Flask-Migrate 建表 + 脚本迁数据

**步骤 1：切到 PostgreSQL 并生成迁移（仅结构，不迁数据）**

- 将 `DATABASE_URL` 改为 PostgreSQL（如上面的 `eai_ide`）。
- 确保当前代码里没有依赖 SQLite 特有的语法（如 `AUTOINCREMENT` 在 PG 中用 SERIAL/IDENTITY）。

```bash
export DATABASE_URL=postgresql://admin:admin123@localhost:5432/eai_ide
flask db upgrade   # 若当前无迁移，先 flask db migrate -m "initial"
```

这样会在 PostgreSQL 里创建所有表结构。

**步骤 2：写数据迁移脚本（SQLite → PostgreSQL）**

下面脚本从 SQLite 读表，按列名写入 PostgreSQL；会处理常见类型差异（见注释）。

```python
# scripts/migrate_sqlite_to_pg.py
import os
import pandas as pd
from sqlalchemy import create_engine, text

SQLITE_URL = "sqlite:///instance/your.db"   # 改成你的 SQLite 路径
PG_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin123@localhost:5432/eai_ide")

def migrate_table(engine_sqlite, engine_pg, table_name, pk_column='id'):
    df = pd.read_sql_table(table_name, engine_sqlite)
    if df.empty:
        print(f"  {table_name}: 无数据，跳过")
        return
    # SQLite 布尔常为 0/1，PostgreSQL 为 TRUE/FALSE
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(bool)
    # 自增主键：若 PG 表用 SERIAL，插入时可不写 id，或关闭序列同步后写入
    df.to_sql(
        table_name,
        engine_pg,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=500,
    )
    # 同步序列（PostgreSQL 自增）
    with engine_pg.connect() as conn:
        conn.execute(text(f"""
            SELECT setval(
                pg_get_serial_sequence('{table_name}', '{pk_column}'),
                COALESCE((SELECT MAX({pk_column}) FROM "{table_name}"), 1)
            );
        """))
        conn.commit()
    print(f"  {table_name}: 迁移 {len(df)} 行")

def main():
    engine_sqlite = create_engine(SQLITE_URL)
    engine_pg = create_engine(PG_URL)
    # 获取 SQLite 中所有表名（按依赖顺序，如有外键可先迁主表）
    with engine_sqlite.connect() as conn:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            conn
        )['name'].tolist()
    for table in tables:
        try:
            migrate_table(engine_sqlite, engine_pg, table)
        except Exception as e:
            print(f"  {table}: 失败 - {e}")
    print("迁移结束")

if __name__ == "__main__":
    main()
```

运行前：

- 已用 Flask-Migrate 在 PostgreSQL 中建好表。
- 安装：`pip install pandas sqlalchemy psycopg2-binary`。

```bash
export DATABASE_URL=postgresql://admin:admin123@localhost:5432/eai_ide
python scripts/migrate_sqlite_to_pg.py
```

**自增主键说明**：若 PG 表主键为 `SERIAL`/`BIGSERIAL` 或 `GENERATED BY DEFAULT AS IDENTITY`，脚本里的 `setval` 会保证后续插入的 ID 不会冲突。若某表主键列名不是 `id`，把 `migrate_table(..., pk_column='你的主键列名')` 传入正确列名。

### 4.2 方案 B：纯 Python 脚本（pandas + SQLAlchemy）一次性导

不依赖 Flask-Migrate 的迁移历史，先建 PG 表再导数据。

**步骤 1：在 PostgreSQL 建表**

- 要么用 Flask-Migrate 只对 PG 做一次 `flask db migrate` + `flask db upgrade`；
- 要么用 `Base.metadata.create_all(engine_pg)` 一次性建表（需先连接 PG 并导入所有 Model）。

**步骤 2：使用上面的同一脚本**

- 把 `SQLITE_URL` 和 `PG_URL` 改成你的路径；
- 先建表，再运行脚本。

**常见类型与兼容处理**

| SQLite 表现     | PostgreSQL      | 处理方式 |
|----------------|-----------------|----------|
| INTEGER 自增   | SERIAL/BIGSERIAL | 建表时用 SERIAL；迁移后 setval |
| 0/1 布尔       | BOOLEAN         | 读入后用 `df[col].astype(bool)` |
| 日期时间       | TIMESTAMP       | 一般兼容；必要时 `pd.to_datetime()` |
| 文本           | TEXT/VARCHAR    | 直接写入 |

---

## 5. 验证与测试

### 5.1 验证数据是否完整

- **行数对比**（在 SQLite 和 PostgreSQL 各自执行）：

```sql
-- SQLite
SELECT 'users' AS tbl, COUNT(*) FROM users
UNION ALL SELECT 'posts', COUNT(*) FROM posts;

-- PostgreSQL
SELECT 'users' AS tbl, COUNT(*) FROM users
UNION ALL SELECT 'posts', COUNT(*) FROM posts;
```

- 用脚本逐表对比行数或抽样几条主键数据也可。

### 5.2 测试 Flask API（CRUD）

- 启动应用并确保 `DATABASE_URL` 指向 PostgreSQL：

```bash
export DATABASE_URL=postgresql://admin:admin123@localhost:5432/eai_ide
flask run
```

- 用 Postman/curl 或前端调用：创建、查询、更新、删除，确认无报错且数据一致。

### 5.3 常见错误与排查

| 现象           | 可能原因           | 处理 |
|----------------|--------------------|------|
| 连接失败       | 端口/主机/密码错误 | 检查 `DATABASE_URL`、pg_hba.conf、防火墙 |
| permission denied | 用户无权限     | 重新执行 2.2 的 GRANT |
| relation "xxx" does not exist | 表未建 | 先 `flask db upgrade` 或 create_all |
| 序列/自增冲突  | 插入后未 setval    | 对自增表执行 setval（见 4.1 脚本） |
| SSL 相关       | 要求 sslmode       | URI 加 `?sslmode=disable`（仅开发） |

---

## 6. 回滚方案（快速切回 SQLite）

- **配置回退**：把 `DATABASE_URL` 改回 SQLite 即可，例如：

```bash
# .env 或 export
DATABASE_URL=sqlite:///instance/your.db
```

- **不删 PostgreSQL**：保留 PG 库和用户，需要时再改回 `DATABASE_URL` 即可。
- **保留 SQLite 文件**：迁移前复制一份 `your.db` 到 `your.db.backup`，出问题可直接用备份文件 + SQLite 的 `DATABASE_URL` 回退。

---

## 快速检查清单

- [ ] PostgreSQL 已安装并可连接（`psql -h localhost -U admin -d eai_ide`）
- [ ] 已创建数据库 `eai_ide` 和用户 `admin` 并授权
- [ ] 已安装 `psycopg2-binary`，Flask 中 `SQLALCHEMY_DATABASE_URI` 使用环境变量
- [ ] 密码写在 `.env` 且 `.env` 已加入 `.gitignore`
- [ ] 在 PG 执行过迁移或 create_all 建表
- [ ] 若需保留数据：已运行数据迁移脚本并做 setval
- [ ] 用 API 做过 CRUD 验证
- [ ] 知道如何通过改 `DATABASE_URL` 回退到 SQLite

若你提供：操作系统、当前 SQLite 路径、是否必须保留数据，我可以按你的项目再精简成「只给你当前环境要执行的那几条命令」的版本。若你实际用的是 **eai-ide（FastAPI + Alembic）**，我也可以基于现有代码给一份针对 eai-ide 的迁移步骤（含 `.env` 和 Alembic 的修改）。
