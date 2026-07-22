# 数据库工具

本目录保存需要人工执行的数据库初始化、修复和验证脚本。

- `verify_postgres.py`、`verify_postgres_inserts.py`：只读或回滚式验证。
- `init_user.py`、`alter_bigint_columns.py`：会修改数据库。
- `delete_all_projects.py`：会级联删除项目数据和关联对象，必须显式确认。
- `init_llm_providers_tables.py`：已废弃的兼容提示入口。
- `sql/`：无法运行 Alembic 时使用的手工 SQL，执行前必须确认目标数据库。

这些脚本不会在后端启动时自动执行。
