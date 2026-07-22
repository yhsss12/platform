"""
审计 action_type 固定枚举（全大写 + 下划线）。禁止将自由描述写入 action_type。

历史库中可能仍存在旧 code，ACTION_LABELS_ZH 中为旧 code 保留同义中文标签，便于列表展示与筛选。
"""

# —— 认证（平台级）——
LOGIN_SUCCESS = "LOGIN_SUCCESS"
LOGIN_FAIL = "LOGIN_FAIL"
LOGOUT = "LOGOUT"

# —— 用户管理 ——
CREATE_USER = "CREATE_USER"
DELETE_USER = "DELETE_USER"
UPDATE_USER = "UPDATE_USER"
RESET_PASSWORD = "RESET_PASSWORD"

# —— 团队 ——
CREATE_TEAM = "CREATE_TEAM"
UPDATE_TEAM = "UPDATE_TEAM"
ADD_TEAM_ADMIN = "ADD_TEAM_ADMIN"
REMOVE_TEAM_ADMIN = "REMOVE_TEAM_ADMIN"
ADD_TEAM_USER = "ADD_TEAM_USER"
REMOVE_TEAM_USER = "REMOVE_TEAM_USER"
DELETE_TEAM = "DELETE_TEAM"

# —— 项目 ——
CREATE_PROJECT = "CREATE_PROJECT"
DELETE_PROJECT = "DELETE_PROJECT"
UPDATE_PROJECT = "UPDATE_PROJECT"
ADD_PROJECT_MEMBER = "ADD_PROJECT_MEMBER"
REMOVE_PROJECT_MEMBER = "REMOVE_PROJECT_MEMBER"

# —— 数据资产 ——
IMPORT_DATA_ASSET = "IMPORT_DATA_ASSET"
DELETE_DATA_ASSET = "DELETE_DATA_ASSET"
BATCH_DELETE_DATA_ASSET = "BATCH_DELETE_DATA_ASSET"
EXPORT_DATA_ASSET = "EXPORT_DATA_ASSET"
BATCH_EXPORT_DATA_ASSET = "BATCH_EXPORT_DATA_ASSET"

# —— 设备 ——
CREATE_DEVICE = "CREATE_DEVICE"
CONNECT_DEVICE = "CONNECT_DEVICE"

# —— 任务（采集任务 / 作业；配合 resource_type 区分实体）——
CREATE_TASK = "CREATE_TASK"
UPDATE_TASK = "UPDATE_TASK"
DELETE_TASK = "DELETE_TASK"
START_TASK = "START_TASK"
STOP_TASK = "STOP_TASK"

# —— 标注结果与审核（扩展枚举，便于筛选）——
SUBMIT_LABEL_RESULT = "SUBMIT_LABEL_RESULT"
APPROVE_LABEL_REVIEW = "APPROVE_LABEL_REVIEW"
REJECT_LABEL_REVIEW = "REJECT_LABEL_REVIEW"

# —— 展示用中文（新 code）——
_ACTION_LABELS_CORE: dict[str, str] = {
    LOGIN_SUCCESS: "登录成功",
    LOGIN_FAIL: "登录失败",
    LOGOUT: "退出登录",
    CREATE_USER: "创建用户",
    DELETE_USER: "删除用户",
    UPDATE_USER: "更新用户",
    RESET_PASSWORD: "重置密码",
    CREATE_TEAM: "创建团队",
    UPDATE_TEAM: "更新团队",
    ADD_TEAM_ADMIN: "添加团队管理员",
    REMOVE_TEAM_ADMIN: "移除团队管理员",
    ADD_TEAM_USER: "添加团队成员",
    REMOVE_TEAM_USER: "移除团队成员",
    DELETE_TEAM: "删除团队",
    CREATE_PROJECT: "创建项目",
    DELETE_PROJECT: "删除项目",
    UPDATE_PROJECT: "更新项目",
    ADD_PROJECT_MEMBER: "添加项目成员",
    REMOVE_PROJECT_MEMBER: "移除项目成员",
    IMPORT_DATA_ASSET: "导入数据资产",
    DELETE_DATA_ASSET: "删除数据资产",
    BATCH_DELETE_DATA_ASSET: "批量删除数据资产",
    EXPORT_DATA_ASSET: "导出数据资产",
    BATCH_EXPORT_DATA_ASSET: "批量导出数据资产",
    CREATE_DEVICE: "添加设备",
    CONNECT_DEVICE: "绑定设备（Agent）",
    CREATE_TASK: "创建任务",
    UPDATE_TASK: "更新任务",
    DELETE_TASK: "删除任务",
    START_TASK: "启动任务/作业",
    STOP_TASK: "停止任务/作业",
    SUBMIT_LABEL_RESULT: "提交标注结果",
    APPROVE_LABEL_REVIEW: "标注审核通过",
    REJECT_LABEL_REVIEW: "标注审核驳回",
}

# —— 历史 code → 与上新动作同义的中文（仅用于展示/筛选，新写入请用上方常量）——
_LEGACY_LABEL_ALIASES: dict[str, str] = {
    "USER_LOGIN_SUCCESS": "登录成功",
    "USER_LOGIN_FAIL": "登录失败",
    "USER_LOGOUT": "退出登录",
    "USER_CREATE": "创建用户",
    "USER_DELETE": "删除用户",
    "USER_ROLE_UPDATE": "更新用户",
    "USER_PASSWORD_RESET": "重置密码",
    "PROJECT_CREATE": "创建项目",
    "PROJECT_DELETE": "删除项目",
    "PROJECT_UPDATE": "更新项目",
    "PROJECT_MEMBER_ADD": "添加项目成员",
    "PROJECT_MEMBER_REMOVE": "移除项目成员",
    "ASSET_IMPORT": "导入数据资产",
    "ASSET_DELETE": "删除数据资产",
    "ASSET_BATCH_DELETE": "批量删除数据资产",
    "ASSET_EXPORT": "导出数据资产",
    "ASSET_BATCH_EXPORT": "批量导出数据资产",
    "COLLECT_TASK_CREATE": "创建任务",
    "COLLECT_TASK_START": "启动任务/作业",
    "COLLECT_TASK_PAUSE": "暂停采集作业",
    "COLLECT_TASK_RESUME": "恢复采集作业",
    "COLLECT_TASK_STOP": "停止任务/作业",
    "LABEL_TASK_CREATE": "创建任务",
    "LABEL_RESULT_SUBMIT": "提交标注结果",
    "LABEL_REVIEW_APPROVE": "标注审核通过",
    "LABEL_REVIEW_REJECT": "标注审核驳回",
    "CONVERT_TASK_CREATE": "创建任务",
    "CONVERT_TASK_START": "启动任务/作业",
    "CONVERT_TASK_DELETE": "删除任务",
}

ACTION_LABELS_ZH: dict[str, str] = {**_ACTION_LABELS_CORE, **_LEGACY_LABEL_ALIASES}

ALL_ACTION_TYPES: tuple[str, ...] = tuple(sorted(ACTION_LABELS_ZH.keys()))
