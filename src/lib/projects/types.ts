export type ProjectStatus = '进行中' | '已暂停' | '已归档';

export type ProjectMemberRole = 'Owner' | 'Admin' | 'Member' | 'Viewer';

export interface ProjectMember {
  id: string;
  name: string;
  role: ProjectMemberRole;
  addedAt: string; // ISO
  lastActiveAt?: string;
}

export interface ProjectTaskRef {
  id: string;
  type: '采集' | '清洗' | '标注' | '转换';
  name: string;
  status: '待处理' | '进行中' | '成功' | '失败';
  createdAt: string;
  updatedAt: string;
}

export interface ProjectDatasetRef {
  id: string;
  name: string;
  format: 'MCAP' | 'HDF5' | 'LeRobot';
  sizeBytes?: number;
  createdAt: string;
  sourceTaskId?: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  tags: string[];
  status: ProjectStatus;
  ownerId: string;
  /** 所属团队；有值时添加项目成员须先在该团队的 team_users 中 */
  teamId?: string | null;
  /**
   * 来自 GET /api/projects：当前用户在 project_members 表是否有行（受邀加入）。
   * 列表接口不展开 members 数组时，「共享项目」Tab 依赖此字段。
   */
  viewerInProjectMembers?: boolean;
  /** 当前用户是否为 projects.owner_id（与列表接口 viewer_is_project_owner 一致） */
  viewerIsProjectOwner?: boolean;
  /** 项目成员展示人数（与详情概览 project.members.length / 后端 GET members 条数一致） */
  memberCount?: number;
  members: ProjectMember[];
  tasks: ProjectTaskRef[];
  datasets: ProjectDatasetRef[];
  createdAt: string;
  updatedAt: string;
}
