"""平台 / 团队登录账号流水计数表（与 Alembic 013 一致；供 metadata.create_all 幂等建表）"""
from sqlalchemy import Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from . import Base


class PlatformAccountCounter(Base):
    __tablename__ = "platform_account_counter"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    next_seq: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class TeamAccountCounter(Base):
    __tablename__ = "team_account_counter"

    team_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    next_seq: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
