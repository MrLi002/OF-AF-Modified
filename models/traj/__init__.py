"""DACT：双锚点连续轨迹光流（Dual-Anchor Continuous Trajectory）。"""
from .basis import BernsteinBasis, cycle_residual
from .estimator import DualTrajEstimator
from .query import TimeContext, TrajQuery, event_feature_at
