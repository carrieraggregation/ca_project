# -*- coding: cp949 -*-
_metrics = []
def record_reward(r):
    _metrics.append(float(r))
def get_metrics():
    return _metrics