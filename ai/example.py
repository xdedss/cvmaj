


import numpy as np


# ai格式范例


# 需要决定吃碰杠/立直/和的时候被调用
def action(info, action_type):
    if (action_type in ['pon', 'chii', 'kan']):
        return False # 取消
    if (action_type in ['ron', 'tsumo']):
        return True # 确认
    return None # 什么都不做

# 需要出牌时被调用
def discard(info):
    return len(info.hand) - 1 # 打最后一张牌




