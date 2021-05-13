


import numpy as np


# 断幺九，1000点！

things_i_dont_like = [
    '1s', '9s',
    '1m', '9m',
    '1p', '9p',
    '1z', '2z', '3z', '4z', '5z', '6z', '7z'
]

# 需要决定吃碰杠/立直/和的时候被调用
def action(info, action_type):
    if (action_type in ['pon', 'chii', 'kan']):
        return None
    if (action_type in ['ron', 'tsumo']):
        return True # 确认
    return None # 什么都不做

# 需要出牌时被调用
def discard(info):
    for i, name in enumerate(info.hand):
        # 检查手牌
        if (name in things_i_dont_like):
            return i
    # 打完了怎么办呢
    return None



