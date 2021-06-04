


import numpy as np


# 断幺九，1000点！

things_i_dont_like = [
    '1s', '9s',
    '1m', '9m',
    '1p', '9p',
    '1z', '2z', '3z', '4z', '5z', '6z', '7z'
]

i2s = []
s2i = {}
for i in range(1, 10):
    i2s.append('%dm' % i)
for i in range(1, 10):
    i2s.append('%dp' % i)
for i in range(1, 10):
    i2s.append('%ds' % i)
for i in range(1, 8):
    i2s.append('%dz' % i)
for i, s in enumerate(i2s):
    s2i[s] = i

def countVec(tiles):
    res = np.zeros(len(i2s), dtype=int)
    for tile in tiles:
        res[s2i[tile]] += 1
    return res

def checkConnection(vec, i):
    res = 0
    if (i >= 27):
        return 0
    numeric = i % 9
    if (numeric >= 2):
        if (vec[i - 2] > 0 and vec[i - 1] > 0):
            res += 1
    if (numeric >= 1 and numeric <= 7):
        if (vec[i - 1] > 0 and vec[i + 1] > 0):
            res += 1
    if (numeric <= 6):
        if (vec[i + 1] > 0 and vec[i + 2] > 0):
            res += 1
    return res

def checkConnectionPotential(vec, i, allow19=True):
    res = 0
    if (i >= 27):
        return 0
    numeric = i % 9
    if (numeric >= 1):
        if (vec[i - 1] > 0 and (allow19 or (numeric - 1 >= 1))):
            res += 1
    if (numeric >= 2):
        if (vec[i - 2] > 0 and (allow19 or (numeric - 2 >= 1))):
            res += 1
    if (numeric <= 7):
        if (vec[i + 1] > 0 and (allow19 or (numeric + 1 <= 7))):
            res += 1
    if (numeric <= 6):
        if (vec[i + 2] > 0 and (allow19 or (numeric + 2 <= 7))):
            res += 1
    return res

# 需要决定吃碰杠/立直/和的时候被调用
def action(info, action_type):
    if (action_type in ['chii', 'kan']):
        return False
    if (action_type in ['pon']):
        return True
    if (action_type in ['ron', 'tsumo']):
        return True # 确认
    return None

# 需要出牌时被调用
def discard(info):
    for i, name in enumerate(info.hand):
        # 检查手牌
        if (name in things_i_dont_like):
            return i
    # 打完了怎么办呢
    counts = countVec(info.hand)
    discardTendency = np.zeros(len(i2s))
    for i in range(len(i2s)):
        if (counts[i] > 0):
            # 检查顺子
            discardTendency[i] -= checkConnection(counts, i) * 5
            # 
            discardTendency[i] -= checkConnectionPotential(counts, i) * 1
            # 检查对子
            discardTendency[i] -= (counts[i] - 1) * 5
        else:
            discardTendency[i] = -np.inf # 没法打
    i2discard = np.argmax(discardTendency)
    print(discardTendency)
    print('discard index %d' % i2discard)
    for i, name in enumerate(info.hand):
        if (name == i2s[i2discard]):
            return i
    return None



