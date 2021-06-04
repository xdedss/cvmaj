

# 让我测试下你工作正不正常


import sys
from cvmaj import MajInfo

def testAI(moduleName):
    m = __import__(moduleName, fromlist=[''])    
    if (not('discard' in dir(m) and 'action' in dir(m))):
        raise Exception('missing  implementation')
    info = MajInfo()
    info.hand = ['8p', '3m', '4m', '5m', '7m', '8m', '5s', '6s', '6s', '6s', '7s', '8s', '8s', '5p']
    info.discard = ['2m', '3m']
    discardIndex = m.discard(info)
    print('discard %d' % discardIndex)
    

if (len(sys.argv) > 1):
    testAI('ai.' + sys.argv[1])

