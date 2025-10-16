# 추가문제
from graphviz import Digraph

# 계산 그래프 초기화
g = Digraph(format='png')
g.attr(rankdir='LR', size='8,5')

# 노드 정의
g.node('x', 'x')
g.node('y', 'y')
g.node('mul1', '×')          # xy
g.node('mul2', '×2')         # 2xy
g.node('sub', '−1')          # (2xy - 1)
g.node('sq', '**2')          # 제곱
g.node('add1', '+1')         # +1
g.node('z', 'z')

# 순전파
g.edge('x', 'mul1')
g.edge('y', 'mul1')
g.edge('mul1', 'mul2', label='t = xy')
g.edge('mul2', 'sub', label='u = 2t')
g.edge('sub', 'sq', label='v = 2xy−1')
g.edge('sq', 'add1', label='w = v²')
g.edge('add1', 'z', label='z = w + 1')

# 역전파 (빨간색)
g.edge('z', 'add1', label='∂z/∂w = 1', color='red')
g.edge('add1', 'sq', label='∂w/∂v = 2v', color='red')
g.edge('sq', 'sub', label='∂v/∂(2xy) = 1', color='red')
g.edge('sub', 'mul2', label='∂(2xy)/∂(xy) = 2', color='red')
g.edge('mul2', 'mul1', label='∂(xy)/∂x, ∂(xy)/∂y', color='red')
g.edge('mul1', 'x', label='∂(xy)/∂x = y', color='red')
g.edge('mul1', 'y', label='∂(xy)/∂y = x', color='red')

# 최종 편미분 결과 노드 (파란색)
g.node('dzdx', '∂z/∂x = 4y(2xy−1)', color='blue', fontcolor='blue', shape='note')
g.node('dzdy', '∂z/∂y = 4x(2xy−1)', color='blue', fontcolor='blue', shape='note')

# 결과 화살표 연결
g.edge('x', 'dzdx', style='dashed', color='blue')
g.edge('y', 'dzdy', style='dashed', color='blue')

# 그래프 렌더링
g.render('추가문제', view=True)