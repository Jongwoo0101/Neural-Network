from graphviz import Digraph


# 계산 그래프 초기화
g = Digraph(format='png')
g.attr(rankdir='LR', size='8,5')

# 노드 정의
g.node('x', 'x')
g.node('y', 'y')
g.node('mul', '×')       # 곱셈 xy
g.node('sub', '−1')      # (xy - 1)
g.node('sq', '**2')      # 제곱
g.node('z', 'z')

# 순전파
g.edges([('x', 'mul'), ('y', 'mul')])
g.edge('mul', 'sub', label='t=xy')
g.edge('sub', 'sq', label='u=t−1')
g.edge('sq', 'z', label='z=u²')

# 역전파 방향
g.edge('z', 'sq', label='∂z/∂u = 2u', color='red')
g.edge('sq', 'sub', label='∂u/∂t = 1', color='red')
g.edge('sub', 'mul', label='∂t/∂(xy)=1', color='red')
g.edge('mul', 'x', label='∂t/∂x = y', color='red')
g.edge('mul', 'y', label='∂t/∂y = x', color='red')

# 최종 편미분 결과 노드 추가
g.node('dzdx', '∂z/∂x = 2(xy−1)y', color='blue', fontcolor='blue', shape='note')
g.node('dzdy', '∂z/∂y = 2(xy−1)x', color='blue', fontcolor='blue', shape='note')

# 결과 화살표 연결
g.edge('x', 'dzdx', style='dashed', color='blue')
g.edge('y', 'dzdy', style='dashed', color='blue')

# 그래프 렌더링
g.render('345번 문제', view=True)


