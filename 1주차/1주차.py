'''
문제: Vector2D 클래스를 정의하시오. (이 클래스는 2차원 벡터를 표현한다)
요구사항
생성자에서 x, y 좌표를 받는다.
magnitude() 메서드를 만들어 벡터의 크기(유클리드 거리)를 계산한다.
공식: 루트 x2 + y2
add(other) 메서드를 만들어, 다른 벡터와의 덧셈 결과를 새로운 Vector2D 객체로 반환한다.
str__() 메서드를 오버라이드(Override)해 "Vector(x, y)" 형식으로 출력한다.

추가 문제: 벡터 클래스에 뺄셈과 내적 메서드 추가하고 테스트.
'''
import math

class Vector2D:
    def __init__(self, x, y ):
        self.x = x
        self.y = y
        
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def add(self, other):
        return Vector2D(self.x + other.x, self.y + other.y) 
    
    # 벡터 뺄셈
    def substract(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    # 벡터 내적
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    

v1 = Vector2D(3, 4)
v2 = Vector2D(1, 2)

print(v1.magnitude())  # 출력: 5.0
v3 = v1.add(v2)
print(v3)              # 출력: Vector(4, 6)

v4 = v1.substract(v2)
print(v4)              # 출력: Vector(2, 2) 

print(v1.dot(v2))     # 출력: 11