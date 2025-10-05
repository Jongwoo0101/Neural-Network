def step(x):
    return 1 if x >= 0 else 0

def xor_gate(x1, x2):
    # 은닉층 뉴런 1 OR
    h1_input = x1*1 + x2*1 + (-0.5)  # w1=1, w2=1, b=-0.5
    h1 = step(h1_input)

    # 은닉층 뉴런 2 AND
    h2_input = x1*1 + x2*1 + (-1.5)  # w1=1, w2=1, b=-1.5
    h2 = step(h2_input)

    # 출력층 뉴런
    y_input = h1*1 + h2*(-2) + (-0.5)  # w1=1, w2=-2, b=-0.5
    y = step(y_input)

    return y

# XOR 진리표 확인
print("XOR 게이트 진리표 확인")
print("x1 x2 | y")
print("------------")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        y = xor_gate(x1, x2)
        print(f" {x1}  {x2} | {y}")
