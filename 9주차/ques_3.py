class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y,):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy    

apple=200
apple_num = 3
apple_tax=1.05

snack=200
snack_num=2
snack_tax=1.1

#계층들
mul_apple_layer = MulLayer()
mul_apple_tax_layer = MulLayer()

mul_snack_layer = MulLayer()
mul_snack_tax_layer = MulLayer()

add_apple_snack_layer = AddLayer()


#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
apple_tax_price = mul_apple_tax_layer.forward(apple_price, apple_tax)

snack_price = mul_snack_layer.forward(snack, snack_num)
snack_tax_price = mul_snack_tax_layer.forward(snack_price, snack_tax)

price = add_apple_snack_layer.forward(apple_tax_price, snack_tax_price)

# 역전파
dprice = 1  # 최종 출력 price를 기준으로 미분값은 1

dapple_tax_price, dsnack_tax_price = add_apple_snack_layer.backward(dprice)
dapple_price, dapple_tax = mul_apple_tax_layer.backward(dapple_tax_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

dsnack_price, dsnack_tax = mul_snack_tax_layer.backward(dsnack_tax_price)
dsnack, dsnack_num = mul_snack_layer.backward(dsnack_price)

# 결과 출력
print(f"dApple: {dapple}")
print(f"dApple_num: {dapple_num}")
print(f"dApple_tax: {dapple_tax}")
print(f"dSnack: {dsnack}")
print(f"dSnack_num: {dsnack_num}")
print(f"dSnack_tax: {dsnack_tax}")
