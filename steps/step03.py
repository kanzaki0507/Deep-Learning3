import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data # データを取り出す
        y = self.forward(x) # 実際の計算
        output = Variable(y) # Variableとして返す
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
print(a.data)
b = B(a)
print(b.data)
y = C(b)
print(y.data)