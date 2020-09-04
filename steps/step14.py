import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        print(funcs) # デバック
        while funcs:
            f = funcs.pop() # 関数を取得
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs): # アスタリスクを付ける
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # アスタリスクを付けてアンパッキング
        if not isinstance(ys, tuple): # タプルではない場合の追加対応
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        # リストの要素が1つの時は最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(sefl, gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data # 修正前は、x = self.input.data
        gx = 2 * x * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)
"""
x = Variable(np.array(2.0))
y = add(x, x)
y.backward()
print(x.grad)

x = Variable(np.array(3.0))
y = add(add(x, x), x)
y.backward()
print(x.grad)
"""
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)

x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)