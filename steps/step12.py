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
            x, y = f.input, f.output # 関数の入出力を取得
            x.grad = f.backward(y.grad) # backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator) # 1つ前の関数をリストに追加
                print(funcs) # デバック

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

class Add(Function):
    def forward(self, x0, x1):
        # x0, x1 = xs
        y = x0 + x1
        return (y,)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0, x1):
    return Add()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
# f = Add()
# y = f(x0, x1)
y = add(x0, x1)
print(y.data)
