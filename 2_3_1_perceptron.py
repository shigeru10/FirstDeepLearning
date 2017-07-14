def perceptron(tmp, theta):
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

def add(x1, x2, w1, w2):
    return x1*w1 + x2*w2

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    return perceptron(add(x1, x2, w1, w2), theta)

def OR(x1, x2):
    w1, w2, theta = 3, 3, 2
    return perceptron(add(x1, x2, w1, w2), theta)

def NAND(x1, x2):
    w1, w2, theta = -2, -3, -4
    return perceptron(add(x1, x2, w1, w2), theta)

def XOR(x1, x2):
    w1, w2, theta = -3, -3, -2
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    f = AND(s1, s2)
    return f

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
