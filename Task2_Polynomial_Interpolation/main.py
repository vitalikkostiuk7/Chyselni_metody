import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.exp(np.sin(x) + np.cos(x))

def f2(x):
    return 3 * np.cos(15 * x)

def interpolating_polynomial(x_val, a_k, b_k, a_0):
    result = a_0 / 2
    for k in range(1, n + 1):
        result += a_k[k - 1] * np.cos(k * x_val)
    for k in range(1, n):
        result += b_k[k - 1] * np.sin(k * x_val)
    return result

n = 17
m = 2 * n

x = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)

print("Вузли для інтерполяції:")
for x_val in x:
    print(x_val)

y1 = f1(x)
a_k10 = (1 / n) * np.sum(y1)
a_k1 = [(1 / n) * np.sum(y1 * np.cos(k * x)) for k in range(1, n + 1)]
b_k1 = [(1 / n) * np.sum(y1 * np.sin(k * x)) for k in range(1, n)]

x_values = np.linspace(0, 2 * np.pi, 1000)
y1_values = f1(x_values)
y1_interpolated = [interpolating_polynomial(x, a_k1, b_k1, a_k10) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y1_values, label='$f(x) = e^{\sin(x) + \cos(x)}$', color='blue')
plt.plot(x_values, y1_interpolated, label='Interpolating Polynomial', color='red', linestyle='dashed')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графік для $f(x) = e^{\sin(x) + \cos(x)}$ та інтерполяційний поліном')
plt.grid(True)
plt.show()

y2 = f2(x)
a_k20 = (1 / n) * np.sum(y2)
a_k2 = [(1 / n) * np.sum(y2 * np.cos(k * x)) for k in range(1, n + 1)]
b_k2 = [(1 / n) * np.sum(y2 * np.sin(k * x)) for k in range(1, n)]
x_values = np.linspace(0, 2 * np.pi, 1000)
y2_values = f2(x_values)
y2_interpolated = [interpolating_polynomial(x, a_k2, b_k2, a_k20) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y2_values, label='$f(x) = 3\cos(15x)$', color='green')
plt.plot(x_values, y2_interpolated, label='Interpolating Polynomial', color='orange', linestyle='dashed')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графік для $f(x) = 3\cos(15x)$ та інтерполяційний поліном')
plt.grid(True)
plt.xlim(0, 2 * np.pi)
plt.show()
