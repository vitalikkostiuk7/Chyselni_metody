import matplotlib.pyplot as plt
import numpy as np


def lagrange(x, y, xi):
    n = len(x)
    result = 0
    for j in range(n):
        term = y[j]
        for k in range(n):
            if k != j:
                term *= (xi - x[k]) / (x[j] - x[k])
        result += term
    return result


def newton_coeffs(x, y):
    n = len(x)
    coeffs = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        coeffs[i][0] = y[i]
    for k in range(1, n):
        for j in range(n - k):
            delta_y = coeffs[j + 1][k - 1] - coeffs[j][k - 1]
            delta_x = x[j + k] - x[j]
            coeffs[j][k] = delta_y / delta_x
    result = []
    for i in range(n):
        result.append(coeffs[0][i])
    return result


def newton_polynomial(x, coeffs, xi):
    n = len(x)
    result = coeffs[0]
    for i in range(1, n):
        term = coeffs[i]
        for j in range(i):
            term *= (xi - x[j])
        result += term
    return result


def runge_function(x):
    return 1 / (1 + 25 * x ** 2)


def ln_function(x):
    return np.log(x + 2)

def equidistant_nodes(a, b, n):
    delta_x = (b - a) / n
    return [a + i * delta_x for i in range(n + 1)]


def chebyshev_nodes(a, b, n):
    nodes = [(a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) / (2 * (n + 1)) * np.pi) for k in range(n + 1)]
    return nodes


a, b = -1, 1
n_values = [2, 5, 10]
m = 100


for n in n_values:
    equidistant_x = equidistant_nodes(a, b, n)
    chebyshev_x = chebyshev_nodes(a, b, n)

    delta_x = (b - a) / (m - 1)
    xi = [a + i * delta_x for i in range(m)]

    runge_y = [runge_function(x) for x in xi]

    equidistant_y = [runge_function(x) for x in equidistant_x]
    lagrange_equidistant_y = [lagrange(equidistant_x, equidistant_y, x) for x in xi]

    chebyshev_y = [runge_function(x) for x in chebyshev_x]
    lagrange_chebyshev_y = [lagrange(chebyshev_x, chebyshev_y, x) for x in xi]

    newton_coeffs_equidistant = newton_coeffs(equidistant_x, equidistant_y)
    newton_coeffs_chebyshev = newton_coeffs(chebyshev_x, chebyshev_y)

    newton_equidistant_y = [newton_polynomial(equidistant_x, newton_coeffs_equidistant, x) for x in xi]

    newton_chebyshev_y = [newton_polynomial(chebyshev_x, newton_coeffs_chebyshev, x) for x in xi]


    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(xi, runge_y, label="f(x) Рунге")
    plt.plot(xi, lagrange_equidistant_y, label="Лагранж (рівновіддалені) ")
    plt.title(f"Лагранж (рівновіддалені) - n = {n}")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(xi, runge_y, label="f(x) Рунге")
    plt.plot(xi, lagrange_chebyshev_y, label="Лагранж (Чебишев)")
    plt.title(f"Лагранж (Чебишев) - n = {n}")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(xi, runge_y, label="f(x) Рунге")
    plt.plot(xi, newton_equidistant_y, label="Ньютон (рівновіддалені)")
    plt.title(f"Ньютон (рівновіддалені) - n = {n}")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(xi, runge_y, label="f(x) Рунге")
    plt.plot(xi, newton_chebyshev_y, label="Ньютон (Чебишев)")
    plt.title(f"Ньютон (Чебишев) - n = {n}")
    plt.legend()

    plt.tight_layout()
    plt.show()

for n in n_values:
    equidistant_x = equidistant_nodes(a, b, n)
    chebyshev_x = chebyshev_nodes(a, b, n)
    delta_x = (b - a) / (m - 1)
    xi = [a + i * delta_x for i in range(m)]

    ln_y = [ln_function(x) for x in xi]

    equidistant_y = [ln_function(x) for x in equidistant_x]
    lagrange_equidistant_y = [lagrange(equidistant_x, equidistant_y, x) for x in xi]

    chebyshev_y = [ln_function(x) for x in chebyshev_x]
    lagrange_chebyshev_y = [lagrange(chebyshev_x, chebyshev_y, x) for x in xi]

    newton_coeffs_equidistant = newton_coeffs(equidistant_x, equidistant_y)
    newton_coeffs_chebyshev = newton_coeffs(chebyshev_x, chebyshev_y)

    newton_equidistant_y = [newton_polynomial(equidistant_x, newton_coeffs_equidistant, x) for x in xi]

    newton_chebyshev_y = [newton_polynomial(chebyshev_x, newton_coeffs_chebyshev, x) for x in xi]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(xi, ln_y, label="f(x)")
    plt.plot(xi, lagrange_equidistant_y, label="Лагранж (рівновіддалені) ")
    plt.title(f"Лагранж (рівновіддалені) - n = {n}")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(xi, ln_y, label="f(x)")
    plt.plot(xi, lagrange_chebyshev_y, label="Лагранж (Чебишев)")
    plt.title(f"Лагранж (Чебишев) - n = {n}")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(xi, ln_y, label="f(x)")
    plt.plot(xi, newton_equidistant_y, label="Ньютон (рівновіддалені)")
    plt.title(f"Ньютон (рівновіддалені) - n = {n}")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(xi, ln_y, label="f(x)")
    plt.plot(xi, newton_chebyshev_y, label="Ньютон (Чебишев)")
    plt.title(f"Ньютон (Чебишев) - n = {n}")
    plt.legend()

    plt.tight_layout()
    plt.show()

