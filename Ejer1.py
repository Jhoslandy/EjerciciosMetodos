import numpy as np
import matplotlib.pyplot as plt

# Parámetros
alpha = 0.8
k = 60
nu = 0.25
A0 = 1
h = 1
t_final = 30
n = int(t_final / h)

def f(t, A):
    return alpha * A * (1 - (A / k)**nu)

# Heun
t_vals = [0]
A_vals_heun = [A0]
A = A0

for i in range(n):
    t = i * h
    y_euler = A + h * f(t, A)
    A = A + (h / 2) * (f(t, A) + f(t + h, y_euler))
    A_vals_heun.append(A)
    t_vals.append(t + h)

# RK4
A_vals_rk4 = [A0]
A = A0

for i in range(n):
    t = i * h
    k1 = h * f(t, A)
    k2 = h * f(t + h/2, A + k1/2)
    k3 = h * f(t + h/2, A + k2/2)
    k4 = h * f(t + h, A + k3)
    A = A + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    A_vals_rk4.append(A)

# Gráficas
plt.plot(t_vals, A_vals_heun, label="Heun")
plt.title("Método de Heun")
plt.xlabel("t (días)")
plt.ylabel("Área A(t)")
plt.grid(True)
plt.savefig("graficas/heun.png")
plt.clf()

plt.plot(t_vals, A_vals_rk4, label="Runge-Kutta 4", color='orange')
plt.title("Método de Runge-Kutta 4° orden")
plt.xlabel("t (días)")
plt.ylabel("Área A(t)")
plt.grid(True)
plt.savefig("graficas/rk4.png")
