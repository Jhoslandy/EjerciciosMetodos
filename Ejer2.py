import numpy as np
import matplotlib.pyplot as plt
import os

# Crear carpeta si no existe
os.makedirs("graficas", exist_ok=True)

# Parámetros
m = 5       # masa (kg)
g = 9.81    # gravedad (m/s^2)
k = 0.05    # coef. resistencia
v0 = 0      # velocidad inicial
t0, tf = 0, 15
h = 1       # paso de tiempo
n = int((tf - t0) / h) + 1

# EDO: dv/dt = (-mg + kv^2) / m
def f(t, v):
    return (-m * g + k * v**2) / m

# Vectores de tiempo y velocidad
t = np.linspace(t0, tf, n)
v_heun = np.zeros(n)
v_rk4 = np.zeros(n)
v_heun[0] = v0
v_rk4[0] = v0

# Método de Heun
for i in range(n - 1):
    v_i = v_heun[i]
    t_i = t[i]
    predictor = v_i + h * f(t_i, v_i)
    v_heun[i + 1] = v_i + (h / 2) * (f(t_i, v_i) + f(t_i + h, predictor))

# Método de Runge-Kutta 4
for i in range(n - 1):
    v_i = v_rk4[i]
    t_i = t[i]
    k1 = h * f(t_i, v_i)
    k2 = h * f(t_i + h / 2, v_i + k1 / 2)
    k3 = h * f(t_i + h / 2, v_i + k2 / 2)
    k4 = h * f(t_i + h, v_i + k3)
    v_rk4[i + 1] = v_i + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Imprimir tabla
print(f"{'t(s)':>4} | {'Heun (m/s)':>12} | {'RK4 (m/s)':>12}")
print("-" * 34)
for i in range(n):
    print(f"{t[i]:>4.0f} | {v_heun[i]:12.4f} | {v_rk4[i]:12.4f}")

# Gráfico Método de Heun
plt.figure(figsize=(8, 5))
plt.plot(t, v_heun, label='Heun', color='blue', marker='o')
plt.title("Caída Libre - Método de Heun")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/caida_heun.png")
plt.close()

# Gráfico Método Runge-Kutta 4
plt.figure(figsize=(8, 5))
plt.plot(t, v_rk4, label='RK4', color='green', marker='s')
plt.title("Caída Libre - Runge-Kutta 4° Orden")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/caida_rk4.png")
plt.close()

# Gráfico comparativo
plt.figure(figsize=(10, 6))
plt.plot(t, v_heun, label='Heun', color='blue', marker='o')
plt.plot(t, v_rk4, label='Runge-Kutta 4', color='green', marker='s')
plt.title("Comparación: Heun vs Runge-Kutta 4° Orden")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graficas/caida_comparacion.png")
plt.close()
