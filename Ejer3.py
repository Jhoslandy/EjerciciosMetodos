import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
k = 0.000095       # constante de crecimiento (1/año)
NM = 5000          # población máxima
N0 = 100           # población inicial
t0, tf = 0, 20     # intervalo de tiempo
h = 1              # paso de integración
n = int((tf - t0) / h) + 1

# Función de crecimiento
def f(t, N):
    return k * N * (NM - N)

# Inicialización
t = np.linspace(t0, tf, n)
N_heun = np.zeros(n)
N_rk4 = np.zeros(n)
N_heun[0] = N0
N_rk4[0] = N0

# Método de Heun
for i in range(n - 1):
    Ni = N_heun[i]
    predictor = Ni + h * f(t[i], Ni)
    N_heun[i + 1] = Ni + (h / 2) * (f(t[i], Ni) + f(t[i + 1], predictor))

# Método de Runge-Kutta 4to orden
for i in range(n - 1):
    Ni = N_rk4[i]
    ti = t[i]
    k1 = f(ti, Ni)
    k2 = f(ti + h / 2, Ni + h * k1 / 2)
    k3 = f(ti + h / 2, Ni + h * k2 / 2)
    k4 = f(ti + h, Ni + h * k3)
    N_rk4[i + 1] = Ni + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Imprimir resultados
print(f"{'t':>4} | {'Heun':>10} | {'Runge-Kutta':>12}")
print("-" * 32)
for i in range(n):
    print(f"{t[i]:4.0f} | {N_heun[i]:10.1f} | {N_rk4[i]:12.1f}")

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t, N_heun, label='Heun', marker='o')
plt.plot(t, N_rk4, label='Runge-Kutta 4° orden', marker='s')
plt.axhline(y=NM, color='gray', linestyle='--', label='Límite (NM=5000)')
plt.title("Comparación de métodos: Crecimiento poblacional")
plt.xlabel("Tiempo (años)")
plt.ylabel("Población N(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_metodos_poblacion.png")
plt.show()
