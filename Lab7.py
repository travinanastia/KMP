import scipy
import numpy as np
from scipy.optimize import minimize_scalar,minimize
from matplotlib import pyplot as plt

def f(x):
    return (x-2)**2 + np.sin(5*x) + 3*np.exp(-x/2)


x_min = minimize_scalar(f, bounds=(0, 5), method='bounded').x
f_min = f(x_min)

print('\nФункція однієї змінної f(x)=(x-2)^2 + sin(5x) + 3*e^(-x/2)')
print(f'Координати мінімуму: x = {x_min}, f(x) = {f_min}')

x_values = np.linspace(0, 5, 1000)
y_values = f(x_values)
plt.plot(x_values, y_values, label='Функція')
plt.scatter(x_min, f_min, color='red', label='Мінімум')
plt.title('Графік функції однієї змінної')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()


def f2(x):
    return (x[0] - 2)**2 + (x[1] + 1)**2 + np.sin(3*x[0]) + np.cos(2*x[1])

x_values = np.linspace(-5, 5, 100)
y_values = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_values, y_values)
Z = f2([X, Y])
result = minimize(f2, (0, 0), method='Nelder-Mead')
x_min = result.x
Z_min = result.fun

print('\nФункція двох змінних  f(x,y)=(x-2)^2+(y+1)^2+sin(3x)+cos(2y)')
print(f"Координати точки мінімуму: X = {x_min[0]}, Y = {x_min[1]}")
print(f"Значення функції в точці мінімуму: {Z_min}\n")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title('Графік функції двох змінних')
ax.scatter(x_min[0], x_min[1], Z_min, color='red', s=100, label='Мінімум')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(x, y)')
plt.legend()
plt.show()


fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
contour1 = ax1.contour(X, Y, Z, cmap='coolwarm',levels=20)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.plot(x_min[0], x_min[1], 'ro', label='Мінімум')  
ax1.legend()
cbar1 = plt.colorbar(contour1, ax=ax1)

ax2 = fig.add_subplot(132)
contour2 = ax2.contour(Y, Z, X, cmap='coolwarm',levels=20)
ax2.set_xlabel('Y')
ax2.set_ylabel('Z')
ax2.plot(x_min[1], Z_min, 'ro', label='Мінімум')
ax2.legend()
cbar2 = plt.colorbar(contour2, ax=ax2)

ax3 = fig.add_subplot(133)
contour3 = ax3.contour(X, Z, Y, cmap='coolwarm',levels=20)
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.plot(x_min[0], Z_min, 'ro', label='Мінімум') 
ax3.legend()
cbar3 = plt.colorbar(contour3, ax=ax3)
plt.suptitle('Карти ліній рівня функції двох змінних')
plt.show()