import matplotlib
matplotlib.use('TkAgg')  # Змініть 'TkAgg' на інший бекенд за потреби
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Згенеруємо дані
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Побудуємо графік
plt.scatter(X, y, label='Дані')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Графік згенерованих даних')
plt.show()

# Побудуємо модель лінійної регресії
X = X.reshape(-1, 1)
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

# Побудуємо графік моделі лінійної регресії
plt.scatter(X, y, label='Дані')
plt.plot(X, y_pred_linear, color='red', label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Модель лінійної регресії')
plt.show()

# Оцінимо якість моделі лінійної регресії
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)
print(f'Помилка середнього квадрату для лінійної регресії: {mse_linear}')
print(f'R^2 для лінійної регресії: {r2_linear}')

# Побудуємо модель поліноміальної регресії (наприклад, поліном 3-го ступеня)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

# Побудуємо графік моделі поліноміальної регресії
plt.scatter(X, y, label='Дані')
plt.plot(X, y_pred_poly, color='green', label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Модель поліноміальної регресії')
plt.show()

# Оцінимо якість моделі поліноміальної регресії
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)
print(f'Помилка середнього квадрату для поліноміальної регресії: {mse_poly}')
print(f'R^2 для поліноміальної регресії: {r2_poly}')
