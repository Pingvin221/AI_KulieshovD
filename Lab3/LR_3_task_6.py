import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

# Згенеруємо дані
m = 100
X = np.linspace(-3, 3, m)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, m)
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X.reshape(-1, 1), y)

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10,include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
plot_learning_curves(polynomial_regression, X.reshape(-1, 1), y)

# Розділення даних на навчальний та валідаційний набори
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Перетворення ознак для поліноміальної регресії 2-го ступеня
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_val_poly = poly.transform(X_val.reshape(-1, 1))

# Навчання поліноміальної моделі
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Передбачення на навчальному та валідаційному наборах
y_train_pred = model.predict(X_train_poly)
y_val_pred = model.predict(X_val_poly)

# Обчислення середньоквадратичної помилки
train_error = mean_squared_error(y_train, y_train_pred)
val_error = mean_squared_error(y_val, y_val_pred)

# Візуалізація кривої навчання
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'b.', label='Навчальні дані')
plt.plot(X_val, y_val, 'r.', label='Валідаційні дані')
plt.plot(X_train, y_train_pred, 'g-', label='Прогноз на навчальних даних')
plt.plot(X_val, y_val_pred, 'y-', label='Прогноз на валідаційних даних')
plt.title(f'Поліноміальна регресія 2-го ступеня\nTrain Error: {train_error:.2f}, Validation Error: {val_error:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

