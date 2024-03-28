import numpy as np
from sklearn import preprocessing

# Варіант 2
input_data = np.array([[4.1, -5.9, -3.5], [-1.9, 4.6, 3.9], [-4.2, 6.8, 6.3], [3.9, 3.4, 1.2]])

# Бінаризація даних
data_binarized = preprocessing.Binarizer(threshold=3.2).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# Виведення середнього значення та стандартного відхилення
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)

# Исключение среднего
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Масштабування MinMax
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Нормалізація даних
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1)
print("\nl2 normalized data:\n", data_normalized_l2)



