import sklearn
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Завантаження даних Iris
iris = load_iris()
X = iris['data']
y = iris['target']

sklearn.cluster.KMeans(n_clusters = 8, init = 'k-means++',n_init = 10, max_iter =
300, tol = 0.0001 , verbose = 0, random_state =
None, copy_x = True, algorithm = 'auto')


# Кластеризація з використанням K-Means
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Візуалізація результатів кластеризації
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Оголошення функції 'find_clusters', яка буде використовуватися для пошуку кластерів і візуалізації їх
def find_clusters(X, n_clusters, rseed = 2):
    # Створення генератора випадкових чисел з фіксованим насінням
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)
        # Обчислення нових центрів кластерів
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        # Перевірка умови зупинки алгоритму
        if np.all(centers == new_centers):
            break
        centers = new_centers
        return centers, labels
        centers, labels = find_clusters(X, 3)
        plt.scatter(X[:, 0], X[:, 1], c=labels,
                    s=50, cmap= 'viridis');
        #Виклик функції find_clusters для X з 3 кластерами та зі зміненим random_state
        centers, labels = find_clusters(X, 3, rseed=0)
        plt.scatter(X[:, 0], X[:, 1], c=labels,
                    s=50, cmap= 'viridis');
        #Використання K-Means моделі для кластеризації X з 3 кластерами і візуалізація результатів
        labels = KMeans(3, random_state=0).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=labels,
                    s=50, cmap= 'viridis');
