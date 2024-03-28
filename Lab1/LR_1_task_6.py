import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Завантажимо дані
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділимо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42)

# Підготуємо класифікатор
svm_classifier = SVC(kernel='linear', C=1.0)

# Підіб'ємо модель
svm_classifier.fit(X_train, y_train)

# Прогнозуємо класи на тестовому наборі
y_pred_svm = svm_classifier.predict(X_test)

# Розрахуємо показники якості класифікації
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='micro')
recall_svm = recall_score(y_test, y_pred_svm, average='micro')
f1_score_svm = f1_score(y_test, y_pred_svm, average='micro')
confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print("Машини опорних векторів (SVM):")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_score_svm)
print("Confusion Matrix:")
print(confusion_matrix_svm)