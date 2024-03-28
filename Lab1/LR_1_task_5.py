import pandas as pd
df = pd.read_csv('data_metrics.csv')
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()
from sklearn.metrics import confusion_matrix
def true_positives(confusion_matrix):
    return confusion_matrix[0, 0]
def false_positives(confusion_matrix):
    return confusion_matrix[1, 0]
def false_negatives(confusion_matrix):
    return confusion_matrix[0, 1]
def true_negatives(confusion_matrix):
    return confusion_matrix[1, 1]
confusion_matrix = confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print("Істинно позитивні:", true_positives(confusion_matrix))
print("Хибні позитивні:", false_positives(confusion_matrix))
print("Невірно негативні:", false_negatives(confusion_matrix))
print("Істинно негативні:", true_negatives(confusion_matrix))

def kulieshov_accuracy_score(confusion_matrix):
    true_positives = confusion_matrix[0, 0]
    true_negatives = confusion_matrix[1, 1]
    return (true_positives + true_negatives) / (
        true_positives + true_negatives + confusion_matrix[0, 1] + confusion_matrix[1, 0]
    )
print("Точність:", kulieshov_accuracy_score(confusion_matrix))

def kulieshov_recall_score(confusion_matrix):
    true_positives = confusion_matrix[0, 0]
    possible_positives = confusion_matrix[0, 0] + confusion_matrix[0, 1]
    return true_positives / possible_positives
print("Відкликання:", kulieshov_recall_score(confusion_matrix))

def kulieshov_precision_score(confusion_matrix):
    true_positives = confusion_matrix[0, 0]
    predicted_positives = confusion_matrix[0, 0] + confusion_matrix[0, 1]
    return true_positives / predicted_positives
print("Точність:", kulieshov_precision_score(confusion_matrix))

from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)
import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
