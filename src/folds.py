import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
from pandas import read_csv

from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from tp1_utils import balance_dataset, transform_dataset, plot_fold_results

# Importa o dataset
dataset = read_csv('../src/bank-marketing.csv')

# Separa atributo alvo
y = dataset['response']

# Pre-processa os dados
X = transform_dataset(dataset, y)

# Diferentes metodos a serem testados
methods = [
  {'name': 'Árvore de Decisão', 'classifier': DecisionTreeClassifier(max_depth=4)},
  {'name': 'KNN', 'classifier': KNeighborsClassifier(n_neighbors=50)},
  {'name': 'Rede Neural', 'classifier': MLPClassifier(hidden_layer_sizes=[20,10,5], max_iter=500, random_state=7)},
]

for method in methods:
  # Media dos resultados de todos os folds para cada k
  results = {"Fold":list(range(2,21)), "Accuracy":[], "Precision":[], "Recall":[], "F_Score":[]}

  for k in range(2, 21):
    # Lista de resultados das metricas
    acc = []
    prec = []
    recall = []
    fscore = []

    # K-folds de dados de treino e teste
    folds = KFold(n_splits=k, shuffle=True, random_state=7).split(X, y)

    for (train, test) in folds:
      # Separa X e y de teste baseado nos indices do fold
      X_test = X[test]
      y_test = y[test]

      # Faz balanceamento dos dados de treino
      X_train, y_train = balance_dataset(X[train], y[train])

      # Treina o modelo e prediz os dados de teste
      y_pred = method['classifier'].fit(X_train, y_train).predict(X_test)

      # Avalia resultado
      acc.append(accuracy_score(y_test, y_pred))
      prec.append(precision_score(y_test, y_pred))
      recall.append(recall_score(y_test, y_pred))
      fscore.append(fbeta_score(y_test, y_pred, beta=2))

    # Guarda o resultado de K folds
    results["Accuracy"].append(mean(acc))
    results["Precision"].append(mean(prec))
    results["Recall"].append(mean(recall))
    results["F_Score"].append(mean(fscore))

  # Exibe o grafico dos resultados desse metodo
  plot_fold_results(results, f'Desempenho por k-folds - {method["name"]}')
