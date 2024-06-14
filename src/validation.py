from numpy import mean
from pandas import read_csv

import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from tp1_utils import balance_dataset, transform_dataset, plot_confusion_matrix

# Importa o dataset
dataset = read_csv('bank-marketing.csv')

# Separa atributo alvo
y = dataset['response']

# Pre-processa os dados
X = transform_dataset(dataset, y)

# Diferentes metodos a serem avaliados
methods = [
  {'name': 'Árvore de Decisão', 'classifier': DecisionTreeClassifier(max_depth=4), 'k': 7, 'scores': []},
  {'name': 'KNN', 'classifier': KNeighborsClassifier(n_neighbors=50), 'k': 4, 'scores': []},
  {'name': 'Rede Neural', 'classifier': MLPClassifier(hidden_layer_sizes=[20,10,5], max_iter=500, random_state=7), 'k': 7, 'scores': []},
]

for method in methods:
  # Lista dos resultados de cada fold
  scores = []
  tp = []
  tn = []
  fp = []
  fn = []

  # K-folds de dados de treino e teste
  folds = KFold(n_splits=method['k'], shuffle=True, random_state=7).split(X, y)

  for (train, test) in folds:
    # Estatisticas da matriz de confusao desse fold
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Separa X e y de teste baseado nos indices do fold
    X_test = X[test]
    y_test = y[test]

    # Faz balanceamento dos dados de treino
    X_train, y_train = balance_dataset(X[train], y[train])

    # Treina o modelo e prediz os dados de teste
    y_pred = method['classifier'].fit(X_train, y_train).predict(X_test)

    # Calcula os parâmetros TP, TN, FP e FN de cada fold
    for prediction, reality in zip(y_pred, y_test):
      if prediction == reality:
        if prediction:
          true_positives += 1
        else:
          true_negatives += 1
      else:
        if prediction:
          false_positives += 1
        else:
          false_negatives += 1
    
    # Adiciona estatisticas a lista de resultados
    tp.append(true_positives)
    tn.append(true_negatives)
    fp.append(false_positives)
    fn.append(false_negatives)

    # Avalia e guarda resultado do score
    method['scores'].append(fbeta_score(y_test, y_pred, beta=4))

  # Monta matriz de confusao com a media dos resultados de todos os folds
  confusion_matrix = [
    [mean(tp), mean(fn)],
    [mean(fp), mean(tn)],
  ]

  # Exibe o plot da matriz de confusao desse metodo
  plot_confusion_matrix(confusion_matrix, f'Matriz de Confusão - {method["name"]}')
  
# Monta lista de scores e labels para o boxplot
data = [method['scores'] for method in methods]
labels = [method['name'] for method in methods]

# Cria o boxplot, adiciona labels e título, e exibe
fig, ax = plt.subplots()
ax.boxplot(data, labels=labels)
ax.set_ylabel('F4-Score')
ax.set_xlabel('Método')
ax.set_title('Métricas de desempenho')
plt.show()
