from math import floor, log2

from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from tp1_utils import balance_dataset, transform_dataset, eval_classification, plot_evaluations

# Importa o dataset
dataset = read_csv('bank-marketing.csv')

# Separa atributo alvo
y = dataset['response']

# Pre-processa os dados
X = transform_dataset(dataset, y)

# Holdout dos dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

# Faz balanceamento dos dados de treino
datasets = {'original': (X_train, y_train), 'balanced': balance_dataset(X_train, y_train)}

# Realiza treinamento com dataset original e balanceado
for dataset_type in datasets:
  # Lista de avaliacoes
  evaluations = []

  # Escolhe o dataset (original/balanceado)
  X_train, y_train = datasets[dataset_type]

  # Valor maximo de profundidade eh o logaritmo do tamanho do dataset de treino
  max_depth = floor(log2(len(X_train)))

  # Lista valores possiveis para profundidade a serem testados
  depth_values = list(range(2, max_depth + 1))

  # Adiciona profundidade vazia, para testar o modelo sem restricao de profundidade
  depth_values.append(None)

  for depth in depth_values:
    # Treina o modelo e prediz os dados de teste
    y_pred = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train).predict(X_test)

    # Avalia resultado
    evaluation = eval_classification(y_test, y_pred)

    # Guarda avaliacao
    evaluation['Depth'] = depth
    evaluations.append(evaluation)

  # Mostra o grafico das avaliacoes desse dataset
  plot_evaluations(evaluations, 'Depth', f'Performance Árvore de Decisão para diferentes profundidades ({dataset_type})')
