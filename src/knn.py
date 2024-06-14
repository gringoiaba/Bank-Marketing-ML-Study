from math import floor, sqrt

from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

  # Valor maximo de k eh a raiz do tamanho do dataset de treino
  max_k = floor(sqrt(len(X_train)))

  # Garante que max_k seja impar
  if (max_k % 2) == 0:
    max_k -= 1

  # Lista valores impares possiveis para k a serem testados
  k_values = list(range(1, max_k + 1, 2))

  for k in k_values:
    # Treina o modelo e prediz os dados de teste
    y_pred = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)

    # Avalia resultado
    evaluation = eval_classification(y_test, y_pred)

    # Guarda avaliacao
    evaluation['K'] = k
    evaluations.append(evaluation)

  # Mostra o grafico das avaliacoes desse dataset
  plot_evaluations(evaluations, 'K', f'Performance KNN para diferentes valores de K ({dataset_type})')
