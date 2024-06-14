from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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

  # Configuracoes das camadas ocultas a serem testadas
  layer_settings = [[30, 15], [20, 10, 5]]

  # Limite de iteracoes a serem testados
  iteractions_values = [300, 1000]

  for layers in layer_settings:
    for iteractions in iteractions_values:
      # Treina o modelo e prediz os dados de teste
      y_pred = MLPClassifier(hidden_layer_sizes=layers, max_iter=iteractions, random_state=7).fit(X_train, y_train).predict(X_test)

      # Avalia resultado
      evaluation = eval_classification(y_test, y_pred)

      # Guarda avaliacao
      evaluation['Config'] = f'L: {layers}, I: {iteractions}'
      evaluations.append(evaluation)

  # Mostra o grafico das avaliacoes desse dataset
  plot_evaluations(evaluations, 'Config', f'Performance Rede Neural para diferentes configurações ({dataset_type})')
