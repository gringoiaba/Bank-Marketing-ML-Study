import matplotlib.pyplot as plt
import seaborn as sns
from numpy import concatenate
from pandas import DataFrame
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_curve, auc
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.utils import resample

# Balanceamento do dataset
def balance_dataset(X, y):
  # Determina valores do atributo alvo
  y_values = list(set(y))

  # Separa grupos do dataset pelo atributo alvo
  group_1 = X[y == y_values[0]]
  group_2 = X[y == y_values[1]]

  # Determina grupo minoritario
  if len(group_1) > len(group_2):
    minority = group_2
    majority = group_1

    y_minority = y_values[1]
    y_majority = y_values[0]
  else:
    minority = group_1
    majority = group_2

    y_minority = y_values[0]
    y_majority = y_values[1]

  # Faz o resample do grupo minoritario para o mesmo tamanho do grupo maioritario
  minority_resampled = resample(minority, n_samples=len(majority))

  # Retorna um novo dataset balanceado
  return concatenate([majority, minority_resampled]), ([y_majority] * len(majority)) + ([y_minority] * len(minority_resampled))

# Pre-processamento do dataset
def transform_dataset(dataset, y=None):
  # Funcao para codificacao correta do mes
  def month_encoder(dataset):
    return StandardScaler().fit_transform(dataset.applymap(
      lambda month: ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'].index(month)
    ))

  # Transformador com diferentes estrategias dependendo do tipo/conteudo das colunas
  transformer = make_column_transformer(
    (StandardScaler(), ['age', 'salary', 'balance', 'day', 'duration', 'pdays']), # Normalizacao min-max de atributos numericos
    (TargetEncoder(), ['job', 'campaign', 'previous']), # Target encoder para atributos com muitas categorias
    (OneHotEncoder(drop='if_binary'), ['eligible', 'marital', 'education', 'targeted', 'default', 'housing', 'loan', 'contact', 'poutcome']), # Codificacao one-hot para atributos com poucas categorias e binarios
    (FunctionTransformer(month_encoder), ['month']), # Codificacao customizada para o mes
    ('drop', ['age group', 'marital-education', 'y', 'response']), # Remocao das colunas desnecessarias
    remainder='passthrough',
  )

  return transformer.fit_transform(dataset, y)

# Imprime avaliacao de predicao
def eval_classification(actual, pred):
  # Calcula estatisticas da curva ROC
  fpr, tpr, thresholds = roc_curve(actual, pred)

  # Imprime metricas de avaliacao
  return {
    'Accuracy': accuracy_score(actual, pred),
    'Precision': precision_score(actual, pred),
    'Recall': recall_score(actual, pred),
    'F1-Score': f1_score(actual, pred),
    'F2-Score': fbeta_score(actual, pred, beta=2),
    'F4-Score': fbeta_score(actual, pred, beta=4),
    'AUC': auc(fpr, tpr),
  }

# Exibe grafico de avaliacoes
def plot_evaluations(evaluations, x_field, title):
  # Converte lista de avaliacoes para um dataframe
  df = DataFrame(evaluations)

  # Define linhas do grafico para cada metrica
  sns.lineplot(x=x_field, y='Accuracy', data=df, label='Accuracy')
  sns.lineplot(x=x_field, y='Precision', data=df, label='Precision')
  sns.lineplot(x=x_field, y='Recall', data=df, label='Recall')
  sns.lineplot(x=x_field, y='F1-Score', data=df, label='F1-Score')
  sns.lineplot(x=x_field, y='F2-Score', data=df, label='F2-Score')
  sns.lineplot(x=x_field, y='F4-Score', data=df, label='F4-Score')
  sns.lineplot(x=x_field, y='AUC', data=df, label='AUC')

  # Ajusta parametros do grafico
  sns.set_style('ticks')
  plt.rcParams.update({'font.size': 8})
  plt.title(title)
  plt.xlabel(x_field)
  plt.ylabel('')
  plt.legend(loc='lower right')
  plt.grid(True)
  sns.despine()

  # Exibe o grafico
  plt.show()

# Exibe grafico de avaliacoes do kfolds
def plot_fold_results(df, title):
  # Define linhas do grafico para cada metrica
  sns.lineplot(x='Fold', y='Recall', data=df, marker='o', color='navy', label='Recall')
  sns.lineplot(x='Fold', y='F_Score', data=df, marker='o', color='teal', label='f Score')
  sns.lineplot(x='Fold', y='Accuracy', data=df, marker='o', color='mediumspringgreen', label='Accuracy')
  sns.lineplot(x='Fold', y='Precision', data=df, marker='o', color='steelblue', label='Precision')

  # Ajusta parametros do grafico
  plt.title(title)
  plt.xlabel('Fold')
  plt.ylabel('')
  plt.legend(loc='lower right')
  plt.grid(True)
  sns.despine()

  # Exibe o grafico
  plt.show()

# Exibe a matriz de confusao graficamente
def plot_confusion_matrix(matrix, title):
  class_labels = ['Positive', 'Negative']
  plt.figure(figsize=(8, 6))
  sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
  plt.title(title)
  plt.show()

