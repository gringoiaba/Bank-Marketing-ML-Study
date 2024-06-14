# Trabalho Prático 1: Treinamento e Avaliação de Modelos de Aprendizado Supervisionado

Esse trabalho consiste na implementação de um pipeline de avaliação de aprendizado supervisionado entre 3 modelos: Árvores de Decisão, KNN e Redes Neurais.

## Grupo 3
- Eduardo Aurélio H. Duarte (311408)
- Lucas P. Pons (312430)
- Paola A. Andrade (306031)
- Vitoria Lentz(301893)

## Implementação

A implementação foi realizada utilizando a linguagem `python 3`, utilizando a biblioteca `scikit-learn` para implementação dos algoritmos de aprendizado, além das bibliotecas `numpy`, `pandas`, `seaborn` e `matplotlib` para auxílio e geração de gráficos.

Essas bibliotecas podem ser instaladas utilizando o gerenciador de pacotes `pip`. O arquivo `requirements.txt` contém a lista de dependências que podem ser instaladas diretamente utilizando o comando:

```pip install -r requirements.txt```

## Arquivos

Todos os arquivos de código `*.py` a seguir podem ser executados utilizando o comando `python3 <filename>`, por exemplo:

```python3 validation.py```

### Otimização de hiperparâmetros
Esses arquivos são utilizados para testar e plotar o desempenho de cada algoritmo variando seus hiperparâmetros. (Esses foram executados e seus resultados já estão inseridos nos próximos códigos, não é necessário executar para obter a validação e comparação final).

- `decision_tree.py`: Otimização da profundidade de corte da árvore de decisão
- `knn.py`: Otimização do valor de k para o KNN
- `neural_network.py`: Otimização da configuração das camadas ocultas da rede neural

### Validação com K-folds
Esses arquivos são utilizados para validar e comparar o desempenho dos 3 algoritmos, utilizando os melhores hiperparâmetros encontrados pelos códigos acima.

- `folds.py`: Teste de diferentes valores de k folds, para os 3 algoritmos, plotando gráficos com os resultados (já foi executado e os resultados já estão inseridos no próximo código, não é necessário executar para obter a validação e comparação final).
- `validation.py`: O código "principal", que realmente realiza a validação cruzada com k-folds entre os 3 algoritmos, utilizando os valores otimizados para a quantidade de folds e hiperparâmetros dos algoritmos, obtidos a partir da execução e análise dos gráficos gerados pelos códigos anteriores.

### Auxiliar

- `bank-marketing.csv`: Dataset utilizado para os testes, obtido no Kaggle, em formato CSV
- `tp1_utils.py`: Funções auxiliares utilizadas pelos arquivos principais acima
