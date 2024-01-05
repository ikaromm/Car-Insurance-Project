# Car-Insurance-Project


#### Objetivo do Projeto
O projeto focou na construção de um modelo preditivo de machine learning para prever a probabilidade de reivindicações de seguro de veículos (is_claim) com base em um conjunto de características do veículo e do titular da apólice.

#### Abordagem e Análise

Preparação e Exploração dos Dados:

Carreguei e explorei os conjuntos de dados train.csv e test.csv.
Realizei a descrição e verificação de valores nulos nos conjuntos de dados.
Executei a análise exploratória de dados, incluindo a visualização de distribuições de características importantes.
Pré-Processamento dos Dados:

Fiz a codificação de variáveis categóricas e a normalização de características numéricas.
Lidei com questões de desbalanceamento de classes, essenciais para um modelo de classificação eficaz.

#### Construção do Modelo:

Escolhi a regressão logística como nosso modelo inicial, considerando sua simplicidade e eficácia em problemas de classificação binária.
Implementamos a regularização e a seleção de características para otimizar o desempenho do modelo.

#### Otimização de Hiperparâmetros:

Realizei uma busca em grade para encontrar a melhor combinação de hiperparâmetros, incluindo C, penalty, solver, max_iter, e tol.

#### Avaliação do Modelo:

Usei métricas como precisão, recall, F1-score e a matriz de confusão para avaliar o modelo.
Identificamos o desafio de prever corretamente a classe minoritária (is_claim = 1) e implementei estratégias para melhorar o recall desta classe.

#### Aplicação do Modelo no Conjunto de Teste:

Preparei o conjunto de teste com as mesmas transformações do conjunto de treino.
Apliquei o modelo treinado para fazer previsões no conjunto de teste e criei um arquivo CSV com os resultados.