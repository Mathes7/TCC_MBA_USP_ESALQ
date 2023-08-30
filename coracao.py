import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # para teste de normalidade.
from sklearn.model_selection import train_test_split #divide o modelo em modelo de teste e treino.
from sklearn.metrics import accuracy_score, confusion_matrix #para medir a acuracia e a matrix de confusão do modelo.
from sklearn.metrics import precision_score #para medir a precisão do modelo.
from sklearn.metrics import recall_score #para medir a sensibilidade do modelo.
from sklearn.ensemble import RandomForestClassifier # modelo de machine random forest.
from sklearn.model_selection import GridSearchCV # funcão para escolher os melhores parametros do modelo.
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier # modelo de machine learn knn.

# importando o banco de dados.
df = pd.read_csv('C:/Users/mathe/Downloads/heart_data.csv')
df = df.set_index('index')
df = df.drop(columns=['id'])

df.dtypes

df.isnull().sum()

# descrições estatísticas básicas.

descrição = df.describe()

correlacao = df.corr()


#%% relação de quem tem ou não problema de coração.

sns.countplot(x = df['cardio'])

x = df['cardio'].value_counts()

#%% age

plt.figure()
age = df.groupby('cardio')['age'].hist(alpha=0.40) 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Não Cardíaco', 'Cardíaco']) 
plt.title('Distribuição da idade em relação aos problemas cardíacos')
plt.xlabel('Idades')

plt.figure()
sns.boxplot(x = df['age'],y = df['cardio'], order = [0,1])

#%% gender.

man = df.loc[df['gender']==1, 'cardio']
woman = df.loc[df['gender']==2, 'cardio']

vc_man = man.value_counts()#.sort_index()
vc_woman = woman.value_counts()#.sort_index()

plt.figure()
plt.suptitle("Avalição dos problemas cardíacos em relação ao gênero")
plt.subplot(1,2,1) 
plt.bar(x = vc_man.index, height = vc_man.values)
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.title('Homens')


plt.subplot(1,2,2)
plt.bar(x=vc_woman.index, height=vc_woman.values, color='pink')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Mulheres')


#%% ap_hi.

plt.figure()
ap_hi = df.groupby('cardio')['ap_hi'].hist(alpha=0.40) 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Não Cardíaco', 'Cardíaco']) 
plt.title('Distribuição da pressão arterial sistólica em relação aos problemas cardíacos')
plt.xlabel('')

plt.figure()
sns.boxplot(x = df['ap_hi'],y = df['cardio'], order = [0,1])

#%% ap_lo.

plt.figure()
ap_lo = df.groupby('cardio')['ap_lo'].hist(alpha=0.40) 
plt.grid(False)
plt.ylim(0, 190)
plt.legend(['Não Cardíaco', 'Cardíaco']) 
plt.title('Distribuição da pressão arterial diastólica em relação aos problemas cardíacos')
plt.xlabel('')

plt.figure()
sns.boxplot(x = df['ap_lo'],y = df['cardio'], order = [0,1])

#%% cholesterol.

baixo = df.loc[df['cholesterol']==1, 'cardio']
medio = df.loc[df['cholesterol']==2, 'cardio']
alto = df.loc[df['cholesterol']==3, 'cardio']

vc_baixo = baixo.value_counts()
vc_medio = medio.value_counts()
vc_alto = alto.value_counts()

plt.figure()
plt.suptitle("Avalição dos problemas cardíacos em relação ao colesterol")
plt.subplot(1,3,1) 
plt.bar(x = vc_baixo.index, height = vc_baixo.values)
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.title('Colesterol Baixo')
plt.xticks(rotation=45)

plt.subplot(1,3,2)
plt.bar(x=vc_medio.index, height=vc_medio.values, color='pink')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Colesterol Médio')
plt.xticks(rotation=45)

plt.subplot(1,3,3)
plt.bar(x=vc_alto.index, height=vc_alto.values, color='green')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Colesterol Alto')
plt.xticks(rotation=45)

#%% gluc

baixo = df.loc[df['gluc']==1, 'cardio']
medio = df.loc[df['gluc']==2, 'cardio']
alto = df.loc[df['gluc']==3, 'cardio']

vc_baixo = baixo.value_counts()
vc_medio = medio.value_counts()
vc_alto = alto.value_counts()

plt.figure()
plt.suptitle("Avalição dos problemas cardíacos em relação a glicose")
plt.subplot(1,3,1) 
plt.bar(x = vc_baixo.index, height = vc_baixo.values)
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.title('Glicose Baixa')
plt.xticks(rotation=45)

plt.subplot(1,3,2)
plt.bar(x=vc_medio.index, height=vc_medio.values, color='pink')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Glicose Média')
plt.xticks(rotation=45)

plt.subplot(1,3,3)
plt.bar(x=vc_alto.index, height=vc_alto.values, color='green')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Glicose Alta')
plt.xticks(rotation=45)

#%% smoke.

no_smoke = df.loc[df['smoke']==0, 'cardio']
smoke = df.loc[df['smoke']==1, 'cardio']

vc_no_smoke = no_smoke.value_counts()#.sort_index()
vc_smoke = smoke.value_counts()#.sort_index()

plt.figure()
plt.suptitle("Avalição dos problemas cardíacos em relação a ser fumante")
plt.subplot(1,2,1) 
plt.bar(x = vc_no_smoke.index, height = vc_no_smoke.values)
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.title('Não fumantes')

plt.subplot(1,2,2)
plt.bar(x=vc_smoke.index, height=vc_smoke.values, color='pink')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Fumantes')

#%% alco.

no_alco = df.loc[df['alco']==0, 'cardio']
alco = df.loc[df['alco']==1, 'cardio']

vc_no_alco = no_alco.value_counts()#.sort_index()
vc_alco = alco.value_counts()#.sort_index()

plt.figure()
plt.suptitle("Avalição dos problemas cardíacos em relação ao álcool")
plt.subplot(1,2,1) 
plt.bar(x = vc_no_alco.index, height = vc_no_alco.values)
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.title('Não consome')

plt.subplot(1,2,2)
plt.bar(x=vc_alco.index, height=vc_alco.values, color='pink')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Consome')

#%% active.

no_active = df.loc[df['active']==0, 'cardio']
active = df.loc[df['active']==1, 'cardio']

vc_no_active = no_active.value_counts()#.sort_index()
vc_active = active.value_counts()#.sort_index()

plt.figure()
plt.suptitle("Avalição dos problemas cardíacos em relação ao sedentarismo")
plt.subplot(1,2,1) 
plt.bar(x = vc_no_active.index, height = vc_no_active.values)
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.title('Sedentarios')

plt.subplot(1,2,2)
plt.bar(x=vc_active.index, height=vc_active.values, color='pink')
plt.ylim()
plt.xticks([0,1], ['Não cardiaco','Cardiaco'])
plt.yticks([])
plt.title('Não sedentarios')

#%% teste de normalidade.

stats.shapiro(df['age'])
sns.displot(df['age'], kind = 'kde')
sns.displot(data=df['age'], x=df['age'], hue=df['cardio'], kind="kde")

stats.shapiro(df['gender'])
sns.displot(df['gender'], kind = 'kde')
sns.displot(data=df['gender'], x=df['gender'], hue=df['cardio'], kind="kde")

stats.shapiro(df['ap_hi'])
sns.displot(df['ap_hi'], kind = 'kde')
sns.displot(data=df['ap_hi'], x=df['ap_hi'], hue=df['cardio'], kind="kde")

stats.shapiro(df['ap_lo'])
sns.displot(df['ap_lo'], kind = 'kde')
sns.displot(data=df['ap_lo'], x=df['ap_lo'], hue=df['cardio'], kind="kde")

stats.shapiro(df['cholesterol'])
sns.displot(df['cholesterol'], kind = 'kde')
sns.displot(data=df['cholesterol'], x=df['cholesterol'], hue=df['cardio'], kind="kde")

stats.shapiro(df['gluc'])
sns.displot(df['gluc'], kind = 'kde')
sns.displot(data=df['gluc'], x=df['gluc'], hue=df['cardio'], kind="kde")

stats.shapiro(df['smoke'])
sns.displot(df['smoke'], kind = 'kde')
sns.displot(data=df['smoke'], x=df['smoke'], hue=df['cardio'], kind="kde")

stats.shapiro(df['alco'])
sns.displot(df['alco'], kind = 'kde')
sns.displot(data=df['alco'], x=df['alco'], hue=df['cardio'], kind="kde")

stats.shapiro(df['active'])
sns.displot(df['active'], kind = 'kde')
sns.displot(data=df['active'], x=df['active'], hue=df['cardio'], kind="kde")

#%% machine learning.

# em x estão todas as informações e em y estão as respostas para serem alcançadas.

x = df.drop(columns = ['cardio', 'height', 'weight'])

y = df['cardio']

#dividindo em dados de teste e treino.

[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2 )



# melhores parametros RandomForestClassifier.

params_2 = {'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'criterion':['gini', 'entropy', 'log_loss'], 
            'max_features':['sqrt', 'log2', None]}

grid_search_2 = GridSearchCV(estimator = RandomForestClassifier(),param_grid = params_2)

grid_search_2.fit(x,y)

melhores_parametros_RandomForestClassifier = grid_search_2.best_params_
melhor_resultado_RandomForestClassifier = grid_search_2.best_score_


# random forest.

random_forest_df = RandomForestClassifier(n_estimators = 19, criterion = 'entropy', max_features = 'sqrt',random_state = 0)
random_forest_df.fit(x_train, y_train)

previsoes_2 = random_forest_df.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_2 = accuracy_score(y_test, previsoes_2) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_2 = confusion_matrix (y_test,previsoes_2) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_2 = precision_score(y_test, previsoes_2, average = None) # teste da precisão do modelo.
recall_2 = recall_score(y_test, previsoes_2, average= None) # teste de sensibilidade do modelo.


# melhores parametros XGBClassifier.

params_3 = {'loss':['log_loss', 'deviance', 'exponential'],
            'criterion':['friedman_mse', 'squared_error'], 
            'max_deth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'max_features':['auto', 'sqrt', 'log2']}

grid_search_3 = GridSearchCV(estimator = XGBClassifier(),param_grid = params_3)

grid_search_3.fit(x,y)

melhores_parametros_XGBClassifier = grid_search_3.best_params_
melhor_resultado_XGBClassifier = grid_search_3.best_score_


# XGBoost.

model=XGBClassifier(criterion = 'friedman_mse', loss = 'log_loss', max_deth = 1, max_features = 'auto')
model.fit(x_train, y_train)

previsoes_3 = model.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_3 = accuracy_score(y_test, previsoes_3) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_3 = confusion_matrix (y_test,previsoes_3) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_3 = precision_score(y_test, previsoes_3, average = None) # teste da precisão do modelo.
recall_3 = recall_score(y_test, previsoes_3, average= None) # teste de sensibilidade do modelo.


# melhores parametros knn.

params_4 = {'n_neighbors':[10,20,30,40,50,60]}

grid_search_4 = GridSearchCV(estimator = KNeighborsClassifier(),param_grid = params_4)

grid_search_4.fit(x,y)

melhores_parametros_KNeighborsClassifier = grid_search_4.best_params_
melhor_resultado_KNeighborsClassifier = grid_search_4.best_score_


# modelo de decisão knn.

knn_modelo = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
knn_modelo.fit(x_train,y_train)

previsoes_4  = knn_modelo.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_4 = accuracy_score(y_test, previsoes_4) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_4 = confusion_matrix (y_test,previsoes_4) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_4 = precision_score(y_test, previsoes_4, average = None) # teste da precisão do modelo.
recall_4 = recall_score(y_test, previsoes_4, average= None) # teste de sensibilidade do modelo.








#%% pode ser util para melhorar o desempenho.

from sklearn.preprocessing import MinMaxScaler

features = df.drop(columns = ['cardio', 'height', 'weight'])

labels = df['cardio']

scaler=MinMaxScaler((-1,1))

x = scaler.fit_transform(features)

y = labels


# melhores parametros XGBClassifier.

params_3 = {'loss':['log_loss', 'deviance', 'exponential'],
            'criterion':['friedman_mse', 'squared_error'], 
            'max_deth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'max_features':['auto', 'sqrt', 'log2']}

grid_search_3 = GridSearchCV(estimator = XGBClassifier(),param_grid = params_3)

grid_search_3.fit(x,y)

melhores_parametros_XGBClassifier = grid_search_3.best_params_
melhor_resultado_XGBClassifier = grid_search_3.best_score_


# XGBoost.

model=XGBClassifier(criterion = 'friedman_mse', loss = 'log_loss', max_deth = 1, max_features = 'auto')
model.fit(x_train, y_train)

previsoes_3 = model.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_3 = accuracy_score(y_test, previsoes_3) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_3 = confusion_matrix (y_test,previsoes_3) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_3 = precision_score(y_test, previsoes_3, average = None) # teste da precisão do modelo.
recall_3 = recall_score(y_test, previsoes_3, average= None) # teste de sensibilidade do modelo.






#%% novo teste

# em x estão todas as informações e em y estão as respostas para serem alcançadas.

x = df.drop(columns = ['cardio'])

y = df['cardio']

#dividindo em dados de teste e treino.

[x_train, x_test, y_train, y_test] = train_test_split( x,y, test_size = 0.2 )

# melhores parametros RandomForestClassifier.

params_2 = {'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'criterion':['gini', 'entropy', 'log_loss'], 
            'max_features':['sqrt', 'log2', None]}

grid_search_2 = GridSearchCV(estimator = RandomForestClassifier(),param_grid = params_2)

grid_search_2.fit(x,y)

melhores_parametros_RandomForestClassifier = grid_search_2.best_params_
melhor_resultado_RandomForestClassifier = grid_search_2.best_score_


# random forest.

random_forest_df = RandomForestClassifier(n_estimators = 19, criterion = 'entropy', max_features = 'sqrt',random_state = 0)
random_forest_df.fit(x_train, y_train)

previsoes_2 = random_forest_df.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_2 = accuracy_score(y_test, previsoes_2) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_2 = confusion_matrix (y_test,previsoes_2) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_2 = precision_score(y_test, previsoes_2, average = None) # teste da precisão do modelo.
recall_2 = recall_score(y_test, previsoes_2, average= None) # teste de sensibilidade do modelo.


#%% svm

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, C = 1.0)

svm.fit(x_train, y_train)

previsoes = svm.predict(x_test)# jogando os dados de teste para tentar prever.
acuracia = accuracy_score(y_test, previsoes)# teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao = confusion_matrix (y_test,previsoes) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision = precision_score(y_test, previsoes, average = None) # teste da precisão do modelo.
recall = recall_score(y_test, previsoes, average= None) # teste de sensibilidade do modelo.




#%% redes neurais.

from sklearn.neural_network import MLPClassifier

rede_neural = MLPClassifier(max_iter=1500, verbose=True, tol=0.000010, solver='adam',activation='relu', hidden_layer_sizes=(9,9,9,9))
rede_neural.fit(x_train,y_train) # treinando o modelo para gerar resultados.
rede_neural.feature_importances_ # mostra a importância de cada variável.

previsoes_1 = rede_neural.predict(x_test) # jogando os dados de teste para tentar prever.
acuracia_1 = accuracy_score(y_test, previsoes_1) # teste de acurácia para ver se o modelo foi eficiente.
Matriz_Confusao_1 = confusion_matrix (y_test,previsoes_1) # matriz de confusão para ver a quantidade de cada informação que ficou correta.
precision_1 = precision_score(y_test, previsoes_1, average = None) # teste da precisão do modelo.
recall_1 = recall_score(y_test, previsoes_1, average = None) # teste de sensibilidade do modelo.

#Redes Neurais Artificiais (RNAs): As RNAs são modelos inspirados no 
#funcionamento do cérebro humano, compostos por camadas de neurônios 
#interconectados. Elas são utilizadas para reconhecer padrões complexos 
#nos dados e podem ser empregadas na detecção de problemas cardíacos.





#%% Teste de ruído.

import numpy as np

def ruido(df,fator_de_ruido):
    
    # Entradas:
        # 1 - df: banco de dados onde será acrescentado o ruído
        # 2 - fator_de_ruido: percentual máximo de ruído a ser inserido.
        
    # Zera o index de df, a função apresenta problemas
    # com index não sequenciais.
    df = df.reset_index(drop=True)
    
    # Nome das colunas.
    features = df.columns
    
    # Matriz de números aleatórios entre -0.5 e 0.5.
    aleatorio = np.random.random(df.shape) - 0.5
  
    # Transforma a matriz em DataFrame.
    aleatorio = pd.DataFrame(aleatorio)
    
    # Insere o nome das colunas no novo DataFrame.
    aleatorio.columns = features 

    # Média dos valores de df por coluna. É uma referência
    # para o ajuste de ordem de grandeza dos valores.
    media = df.mean(axis=0)

    # DataFrame vazio onde serão inseridos os resultados.
    nova = pd.DataFrame()
    
    # Aplicação da fórmula:
        # x = original + fr * media * aleatorio
    # Para cada coluna.
    for i in range(len(features)):
        nova[features[i]] = df.iloc[:,i] + fator_de_ruido*media[i]*aleatorio.iloc[:,i]
  
    # Retorna um DataFrame no mesmo formato que o de entrada,
    # porém com variação nos dados conforme o ruído inserido.
    return nova

#%% Importação do banco.

df = pd.read_csv('C:/Users/mathe/Downloads/heart_data.csv')
df = df.set_index('index')
df = df.drop(columns=['id'])

# Do mesmo jeito que estamos acostumados.
x = df.drop(columns = ['cardio', 'height', 'weight'])
y = df['cardio']

#%% Separação de dados de treino e teste.

from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste = train_test_split(x,y, test_size = 0.2)

#%% Modelo random forest.


from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators = 19, criterion = 'entropy', max_features = 'sqrt',random_state = 0)


# Treinamento.
modelo.fit(x_treino,y_treino)

# Predição dos dados de teste.
predicao = modelo.predict(x_teste)

# Acurácia.
from sklearn.metrics import accuracy_score
acuracia = accuracy_score(predicao,y_teste)

#%% Agora vem a novidade.

# Vou executar a função que criamos no início.
# Ela deve entrar com dois parâmetros:
# 1 - os dados que você quer inserir o ruído
# 2 - o percentual máximo do ruído.

x_ruido = ruido(x_teste,0.7)

# Queremos que modifique apenas os dados de teste,
# pois são sempre eles que estamos usando para a
# validação dos resultados.
# Compara x_treino com x_ruido, são muito parecidos,
# apenas com uma leve variação. É exatamente isso que
# queremos, vamos ver se o modelo aceita erros de medição.

# O modelo já está criado, TEMOS QUE USAR O MESMO.
# Queremos apenas investigar se esse modelo é robusto
# o suficiente pra acertar os resultados se tiver algum 
# erro de medição. Vamos testá-lo. 
predicao = modelo.predict(x_ruido)

# Por fim, vamos ver se ele acertou.
acuracia = accuracy_score(predicao,y_teste)

# Vá modificando o nível de ruído e olha o comportamento
# da acurácia. Me dê uma explicação da mudança.