#Comenzamos un proyecto didáctico en el que estudiaremos una base de datos de movimientos financieros para después tratar de hacer algunas predicciones empleando diferentes tipos de algoritmos y métodos de Machine Learning.

#Importamos las librerías pandas y numpy.

import pandas as pd
import numpy as np

#Importamos librerías para graficar y establecemos algunas configuraciones.

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

google = pd.read_csv("C:/Users/aleja/OneDrive/Escritorio/Market Data/Stocks/googl.us.txt") #Dentro de nuestra base de datos comenzamos con la información de google.

#print(google.head(5)) #Analizamos un poco el dataframe y observamos que tenemos 6 columnas: Fecha, Apertura, Máximo, Minimo, Cierre, y Open Interest. Esto es así para 3333 días concatenados.
#print(google.info())
#print(google.describe())

#Guardamos en dataframe original por si después necesitamos volver a él:

google_orig = google.copy()

#Nos deshacemos de la columna Open Interest, puesto que está vacía.

google = google.drop(columns=["OpenInt"])

#Además incluímos algunas columnas que pueden sernos de utilidad:

google["Diferencia"] = google["Close"] - google["Open"] #Diferencia entre el precio de cierre y el de apertura.

google["CV"] = np.where(google["Diferencia"] >= 0, "Compra", "Venta") #La nueva columna CV  marca "Compra" y "Venta"

#A continuación creamos una función que nos calcula la diferencia entre el precio de apertura de un día con el de x días anterior para todas las fechas en valores que va almacenando en una variable.
#Posteriormente trataremos de crear nuevas columnas en nuestro data frame que almacenen esta información.

def DiferenciaX(dias):

	dif = np.zeros(len(google))

	for i in range(len(google)):

		if i < dias:       #Si no hay un día con el que comparar porque la fecha sea muy temprana lo dejamos en 0.

			dif[i] = 0

		else:

			dif[i] =  google.iloc[i,1] - google.iloc[i-dias,1]

	return dif

#Almacenamos variables para 5, 25  y 50 días

diff5 = DiferenciaX(5)
diff25 = DiferenciaX(25)
diff50 = DiferenciaX(50)

#Las incluímos en el DataFrame

google["Diferencia5"] = diff5
google["Diferencia25"] = diff25
google["Diferencia50"] = diff50

#A continuación creamos una función que calcule el RSI financiero = 100-100/(1+RS):

def RSIcalcul(periodos):

	rsi = np.zeros(len(google))
	subidas=0
	bajadas=0
	variacion=0

	for i in range(len(google)):

		if i < periodos-1:  #Volvemos a dejar con un RSI = 0 aquellos días para los que no se pueda calculas por ser muy rempranos

			rsi[i]=0

		else:

				for n in range(periodos-1):

					variacion = google.iloc[i-(n+1),1] - google.iloc[i-n,1]

					if variacion <= 0:

						subidas = subidas - variacion

					else:

						bajadas = bajadas + variacion

				media_subidas = subidas/periodos
				media_bajadas = bajadas/periodos
				RS = media_subidas/media_bajadas
				RSI = 100 - (100/(1+RS))
				rsi[i] = RSI

	return rsi

#Normalmente el RSI se calcula para un periodo de 14 días, que es el que vamos a utilizar:

rsi = RSIcalcul(14)
google["RSI"] = rsi

#Podemos seguir añadiendo variables, que analicen por ejemplo el cambio en el RSI o en el Volumen de los últimos días, pero de momento nos quedamos con las que tenemos para tantear un poco los resultados.
#A continuacion tenemos que pasar las variables CV y Diferencia al día anterior, puesto que dada la información de un determinado día, son las que queremos predecir para el día siguiente.
#Con la primera podríamos emplear algoritmos de clasificación y con la segunda algoritmos de regresión.

for i in (range(len(google)-1)):

	google["CV"][i] = google["CV"][i+1]
	google["Diferencia"][i] = google["Diferencia"][i+1]

#A continuación vamos a eliminar el último elemento por no tener una categoría "CV" y "Diferencia" válida (no hay día posterior con el que comparar)
#También eliminamos los 50 primeros por no tener una categoría "Diferencia50" válida (no hay suficientes días anteriores con los que calcularla)

google = google.drop(range(0,50), axis=0)
google = google.drop([3332], axis=0)
google = google.reset_index() #Reinicializamos los índices

#A continuación definimos una función que nos separe un dataframe en el que no hay que mezclar las instancias.
#Ese es nuestro caso puesto que la secuencia de cómo están ordendas importa. Intentamos predecir en base a los días anteriores.

def split_train_test_ordered(data,test_ratio): #El ratio es el de aprendizaje y debe estar entre 0 y 1

	indices = range(len(google))
	train_set_size = int(len(data) * test_ratio)
	train_set_indices = indices[:train_set_size]
	test_set_indices = indices[train_set_size:]
	return data.iloc[train_set_indices], data.iloc[test_set_indices]

#Aplicamos la función:

train_set, test_set = split_train_test_ordered(google, 0.8) #Usamos el 80 para entrenar

#print(len(train_set), "train +", len(test_set), "test")

#Vemos la matriz de correlación: (solo mide una relación lineal)

corr_matrix = train_set.corr()
#print(corr_matrix)
#print(corr_matrix["Diferencia"].sort_values(ascending=False)) 

#No vemos mucha correlación lineal entre la variable Diferencia y las demás (previsible)
#Aún así vamos a continuar preparando los datos para aplicar algoritmos y después volveremos a intentar mejorar este asunto con la preparación de otras variables por ejemplo.

train_set_prep = train_set.drop("CV",axis=1)
train_set_prep = train_set_prep.drop("Diferencia",axis=1)
train_set_labels1 = train_set["CV"].copy() 
train_set_labels2 = train_set["Diferencia"].copy()

#print(train_set_prep.info())
#print(train_set_labels1[0:10])
#print(train_set_labels2[0:10])

#Vamos además a eliminar la columna de las fechas porque no nos interesa demasiado, ya lo tenemos indexado.

train_set_prep = train_set_prep.drop("Date",axis=1)

#Empleamos ahora "standard scaler" (normalización)

from sklearn.preprocessing import StandardScaler

print(train_set_prep.head(10))

scaler = StandardScaler().fit(train_set_prep)
train_set_prep_scaled = scaler.transform(train_set_prep)

#Al normalizarlo lo convertimos en un array y deja de ser un dataframe

#print(np.shape(train_set_prep_scaled))
#print(type(train_set_prep))
#print(type(train_set_prep_scaled))

#Empezamos probando con la regresión lineal:

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_set_prep_scaled, train_set_labels2)

#Lo probamos ahora con algunas instancias del training set:

some_data = train_set_prep_scaled[:20] #Las 20 primeras filas
some_labels = train_set_labels2[:20]

print("Predictions:", lin_reg.predict(some_data))  #Malos resultados 
print("Valores Reales:", list(some_labels))

from sklearn.metrics import mean_squared_error

#Vamos ahora a ver cómo funciona en todo el training set a través de los errores

predictions_l = lin_reg.predict(train_set_prep_scaled)
lin_mse_l = mean_squared_error(train_set_labels2, predictions_l)
lin_rmse_l = np.sqrt(lin_mse_l)

print(train_set_labels2.min())
print(train_set_labels2.max())
print(lin_rmse_l)  #Obtenemos un error medio de aproximadamente 4.17 cuando los labels varían entre -31.5 y 18.54. No está mal del todo pero tampoco podemos decir que funcione.

print(google["Diferencia"].mean()) #La media es de -0.14

#Podemos seguir profundizando y mejorando el tema de las regresiones lineales, pero vamos a cambiar de algoritmo. Usamos ahora árboles de decisión regresivos:

from sklearn.tree import DecisionTreeRegressor

#Lo entrenamos:

tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(train_set_prep_scaled, train_set_labels2)

#Lo evaluamos:

predictions_t = tree_reg.predict(train_set_prep_scaled)
lin_mse_t = mean_squared_error(train_set_labels2, predictions_t)
lin_rmse_t = np.sqrt(lin_mse_t)
print(lin_rmse_t)  #El error es cuatro veces menos, un rmse de 1.19. Puede que estemos pecando de overfitting. Empleamos para reducirlo cross validation (10 folds):

from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(tree_reg, train_set_prep_scaled, train_set_labels2,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

#Definimos una función para ver los resultados:

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores) #Efectivamente existía bastante overfitting. Vemos que el error (6.23) ha aumentado.

#Vemos el score también para la refresión lineal:

lin_scores = cross_val_score(lin_reg, train_set_prep_scaled, train_set_labels2,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)  #De hecho funciona mejor la regresión lineal (4.02).

#Probamos regresión lineal para grados superiores:

from sklearn.preprocessing import PolynomialFeatures    
from sklearn.pipeline import Pipeline                   #Facilitamos el proceso con Pipeline

polynomial_regression2 = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

lin_scores_2 = cross_val_score(polynomial_regression2, train_set_prep_scaled, train_set_labels2,
                         scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores_2 = np.sqrt(-lin_scores_2)
display_scores(lin_rmse_scores_2)           #El grado dos da un error medio de 4.22, mayor que la regresión lineal. A partir del grado 3 el error se dispara.

#Tanteamos ahora el randomforestregressor:

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(train_set_prep_scaled, train_set_labels2)

predictions_f = forest_reg.predict(train_set_prep_scaled)
lin_mse_f = mean_squared_error(train_set_labels2, predictions_f)
lin_rmse_f = np.sqrt(lin_mse_f)

print(lin_rmse_f) #Obtenemos un buen error (1.83). Reducimos un posible overfitting con cross validation:

forest_scores = cross_val_score(forest_reg, train_set_prep_scaled, train_set_labels2,
							scoring="neg_mean_squared_error", cv=10)


display_scores(forest_scores) #Había mucho Overfitting. La media del score ahora es de 23.00

#Una forma de luchar contra el overfitting podría ser alimentando el modelo con una mayor cantidad de datos (que los tenemos)
#Nos hacemos ya una idea de en qué algoritmos debemos profundizar y cuales tenemos que descartar
#En el siguiente script estudiaremos qué tal funciona el algoritmo de Suppor Vector Machine

#Aplicamos el LinealSVR Regressor:

from sklearn.svm import LinearSVR

SVR_l = LinearSVR(epsilon=1.5, random_state=42)
SVR_l.fit(train_set_prep_scaled, train_set_labels2)
SVR_l_scores = cross_val_score(SVR_l, train_set_prep_scaled, train_set_labels2, scoring = "neg_mean_squared_error", cv=10)
SVR_l_rmse_scores = np.sqrt(-SVR_l_scores)
display_scores(SVR_l_rmse_scores)

#Conseguimos un error del 4.02, el mejor hasta el momento. Probemos ahora con una regresión no lineal:

from sklearn.svm import SVR

SVR_p2 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="auto")
SVR_p2.fit(train_set_prep_scaled, train_set_labels2)
SVR_p2_scores = cross_val_score(SVR_p2, train_set_prep_scaled, train_set_labels2, scoring = "neg_mean_squared_error", cv=10)
SVR_p2_rmse_scores = np.sqrt(-SVR_p2_scores)
display_scores(SVR_p2_rmse_scores)

#Con una regresion cuadrática el error es de 4.23

SVR_p3 = SVR(kernel="poly", degree=3, C=100, epsilon=0.1, gamma="auto")
SVR_p3.fit(train_set_prep_scaled, train_set_labels2)
SVR_p3_scores = cross_val_score(SVR_p3, train_set_prep_scaled, train_set_labels2, scoring = "neg_mean_squared_error", cv=10)
SVR_p3_rmse_scores = np.sqrt(-SVR_p3_scores)
display_scores(SVR_p3_rmse_scores)

#Si nos vamos al tercer grado el eror aumenta a 5.28

#Vamos ahora a hacer una GridSearch del algoritmo LinearSVR, que es el que mejor nos ha funcionado de momento, para mejorar los parámetros:

from sklearn.model_selection import GridSearchCV

param_grid = [
     #try 12 (3×4) combinations of hyperparameters
    {'C': [0.1, 0.5, 1], 'epsilon': [0, 1.25, 1.5, 1.75, 2]},
     #then try 6 (2×3) combinations with bootstrap set as False
     ]

grid_search = GridSearchCV(SVR_l, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(train_set_prep_scaled, train_set_labels2)

print(grid_search.best_params_) #Accedemos a los mejores parámetros

#Con el código siguiente además podemos acceder al score de cada combinación de parámetros:

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#Después del Grid-search el mejor epsilon entre los parámetros probados sigue siendo 1.5. En cuanto a la C, aparentemente funciona mejor si la establecemos a 0.1
#Predecimos finalmente el funcionamiento del mejor algoritmo que tenemos hasta ahora sobre los datos del test.Preparamos los datos:

test_set_prep = test_set.drop("CV",axis=1)
test_set_prep = test_set_prep.drop("Diferencia",axis=1)
test_set_prep = test_set_prep.drop("Date",axis=1)
test_set_labels1 = test_set["CV"].copy() 
test_set_labels2 = test_set["Diferencia"].copy()
test_set_prep_scaled = scaler.transform(test_set_prep)

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(test_set_prep_scaled)

final_mse = mean_squared_error(test_set_labels2, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse) 

#Obtenemos un error de 7.84. En el futuro buscaremos mejores atributos y mejores parámetros. Además utilizaremos una mayor cantidad de información importando los datos de más empresas.
#También emplearemos algoritmos de clasificación para ver si la distinción en si debemos comprar o vender da mejores resultados. También haremos Ensembling.













