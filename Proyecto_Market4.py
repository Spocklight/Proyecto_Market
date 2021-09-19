#En este script vamos a emplear un modelo de red neuronal secuencial pero esta vez de regresión y no de clasificación. Para ello tendremos como output una sola neurona sin activation function.
#Empleamos tantas neuronas en el output como variables queremos obtener (En nuestro caso 1)
#Si queremos sesgar el valor del output, entonces sí que necesitamos activation function
#Empezamos cargando repositorios y cargando y tratando nuevamente los datos

import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import pandas as pd

np.random.seed(42)
tf.random.set_seed(42)

google = pd.read_csv("C:/Users/aleja/OneDrive/Escritorio/Market Data/Stocks/googl.us.txt")
google_orig = google.copy()
google = google.drop(columns=["OpenInt"]) 
google = google.drop("Date", axis = 1)
google["Diferencia"] = google["Close"] - google["Open"]  #En este caso la diferencia será el único output que necesitemos

def DiferenciaX(dias):

	dif = np.zeros(len(google))

	for i in range(len(google)):

		if i < dias:       

			dif[i] = 0

		else:

			dif[i] =  google.iloc[i,1] - google.iloc[i-dias,1]

	return dif

diff5 = DiferenciaX(5)
diff25 = DiferenciaX(25)
diff50 = DiferenciaX(50)

google["Diferencia5"] = diff5
google["Diferencia25"] = diff25
google["Diferencia50"] = diff50

def RSIcalcul(periodos):

	rsi = np.zeros(len(google))
	subidas=0
	bajadas=0
	variacion=0

	for i in range(len(google)):

		if i < periodos-1:  
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

rsi = RSIcalcul(14)
google["RSI"] = rsi

for i in (range(len(google)-1)):

	google["Diferencia"][i] = google["Diferencia"][i+1]

google = google.drop(range(0,50), axis=0)
google = google.drop([3332], axis=0)
google = google.reset_index(drop=True) 

def split_train_test_ordered(data,test_ratio): 

	indices = range(len(data))
	train_set_size = int(len(data) * test_ratio)
	train_set_indices = indices[:train_set_size]
	test_set_indices = indices[train_set_size:]
	return data.iloc[train_set_indices], data.iloc[test_set_indices]

train_set, test_set = split_train_test_ordered(google, 0.8) 

y_train = train_set["Diferencia"].copy() 
y_train = y_train.reset_index(drop=True)
train_set = train_set.drop("Diferencia", axis=1)
train_set = train_set.reset_index(drop=True)
y_test = test_set["Diferencia"].copy()
y_test = y_test.reset_index(drop=True)
test_set = test_set.drop("Diferencia", axis=1)
test_set = test_set.reset_index(drop=True)

train_set, valid_set = split_train_test_ordered(train_set, 0.75)
y_train, y_valid = split_train_test_ordered(y_train, 0.75)
y_valid = y_valid.reset_index(drop=True)
valid_set = valid_set.reset_index(drop=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)
valid_set = scaler.transform(valid_set)
test_set = scaler.transform(test_set)

#A partir de aquí empieza lo nuevo, que es la configuración de la red neuronal

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=train_set.shape[1:]),
    keras.layers.Dense(1)
])

#Como ahora el ruido es mayor (estamos haciendo una regresión), simplificamos la red, metiendo una única capa con 30 neuronas.

model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

history = model.fit(train_set, y_train, epochs=20, validation_data=(valid_set, y_valid))
print(history)


mse_test = model.evaluate(test_set, y_test)

#Vemos que el loss llega a bajar de 33 a 29 en el set de validacón, para después subir a 64 en el set de test. Un poco esperanzador.

y_test_positive = np.where(y_test<0, -y_test, y_test)
print(y_test[0:10])
print(y_test_positive[0:10])
print(np.mean(y_test_positive))

#Para una media de la diferencia entre el precio de apertura y de cierre en valor absoluto de 5.72, un loss de 30 es una barbaridad.
#Podríamos ya empezar a hacer predicciones 

X_new = test_set[:10]
y_pred = model.predict(X_new)
print(y_pred)

#Predice todos los resultados como negativos y no se alejan tanto del valor real como para que el loss (mse) sea 30, al menos con los 10 primeros valores.
#Volveremos más tarde a este tema.
