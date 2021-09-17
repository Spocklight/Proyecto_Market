#Aplicamos  ahora redes neuronales:
#Empezamos importando librerías

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

#Tratamos los datos de la misma forma de la que lo habíamos tratado anteriormente

google = pd.read_csv("C:/Users/aleja/OneDrive/Escritorio/Market Data/Stocks/googl.us.txt")
google_orig = google.copy()
google = google.drop(columns=["OpenInt"]) 
google = google.drop("Date", axis = 1)
google["Diferencia"] = google["Close"] - google["Open"]
#google["CV"] = np.where(google["Diferencia"] >= 0, "Compra", "Venta")
google["CV_binary"] = np.where(google["Diferencia"] >= 0, 0, 1)          #Creamos una columna de compra venta numérica para poder pasársela al modelo


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

rsi = RSIcalcul(14)
google["RSI"] = rsi

for i in (range(len(google)-1)):

	google["CV_binary"][i] = google["CV_binary"][i+1]
	google["Diferencia"][i] = google["Diferencia"][i+1]

google = google.drop(range(0,50), axis=0)
google = google.drop([3332], axis=0)
google = google.reset_index(drop=True) #Drop = true hace que no se añada el antiguo indice como columna 

def split_train_test_ordered(data,test_ratio): 

	indices = range(len(data))
	train_set_size = int(len(data) * test_ratio)
	train_set_indices = indices[:train_set_size]
	test_set_indices = indices[train_set_size:]
	return data.iloc[train_set_indices], data.iloc[test_set_indices]

#Separamos entre el test y el train set

train_set, test_set = split_train_test_ordered(google, 0.8) 

#Dividimos la información en variable X y variable Y. En este caso nuestro target será la columna CV, que nos indica si ese día hay que comprar o vender.
#Por tanto empezaremos creando una red neuronal de clasificacion binaria. Además restablecemos los índices, puesto que deben ser independientes entre las diferentes muestras.

y_train = train_set["CV_binary"].copy() 
y_train = y_train.reset_index(drop=True)
train_set = train_set.drop("CV_binary", axis=1)
train_set = train_set.reset_index(drop=True)
y_test = test_set["CV_binary"].copy()
y_test = y_test.reset_index(drop=True)
test_set = test_set.drop("CV_binary", axis=1)
test_set = test_set.reset_index(drop=True)

#Creamos ahora a partir del train set, el data frame de validación.

train_set, valid_set = split_train_test_ordered(train_set, 0.75)
y_train, y_valid = split_train_test_ordered(y_train, 0.75)
y_valid = y_valid.reset_index(drop=True)
valid_set = valid_set.reset_index(drop=True)

#Podemos pasar ahora a crear nuestra primera red neuronal, que será un modelo secuencial

#El standardscaler hace que deje de ser un dataframe para pasar a ser un ndarray!

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)
valid_set = scaler.transform(valid_set)
test_set = scaler.transform(test_set)

model = keras.models.Sequential([
	keras.layers.InputLayer(input_shape=train_set.shape[1:]),
	keras.layers.Dense(300, activation="relu"),
	keras.layers.Dense(100, activation="relu"),
	keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())
print(model.layers)

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1e-3), metrics=["accuracy"])

#Entrenamos el modelo

history = model.fit(train_set, y_train, epochs=30,
                    validation_data=(valid_set, y_valid))

print(history)

print(history.params)
print(history.epoch)
print(history.history.keys())

#Podemos ver ahora las curvas de aprendizaje:

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

#Podemos ver como tanto el training como el validation loss decrecen, mientras que la accuracy aumenta a lo largo del entrenamiento. Vemos además que las curvas de validation
#y de training están cerca la una de la otra, lo que indica que no hay demasiado overfitting. El modelo trabaja mejor al principio en el set de validation que en el
#de training. Eso puede deberse al azar y al pequeño tamaño de los sets. Sin embargo, después el training set acaba teniendo un acierto mayor. (Todo esto ha cambiado ya)
#Vemos además como las curvas de validation loss y del training loss van separándose, lo que indica un poco de overfitting.

#Pasamos ahora a evaluar el modelo y a hacer algunas predicciones:

model.evaluate(test_set, y_test)

X_new = test_set[:10]
class_names = ["Comprar", "Vender"]
y_prob1 = model.predict(X_new)
print(y_prob1.round(2)) #Nos da la probabilidad de cada elemento y cada clase (Como solo son dos clases, solo aparece la probabilidad de que sea la primera)

#Podemos también predecir directamente:

y_prob2 = 1 - y_prob1
#y_prob = np.zeros(np.shape(y_prob1)[0], 2)
y_prob = np.append(y_prob1.round(2), y_prob2.round(2), axis=1) #Unimos las probabilidades para cada una de las clases
print(y_prob)


y_pred = np.argmax(y_prob, axis=-1)
clases_pred = np.array(class_names)[y_pred]
print(clases_pred)

#Lo comparamos con los valores reales:

clases_reales = np.array(class_names)[y_test[:10]] 
print(clases_reales)                                                      

#El modelo de momento no da nada. Posiblemente hubiese antes overfitting, porque teníamos mucha accuracy. Más adelante haremos tunning de parámetros a ver si mejora algo.