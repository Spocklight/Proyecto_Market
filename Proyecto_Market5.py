#Crearemos ahora modelos no secuenciales, con topologías algo más complejas. Además emplearemos tuning y callbacks.
#Empezamos cargando los datos y las librerías, como de costumbre.

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
google["Diferencia"] = google["Close"] - google["Open"]
google["CV_binary"] = np.where(google["Diferencia"] >= 0, 0, 1)

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

	google["CV_binary"][i] = google["CV_binary"][i+1]
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

#Separamos ahora los dos outputs, el del algoritmo clasificador y el de la regresión. Vamos a hacer las dos labores al mismo tiempo.
#y_train para la regresión e y_train2 para la clasificación

y_train1 = train_set["Diferencia"].copy() 
y_train1 = y_train1.reset_index(drop=True)
y_train2 = train_set["CV_binary"].copy()
y_train2 = y_train2.reset_index(drop=True)
train_set = train_set.drop(columns=(["Diferencia","CV_binary"]), axis=1)
train_set = train_set.reset_index(drop=True)

y_test1 = test_set["Diferencia"].copy()
y_test2 = test_set["CV_binary"].copy()
y_test1 = y_test1.reset_index(drop=True)
y_test2 = y_test2.reset_index(drop=True)
test_set = test_set.drop(columns=(["Diferencia","CV_binary"]), axis=1)
test_set = test_set.reset_index(drop=True)

train_set, valid_set = split_train_test_ordered(train_set, 0.75)
y_train1, y_valid1 = split_train_test_ordered(y_train1, 0.75)
y_train2, y_valid2 = split_train_test_ordered(y_train2, 0.75)
y_valid1 = y_valid1.reset_index(drop=True)
y_valid2 = y_valid2.reset_index(drop=True)
valid_set = valid_set.reset_index(drop=True)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_set = scaler.fit_transform(train_set)
valid_set = scaler.transform(valid_set)
test_set = scaler.transform(test_set)

#Vamos ahora a dividir los inputs en dos también. Uno lo enviaremos por una ruta para hacer la regresión, y otros por la otra para hacer la clasificación:
#Estos inputs podrían ser diferentes, escogiendo que variables queremos emplear para cada función, pero en nuestro caso emplearemos todas. Sólo cambiaremos la ruta que sigan

train_set1, train_set2 = train_set, train_set
test_set1, test_set2 = test_set, test_set
valid_set1, valid_set2 = valid_set, valid_set

#Creamos ahora el modelo:

input_A = keras.layers.Input(shape=train_set1.shape[1:], name="wide_input") #Entrada de la regresión. Al tener más ruido simplificaremos su camino
input_B = keras.layers.Input(shape=train_set2.shape[1:], name="deep_input") #Entrada de la clasificación. Entra en el camino más "deep" 
hidden1 = keras.layers.Dense(300, activation="relu")(input_B) #Camino que seguirá la clasificación
hidden2 = keras.layers.Dense(100, activation="relu")(hidden1)
concate = keras.layers.concatenate([input_A, hidden2]) #Este es el atajo que cogerán los datos de la regresión, saltándose el hidden1 y hidden2 y yendo directamente al 3
hidden3 = keras.layers.Dense(30, activation="relu")(concate)
output1 = keras.layers.Dense(1, name="output1")(hidden3)
output2 = keras.layers.Dense(1, name="output2", activation="sigmoid")(hidden2)

model = keras.models.Model(inputs=[input_A, input_B],
                           outputs=[output1, output2])

print("Resumen del modelo:")
print(model.summary())
print(model.layers)

#Ahora cada input necesitará su loss function

model.compile(loss=["mse", "binary_crossentropy"], optimizer=keras.optimizers.SGD(learning_rate=1e-3))

#Y al entrenarlo también necesitará dos outputs diferentes, evidentemente.

history = model.fit([train_set1, train_set2], [y_train1, y_train2], epochs=20,
                    validation_data=([valid_set1, valid_set2], [y_valid1, y_valid2]))
print("Print history:")
print(history)

total_loss, loss1, loss2 = model.evaluate(
    [test_set1, test_set2], [y_test1, y_test2])

print(total_loss, "Total_loss")
print(loss1, "loss1")
print(loss2, "loss2")

#Y podemos también hacer predicciones:


X_new_1 = test_set1[:5]
X_new_2 = test_set2[:5]
y_pred_1, y_pred_2 = model.predict([X_new_1, X_new_2])
print(y_pred_1, "Predicción de la regresión")
print(y_test1[:5], "Valores reales de la regresión")
print(y_pred_2, "Predicción de la clasificación")
print(y_test2[:5], "Valores reales de la clasificación")

#Empleamos ahora callbacks para, por ejemplo, quedarnos con los parámetros con los que el modelo obtenido un mejor rendimiento (checkpoint) o para detener el entrenamiento cuando el algoritmo deje de mejorar (earlystopping)
#Podemos implementar también los dos a la vez, para quedarnos con el mejor modelo y para interrumpir el entrenamiento cuando ya no haya progreso.
#Lo hacemos con la regresión:

keras.backend.clear_session()  #Resets all state generated by Keras.
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=train_set1.shape[1:]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True) #Guarda el checkpoint como "my_keras_kodel.h5"
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True) #Si no mejora en 5 epochs, el modelo para de entrenar

history = model.fit(train_set1, y_train1, epochs=10,
                    validation_data=(valid_set1, y_valid1),
                    callbacks=[checkpoint_cb, early_stopping_cb])

#De esta forma no tenemos que preocuparnos por entrenarlo demasiado tiempo y por el overfitting. El modelo se guardará únicamente si su desempeño es mejor que el mejor modelo anterior.

model = keras.models.load_model("my_keras_model.h5") # rollback to best model
mse_test = model.evaluate(test_set1, y_test1)
print(mse_test, "mse del mejor modelo tras haber hecho callback")

#Pasamos ahora a hacer tunning. Para eso tenemos que disfrazar el modelo como si fuese un regressor (una función)
#Creamos un modelo con una única neurona de salida, secuencial, al que le viene dado el input shape, el numero de hidden layers y las neuronas

keras.backend.clear_session() 
np.random.seed(42)
tf.random.set_seed(42)

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=train_set1.shape[1:]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


#Ahora creamos el keras regressor basado en este modelo, es decir el disfraz.

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

keras_reg.fit(train_set1, y_train1, epochs=100,
              validation_data=(valid_set1, y_valid1),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

mse_test = keras_reg.score(test_set1, y_test1)
y_pred = keras_reg.predict(X_new_1)

#Podríamos hacer también un kfold con crossvalidation_score y accuracy_score (ver link)
#https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor

#Ahora podemos utilizar el randomized search que ya conocemos para hacer el tuning:

from scipy.stats import reciprocal  #random continious function
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100)               .tolist(),
    "learning_rate": reciprocal(3e-4, 3e-2)      .rvs(1000).tolist(), 
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(train_set1, y_train1, epochs=100,
                  validation_data=(valid_set1, y_valid1),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

print(rnd_search_cv.best_params_, "Mejores parámetros del random search")
print(rnd_search_cv.best_score_, "Mejor score del random search")
print("A continuación el score en el test_set")
print(rnd_search_cv.score(test_set1, y_test1))
model = rnd_search_cv.best_estimator_.model #Nos quedamos con la mejor configuración como modelo
print("Evaluamos el mejor modelo en el test set")
model.evaluate(test_set1, y_test1)

#El score y el evaluate dan el mismo output, que es el loss en el test_set1