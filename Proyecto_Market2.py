#En este script estudiaremos la implementación de algunos de algoritmos de clasificación y el estudio de sus resultados. Es por eso por lo que emplearemos labels1 como etiquetas.
#Comenzamos importando bibliotecas básicas y los datos de Google. Las funciones para modificar nuestros datos volverána  ser definidas por problemas a la hora de importarlas (resolver)

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

google = pd.read_csv("C:/Users/aleja/OneDrive/Escritorio/Market Data/Stocks/googl.us.txt")

google = google.drop(columns=["OpenInt"])
google["Diferencia"] = google["Close"] - google["Open"]
google["CV"] = np.where(google["Diferencia"] >= 0, "Compra", "Venta")

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

	google["CV"][i] = google["CV"][i+1]
	google["Diferencia"][i] = google["Diferencia"][i+1]

google = google.drop(range(0,50), axis=0)
google = google.drop([3332], axis=0)
google = google.reset_index() 

def split_train_test_ordered(data,test_ratio): 
	indices = range(len(google))
	train_set_size = int(len(data) * test_ratio)
	train_set_indices = indices[:train_set_size]
	test_set_indices = indices[train_set_size:]
	return data.iloc[train_set_indices], data.iloc[test_set_indices]

train_set, test_set = split_train_test_ordered(google, 0.8) 

train_set_prep = train_set.drop("CV",axis=1)
train_set_prep = train_set_prep.drop("Diferencia",axis=1)
train_set_prep = train_set_prep.drop("Date",axis=1)
train_set_labels1 = train_set["CV"].copy() 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_set_prep)
train_set_prep_scaled = scaler.transform(train_set_prep)

#A partir de este punto podemos aplicar algoritmos de clasificación
#Empezamos con StochasticGradientDescent (SGD), que es un clasificador binario (sólo distingue entre dos clases)

from sklearn.linear_model import SGDClassifier

sgdc = SGDClassifier(random_state = 42)
sgdc.fit(train_set_prep_scaled, train_set_labels1)

#Evaluar un clasificador normalmente es algo más complicado que una regresión. Podemos emplear también cross validation, aunque a veces necesitamos mayor control sobre la funcion
#(En Skicit Learn tenemos la función escrita)

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgdc, train_set_prep_scaled, train_set_labels1, cv=3, scoring="accuracy"))

#Vemos que aciertan en torno al 50% de las veces, pero esto no es ningún logro, puesto que solo tenemos dos opciones
#Es por esto que es más interesante que nos vayamos a mirar la confussion matrix:

from sklearn.model_selection import cross_val_predict #En lugar de devolver el score nos devuelve las predicciones de cada fold
sgdc_prediction = cross_val_predict(sgdc, train_set_prep_scaled, train_set_labels1, cv=3)

from sklearn.metrics import confusion_matrix
cmatrix = confusion_matrix(train_set_labels1, sgdc_prediction)
print("Confussion Matrix SGDC: \n", cmatrix) #Vemos que la matriz da unos resultados muy malos. 

#Veamos la precision (TP/(TP+ FP) y el recall (TP/(TP + FN)

from sklearn.metrics import precision_score, recall_score

print("Precision Score SGDC: ", precision_score(train_set_labels1, sgdc_prediction, pos_label = "Compra")) #Solo el 48% de las veces acierta cada vez que hay que comprar
print("Recall Score SGDC: ", recall_score(train_set_labels1, sgdc_prediction, pos_label = "Compra")) #Solo detecta el 29% de las compras

#Probamos ahora con otros algoritmos, como los árboles de decisión o SVM:

from sklearn.tree import DecisionTreeClassifier

TreeC = DecisionTreeClassifier(max_depth=3, random_state=42)
TreeC.fit(train_set_prep_scaled, train_set_labels1)
Tree_predictions = cross_val_predict(TreeC, train_set_prep_scaled, train_set_labels1, cv=3)
TreeMatrix = confusion_matrix(train_set_labels1,Tree_predictions)
print("Confussion Matrix Decision Tree Clasiffier: \n", TreeMatrix)
print("Precison Score Tree: ", precision_score(train_set_labels1, Tree_predictions, pos_label = "Compra"))
print("Recall Score Tree: ", recall_score(train_set_labels1, Tree_predictions, pos_label = "Compra"))  #Los resultados siguen siendo terribles.

#Support Vector Classifier:

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

SVC_l = LinearSVC(C=1, loss="hinge", max_iter =10000, random_state=42)
SVC_l.fit(train_set_prep_scaled, train_set_labels1)
SVC_l_predictions = cross_val_predict(SVC_l, train_set_prep_scaled, train_set_labels1, cv=3)
SVC_l_Matrix = confusion_matrix(train_set_labels1, SVC_l_predictions)
print("Confussion Matrix LinearSVM: \n", SVC_l_Matrix)
#print("Precision Score Linear SVM: ", precision_score(train_set_labels1, SVC_l_predictions, pos_label = "Compra"))
#print("Recall Score Linear SVM: ", recall_score(train_set_labels1, SVC_l_predictions, pos_label = "Compra"))  #Los resultados siguen siendo terribles.

#Podemos intentar modificar el límite de los algoritmos de clasificación y ver si conseguimos mejores resultados:
#Usamos decision_function que devuelve un score y después hacemos predicciones basadas en ese score usando el threshold deseado:
#Para saber que threshold queremos pintamos el recall y la precision para cada threshold.

labels_compra = (train_set_labels1 == "Compra") #Marca un uno las compras y un cero las ventas

scores = cross_val_predict(sgdc, train_set_prep_scaled, labels_compra, cv = 3, method="decision_function") #especificamos que son los scores lo que queremos

#ahora podemos dibujar la curva de precision y recall para todos los posibles threshold

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(labels_compra, scores)

#Función para graficar:

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-10, 5])
plt.show()

#Podemos hacer también el plot de precision vs recall para elegir un threshold:

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
#plt.show()

#Elegimos ahora un threshold de -2:

SGDC_prediction_m2 = (scores > -2)

#Estudiamos ahora la precisión y el recall y la confusion matrix:

print("-----------------------")

print(confusion_matrix(labels_compra, SGDC_prediction_m2))

print(precision_score(labels_compra, SGDC_prediction_m2))

print(recall_score(labels_compra, SGDC_prediction_m2))

#Vemos que han mejorado los resultados