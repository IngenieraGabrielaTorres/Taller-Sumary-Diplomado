#Taller 
#Gariela Torres
#Summary
#ID:1001970935
#ID:502193
#correo:gabriela.torresr@upb.edu.co
#Cel:3234708201
#Diplomado de PYTHON APLICADO A LA INGENIERIA 
#Docente:Roberto Paez Salgado
#Modulo 2



from email.policy import default
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score

scale = StandardScaler()

midataNetflix = pd.read_excel("Netflix_list.xlsx").dropna()


condiciones  = [
    (midataNetflix["type"] == "Movie"),
    (midataNetflix["type"] == "TV Show")
]

selecciones  = [1.0,2.0]



midataNetflix["type_normalized"] = np.select(condiciones , selecciones , default = "Not_specified")

separate_duration_movies = midataNetflix["duration"].str.split(expand=True)
midataNetflix.insert(4,"durationInt", separate_duration_movies[0].astype(int))
print(midataNetflix)

x,y = midataNetflix["durationInt"], midataNetflix["type_normalized"]
x,y = np.array(x).reshape(-1,1), np.array(y)
scaledX = scale.fit_transform(x)


trainX  = scaledX[:1600]
trainY = scaledX[:1600]


testX = scaledX[1600:]
testY = scaledX[1600:]

plt.scatter(scaledX[:100], y[:100])
plt.show()


model = linear_model.LinearRegression().fit(trainX, trainY)
print(model.score(trainX, trainY))
print(model.score(testX, testY))


pred_scaleX = model.predict([testX[0]])
print(pred_scaleX)


r2_train = r2_score(trainY, model.predict(trainX))
print(r2_train)

r2_test = r2_score(testY, model.predict(testX))
print(r2_test)



