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

 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score

scale = StandardScaler()


studentData = pd.read_csv("student_data.csv")
print(studentData.corr())

studentX = studentData[["G1","G2"]]
studentY = studentData[["G3"]]


scaledstudentX = scale.fit_transform(studentX)


studentXTrain = scaledstudentX[:400]
yTrain = studentY[:400]


studentXTest = scaledstudentX[400:]
yTest = studentY[400:]
 
model = linear_model.LinearRegression()
model.fit(studentXTrain,yTrain)


pred_scalestudentX = model.predict([studentXTest[0]])
print(pred_scalestudentX)

r2_train = r2_score(yTrain, model.predict(studentXTrain))
print(r2_train)

r2_test = r2_score(yTest, model.predict(studentXTest))
print(r2_test)

