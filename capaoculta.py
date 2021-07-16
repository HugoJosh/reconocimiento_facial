import cv2 as cv
import os
import numpy as np
from time import time
rutadatos='C:/Users/5470/Downloads/python/reconocimientof/data'
listaData=os.listdir(rutadatos)
# print("data",listaData)
ids=[]
rostroData=[]
id=0
tiempoin=time()
for fila in listaData:
    rutacomp=rutadatos+'/'+fila
    for archivo in os.listdir(rutacomp):
        ids.append(id)
        print("Imagenes: "+fila+'/'+archivo)
        #transforma en automatico a escala de grises
        rostroData.append(cv.imread(rutacomp+'/'+archivo,0))
    id+=1
entrenamiento=cv.face.EigenFaceRecognizer_create()
print("iniciando entrenamiento")
entrenamiento.train(rostroData,np.array(ids))

entrenamiento.write("entrenamientoEigen.xml")
tiempofin=tiempoin-time()
print("Entrenamiento completado, tiempo transcurrido: "+str(tiempofin))