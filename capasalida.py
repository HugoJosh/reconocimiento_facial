import cv2 as cv
import os

import imutils

rutadatos='C:/Users/5470/Downloads/python/reconocimientof/data'
listaData=os.listdir(rutadatos)
entrenamiento=cv.face.EigenFaceRecognizer_create()
entrenamiento.read('entrenamientoEigen.xml')
ruido=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(0)
while True:
    res,captura=camara.read()
    if res==False:
        break
    captura=imutils.resize(captura,width=640)
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara=ruido.detectMultiScale(grises,1.3,5)
    for (x,y,e1,e2) in cara:
        caracap=idcaptura[y:y+e2,x:x+e1]
        caracap=cv.resize(caracap,(160,160),interpolation=cv.INTER_CUBIC)
        resultado=entrenamiento.predict(caracap)
        if resultado[1]<8000:
            cv.putText(captura,'{}'.format(listaData[resultado[0]]),(x,y-20),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)
        else:
            cv.putText(captura,'no encontrado',(x,y-5),2,0.7,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)

    cv.imshow("Resultado",captura)
    if cv.waitKey(1)==ord('s'):
        break
camara.release()
cv.destroyAllWindows()