import cv2 as cv
import os
import imutils as im

modelo="fotoselon"
ruta1='C:/Users/5470/Downloads/python/reconocimientof'
rutacomp=ruta1+'/'+modelo
if not os.path.exists(rutacomp):
    os.makedirs(rutacomp)

ruido=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print(ruido)
# camara=cv.VideoCapture(0)
camara=cv.VideoCapture('ElonMusk.mp4')
id=0
while camara.isOpened():
    _,frame=camara.read()
    frame=im.resize(frame,width=640)

    grises=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    idcaptura=frame.copy()

    cara=ruido.detectMultiScale(grises,1.3,5)
    for (x,y,e1,e2) in cara:
        cv.rectangle(frame,(x,y),(x+e1,y+e2),(255,0,0),2)
        #almacenafotos
        # caracap=idcaptura[y:y+e2,x:x+e1]
        # caracap=cv.resize(caracap,(160,160),interpolation=cv.INTER_CUBIC)
        # cv.imwrite(rutacomp+"/imagen_{}.jpg".format(id),caracap)
        # id+=1    

    cv.imshow("Resultado",frame)
    # if id==350:
    #     break
    if cv.waitKey(1)==ord("s"):
        break
camara.release()
cv.destroyAllWindows()

