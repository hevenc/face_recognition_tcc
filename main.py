from main_GUI import Ui_MainWindow
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import threading
import time

from PyQt5 import QtCore, QtGui, QtWidgets

import cv2
import numpy as np
import face_recognition
import os
import imutils
from PIL import Image
import pandas as pd
# Caminho para as imagens conhecidas

global encodeListKnown,images,classNames
images = []
classNames = []
def findEncodings(images):
    encodeList = []
    for img in images:
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    return encodeList
def update_cadastro():
    global encodeListKnown
    path = 'images/'

    # Carregando imagens conhecidas
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    encodeListKnown = findEncodings(images)

# Função para encontrar encodings das imagens conhecidas


# Encontrar encodings das imagens conhecidas
encodeListKnown = findEncodings(images)

print(len(encodeListKnown))


# Função para reconhecer um rosto
def reconhecer_rosto(image):
    
    # Tirar uma foto para reconhecimento
    
    #image = cv2.imread(image)
    reconhecido = False
    image = imutils.resize(image, width=400)

    # Executar reconhecimento facial
    imgS = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    imgs = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    name_save = ""
    if len(encodesCurFrame) ==0:
        return image, None
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.48)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if True in matches:
            first_match_index = matches.index(True)

            name = classNames[first_match_index].upper()

            # Desenhar um retângulo ao redor do rosto
            (top, right, bottom, left) = faceLoc
            cv2.rectangle(image, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)

            # Exibir o nome da pessoa na tela
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, name, (left * 4, top * 4 - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            reconhecido = True 
            name_save = classNames[first_match_index]
        else:
            print("Acesso negado.")
            # Se não houver correspondência, mostrar "Acesso Negado"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (top, right, bottom, left) = faceLoc
            cv2.rectangle(image, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
            cv2.putText(image, "Acesso Negado", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        

    return image, reconhecido, name_save

class Backend():

    def __init__(self, ui, main):
        self.__ui = ui
        self.__main = main
        self._cap = cv2.VideoCapture(0)
        self.__pause_image_path = "teste.png"
        self.__run_button = self.__ui.runButtonMainPage
        self.__stop_button = self.__ui.stopButton
        self.__image_display = self.__ui.ImageShow
        self.__image_cadastro = self.__ui.ImageCadastro
        self.__botao_cadastro_main = self.__ui.b_cadastrar_main
        self.__botao_cadastro_cadastro = self.__ui.b_cadastrar
        self.__botao_take_photo = self.__ui.take_photo
        self.__reconhecido = False
        self.__current_data =""
        self.__times_rec = 0
        self.__times_not_rec = 0
        self._save_image = None
        self.backend_setup()
       
        self._finish = False
    
    def backend_setup(self):
        self.update_image(cv2.imread(self.__pause_image_path))
        self.__run_button.clicked.connect(self.run_image)
        self.__stop_button.clicked.connect(self.stop_image)
        self.__botao_cadastro_cadastro.clicked.connect(self.cadastrar)
        self.__botao_cadastro_main.clicked.connect(self.change_page)
        self.__botao_take_photo.clicked.connect(self.take)
    
    def take(self):
        ret, frame = self._cap.read()
        self._save_image = frame
        self.update_image(frame,which="cadastro")
       

    def change_page(self):
        self.__ui.Nome.setText("")
        self.__ui.CPF.setText("")
        self.__ui.Email.setText('')
        self.__ui.Telefone.setText('')
        self.__ui.Endereco.setText('')
        self.update_image(cv2.imread(self.__pause_image_path), which="cadastro")
        self.__ui.MainTabWidget.setCurrentIndex(1)



    def cadastrar(self):
        nome = self.__ui.Nome.text()
        cpf = self.__ui.CPF.text()
        Email = self.__ui.Email.text()
        Telefone = self.__ui.Telefone.text()
        Endereco = self.__ui.Endereco.text()

        dict_to_save = {"nome":[nome],
        "cpf":[cpf],
        "Email":[Email],
        "Telefone":[Telefone],
        "Endereco":[Endereco]}
        df = pd.DataFrame.from_dict(dict_to_save)
        df.to_csv("dados/"+nome+".csv")
        cv2.imwrite("images/"+ nome+".jpg", self._save_image)
        time.sleep(2)
        update_cadastro()
        print(nome,cpf,Email,Telefone,Endereco)
        self.__ui.MainTabWidget.setCurrentIndex(0)



    def closeEvent(self):
        self._finish = True
        cv2.destroyAllWindows()
        self._cap.release()

   
    def change_signal(self,result,name):
        
        if result:
            self.update_image(cv2.resize(cv2.imread("acessoLiberado.png"), (400,400)))
            self.__ui.lblReponse.setText("Liberado")
            self.__ui.lblReponse.setStyleSheet("background-color:rgb(0,255,0)")
            self.__ui.info_label.setText(str(pd.read_csv("dados/"+name+".csv"))) #### MExer aqui
        else:
            self.update_image(cv2.resize(cv2.imread("acessoRecusado.jpg"), (400,400)))
            self.__ui.lblReponse.setText("Acesso negado")
            self.__ui.lblReponse.setStyleSheet("background-color:rgb(255,0,0)")
        time.sleep(2)
        self.__ui.lblReponse.setText("Processando...")
        self.__ui.lblReponse.setStyleSheet("background-color:rgb(255,255,255)")
        self.__ui.info_label.setText("  ")

    
    def im_read(self):
        update_cadastro()
        while True:
            ret, frame = self._cap.read()
            try:
                image, resposta, name  = reconhecer_rosto(frame)
            except:
                pass
            try:
                image, resposta  = reconhecer_rosto(frame)
            except:
                pass
            if resposta:
                self.__times_rec += 1
                self.__times_not_rec = 0
            else:
                if resposta == False:
                    self.__times_not_rec +=1
                self.__times_rec = 0

            if self.__times_rec > 10:
                self.change_signal(True,name)
                self.__times_rec = 0
            elif self.__times_not_rec > 10:
                self.change_signal(False,name)
                self.__times_not_rec = 0

            self.update_image(image)
            if self._finish:
                return
    
    def update_image(self, cv_img, which="display"):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        if which == 'display':
            self.__image_display.setPixmap(qt_img)
        else:
            self.__image_cadastro.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QPixmap.fromImage( QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888))

    def run_image(self):
        self._finish = False
        self.x = threading.Thread(target=self.im_read)
        self.x.start()
        
    
    def stop_image(self):
        self._finish = True
        time.sleep(0.5)
        self.update_image(cv2.imread(self.__pause_image_path))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    bck = Backend(ui, MainWindow)
    MainWindow.show()
    if not app.exec_():
        bck.closeEvent()
        sys.exit(0)