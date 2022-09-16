import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
import random
import math
import matplotlib.pyplot as plt

# Função para criar uma imagem com determinada altura x largura e tonalidade, (cinza, BG, BGR ou BGRA)
def criarImg(altura, largura, canaisCores):
  cor = list()
  if canaisCores == 1: #Escala de cinza
    cor = (255)
  elif canaisCores == 2: #BG
    cor = (255, 0)
  elif canaisCores == 3: #BGR
    cor = (255,0,255)
  else: #BGRA
    cor = (255, 0, 255, 0)
  imagem = np.full((altura, largura, canaisCores), cor, dtype=np.uint8)
  return imagem

# Função para somar duas imagens, essa soma divide os tons de cinza das 2 imagens por 2. Em seguida soma-os
def soma_img_cinza(img_1, img_2):
  imResultado = criarImg(img_1.shape[0], img_1.shape[1], img_2.shape[2])
  for i in range(img_1.shape[0]):
    for j in range(img_1.shape[1]):
      imResultado[i, j] = int(img_1[i][j]/2) + int(img_2[i][j]/2)
  return imResultado

# Função para somar duas imagens, essa função apenas soma os valores das 2 imagens pixel a pixel e caso o valor exceda 255 ou seja menor que 0 ele se torna o próprio valor
def soma_img_cinza_2(img_1, img_2):
  imResultado = criarImg(img_1.shape[0], img_1.shape[1], 1)
  for i in range(img_1.shape[0]):
    for j in range(img_1.shape[1]):
      imResultado[i][j] = int(img_1[i][j] + img_2[i][j])
      if((img_1[i][j] + img_2[i][j]) > 255):
        imResultado[i][j] = 255
      if((img_1[i][j] + img_2[i][j]) < 0):
        imResultado[i][j] = 0
  return imResultado

# Calculo do MSE
def calculaMSE(img, img_ruido):
  somatorio = 0
  max = 0
  for i in range(img.shape[0]):
   for j in range(img.shape[1]):
     somatorio += (img[i][j] - img_ruido[i][j])**2
     if(max < img[i][j]):
       max = int(img[i][j])
  return float(somatorio/(img.shape[0]*img.shape[1])), max

# Calculo do PSNR
def PSNR(mse, max):
  PSNR = 10*math.log10((max**2)/mse)
  return PSNR

# Lendo a imagem
lena = cv.imread('/content/lena.png', 0)
jetplane = cv.imread('/content/jetplane.tif', 0)
gato = cv.imread('/content/gato.jpg', 0)

# Criando tres matrizes com o tamanho de cada imagem
lena_gauss = [[0 for x in range(lena.shape[0])] for y in range(lena.shape[1])]
gato_gauss = [[0 for x in range(gato.shape[1])] for y in range(gato.shape[0])]
jetplane_gauss = [[0 for x in range(jetplane.shape[0])] for y in range(jetplane.shape[1])]

# Preenchendo as imagem dos ruidos com valores de media 0 e desvio padrao
desvio_padrao = 100

for i in range(lena.shape[0]):
  for j in range(lena.shape[1]):
    lena_gauss[i][j] = int(random.gauss(0, desvio_padrao))
    
for i in range(jetplane.shape[0]):
  for j in range(jetplane.shape[1]):
    jetplane_gauss[i][j] = int(random.gauss(0, desvio_padrao))

for i in range(gato.shape[0]):
  for j in range(gato.shape[1]):
    gato_gauss[i][j] = int(random.gauss(0, desvio_padrao))

print("Lena com ruido: ")
lena_com_ruido = soma_img_cinza_2(lena, lena_gauss)
cv2_imshow(lena_com_ruido)
print("Histograma Lena: ")
histr_1 = cv.calcHist(lena_com_ruido, [0], None, [256], [0, 256])
plt.plot(histr_1) 
plt.show()
print("PSNR Lena: ", cv.PSNR(lena, lena_com_ruido))

print("Jetplane com ruido: ")
jetplane_com_ruido = soma_img_cinza_2(jetplane, jetplane_gauss)
cv2_imshow(jetplane_com_ruido)
print("Histograma Jetplane: ")
histr_2 = cv.calcHist(jetplane_com_ruido, [0], None, [256], [0, 256])
plt.plot(histr_2) 
plt.show()
print("PSNR Jetplane: ", cv.PSNR(jetplane, jetplane_com_ruido))


print("Gato com ruido: ")
gato_com_ruido = soma_img_cinza_2(gato, gato_gauss)
cv2_imshow(gato_com_ruido)
print("Histograma Gato: ")
histr_3 = cv.calcHist(gato_com_ruido, [0], None, [256], [0, 256])
plt.plot(histr_3) 
plt.show()
print("PSNR gato: ", cv.PSNR(gato, gato_com_ruido))



#operação de convolução:
mascara = np.array([[1/9, 1/9, 1/9],
	                    [1/9, 1/9, 1/9],
	                    [1/9, 1/9, 1/9]])

print("LENA COM MASCARA DE CONVOLUÇÃO APLICADA")
lena_convolucao = cv.filter2D(src=lena_com_ruido, ddepth=-1, kernel=mascara)
cv2_imshow(lena_convolucao)

print("JETPLANE COM MASCARA DE CONVOLUÇÃO APLICADA")
jetplane_convolucao = cv.filter2D(src=jetplane_com_ruido, ddepth=-1, kernel=mascara)
cv2_imshow(jetplane_convolucao)

print("GATO COM MASCARA DE CONVOLUÇÃO APLICADA")
gato_convolucao = cv.filter2D(src=gato_com_ruido, ddepth=-1, kernel=mascara)
cv2_imshow(gato_convolucao)

