from pyeasyga import pyeasyga
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from timeitd import timeit
from timeit import default_timer as timer
import math
import random
im = Image.open('orzel.bmp') # import obrazu za pomocą biblioteki PIL
p = np.array(im) # konwertowanie obrazu na macierz z wartościami RGB

#d = np.ones((p.size[0], p.size[1]))
def bitconvertion(wej): # funkcja konwertowująca macierz z danymi RGB na macierz bitową
    wyj = np.zeros((len(wej), len(wej[0])))
    for i in range(len(wej[0])):
        for j in range(len(wej)):
            if np.mean(wej[j][i]) < 50: # jeśli średnia wartość piksela (j,i) edzie mniejsza niż 50 to wartość jest równa 1
                wyj[j][i] = 1
    return wyj
def tonono(wej): # funkcja konwertująca macierz bitów na format nonogramu
    wyj=[]
    for i in range(len(wej)):
        rzad = []
        licz = 0
        for j in range(len(wej[0])):
            if wej[i][j] == 1:
                licz = licz + 1
            else:
                if licz != 0:
                    rzad.append(licz)
                licz = 0
        if licz != 0:
            rzad.append(licz)
        wyj.append(rzad)
    return wyj # wynikiem jest tablica zawierająca dwie podtablice, gdzie w każdej zawarty jest wektor ciągów wartości "1" - w tablicy o indeksie 0 dla wierszy, w tablicy o indeksie 1 dla kolumn
def roznica(x,y): # funkcja obliczająca różnicę dwóch wektorów (niekoniecznie równych długości)
    wynik=[]
    if len(x)>len(y): # jeśli któryś z wektórów jest krótszy to dopisywana jest do niego odpowiednia liczba zer
       y=np.append(y,np.zeros(len(x)-len(y)))
    if len(x)<len(y):
        x=np.append(x, np.zeros(len(y) - len(x)))
    for i in range(len(x)):
        wynik.append(abs(x[i]-y[i]))
    return sum(wynik)
def dlugosc_euklidesowa(wektor): # funkcja zwracająca długość euklidesową wektora
    suma=0
    for n in wektor:
        suma+=n**2
    wynik = suma**0.5
    return wynik

# tworzenie danych wejściowych
obraz = bitconvertion(p)
data=[tonono(obraz),tonono(np.transpose(obraz))]
print(data)


def fitness1(chromosom, data):  # funkcjka fitness1 - zliczająca poprawne kolumny i wiersze
    mac_chr = np.array(chromosom).reshape(len(data[0]), len(data[1])) #konwersja chromosomu na macierz
    pion = tonono(mac_chr) # wyznaczanie odpowiednich wektorów (analogicznych do tych w nonogramie) dla wszystkich wierszy
    poziom = tonono(np.transpose(mac_chr)) # wyznaczanie odpowiednich wektorów (analogicznych do tych w nonogramie) dla wszystkich kolumn
    wynik = []
    for i in range(len(pion)):
        wynik.append(roznica(pion[i], data[0][i])==0)
    for i in range(len(poziom)):
        wynik.append(roznica(poziom[i], data[1][i])==0)
    return sum(wynik)

def fitness2 (chromosom,data): # funkcja fitness2 - sumującego różnice miedzy wektorami chormosomu a danymi wejściowymi
     mac_chr = np.array(chromosom).reshape(len(data[0]),len(data[1]))
     pion = tonono(mac_chr)
     poziom = tonono(np.transpose(mac_chr))
     wynik=[]
     for i in range(len(pion)):
         wynik.append(roznica(pion[i],data[0][i]))
     for i in range(len(poziom)):
         wynik.append(roznica(poziom[i], data[1][i]))
     return -dlugosc_euklidesowa(wynik)
ga = pyeasyga.GeneticAlgorithm(data, # ustalanie parametrów algorytmu genetrycznego ga
                               population_size=200,
                               generations=500,
                               crossover_probability=0.8,
                               mutation_probability=1,
                               elitism=False,
                               maximise_fitness=True)
def create_individual(data): #funkcja generująca pierwsze pokolenie
    chromosom = []
    for i in range(0, len(data[0]) * len(data[1])):
        chromosom.append(np.random.choice([0, 1]))
    return chromosom
print(create_individual(data))
ga.create_individual = create_individual
ga.fitness_function = fitness2
start = timer()
[fitnesy,best] = ga.run() #inicjacja algorytmu genetycznego
end = timer()
print(best)
plt.pcolor( np.flip(np.array(ga.best_individual()[1]).reshape(len(data[0]),len(data[1])),0) , cmap = 'Greys' ) #rysowanie najleszego rozwiązania
plt.axis('equal')
plt.show()
fig = plt.figure()


#tworzenie animacji z najlepszych rozwiązań - jedna klatka odpowiada jednemu pokoleniu
historia = np.flip(np.array(best[0]).reshape(len(data[0]), len(data[1])),0)
plt.axis('equal')
wykres = plt.imshow(historia, cmap='Greys')
k=True

def init():
     wykres.set_data(historia)

def animate(i):
     historia = np.array(best[i+1]).reshape(len(data[0]), len(data[1]))
     print(i)
     wykres.set_data(historia)

     return wykres
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(best)-1,
                               interval=33,repeat = False)
plt.show()
print(fitnesy)
plt.plot(fitnesy)
plt.ylabel('Wartość maksymalna funkcji fitness')
plt.xlabel('Pokolenie')
plt.grid()
plt.show()
print(end-start)