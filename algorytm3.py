from pyeasyga import pyeasyga
from pyeasyga2 import pyeasyga2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from matplotlib import animation
import math
import random
im = Image.open('orzel.bmp')
p = np.array(im)

#d = np.ones((p.size[0], p.size[1]))
def bitconvertion(wej):
    wyj = np.zeros((len(wej), len(wej[0])))
    for i in range(len(wej[0])):
        for j in range(len(wej)):
            if np.mean(wej[j][i]) < 50:
                wyj[j][i] = 1
    return wyj
def tonono(wej):
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
    return wyj
def roznica(x,y):
    ujemne=[]
    dodatnie =[]
    if len(x)>len(y):
       y=np.append(y,np.zeros(len(x)-len(y)))
    if len(x)<len(y):
        x=np.append(x, np.zeros(len(y) - len(x)))
    for i in range(len(x)):
        if x[i] ==0:
            dodatnie.append(y[i])
        else:
            if(x[i]-y[i]<0):
               ujemne.append(abs(x[i]-y[i]))
            else:
                dodatnie.append(abs(x[i]-y[i]))
    return [np.sum(ujemne),np.sum(dodatnie)]
def dlugosc_euklidesowa(wektor):
    suma=0
    for n in wektor:
        suma+=n**2
    wynik = suma**0.5
    return wynik
obraz = bitconvertion(p)
data=[tonono(obraz),tonono(np.transpose(obraz))]
def makefrom(k,l,dl): # funkcja tworząc wektor o długości dl składający się z zer i jedynek z podanymi ciągami k oraz odstępami l
    wyn=np.zeros(dl)
    m=0
    for i in range(len(k)):
        for j in range(l[i]):
            m=m+1
        for j in range(k[i]):
            if m < dl:
                wyn[m]=1
            m= m + 1
        m=m+1
    return wyn
def zamien(chromosom,data): # funkcja konwertująca wektor odstępów (chromosom na macierz zer i jedynek)
    dlugosc = len(data[1])
    mac_chr=[]
    od = 0
    do = len(data[0][0])
    for i in range(len(data[0])):
        mac_chr.append(makefrom(data[0][i], chromosom[od:do], dlugosc))
        od = od + len(data[0][i])
        if i < len(data[0]) - 1:
            do = do + len(data[0][i + 1])
    return mac_chr
def fitness1 (chromosom,data): # funkja fitness wykorzystana w algorytmie ga - jej chromosomami są wektory zawierające wartości -1, 0 i 1
    global test
    wyn=[]
    for i in range(len(test)):
        wyn.append(test[i]+ga.best_individual()[1][i])
    # print(wyn)
    for i in range(len(test)):
        wyn[i] =int( test[i] + chromosom[i])
        if wyn[i] < 0:
                wyn[i]=0
    mac_chr=zamien(wyn,data)
    poziom = tonono(np.transpose(mac_chr))
    dodatnie = []
    pion = tonono(mac_chr)
    for i in range(len(pion)):
        dodatnie.append(sum(roznica(pion[i], data[0][i])))
    for i in range(len(poziom)):
        dodatnie.append(sum(roznica(poziom[i], data[1][i])))
    return -dlugosc_euklidesowa(dodatnie)
def fitness2 (chromosom,data): # funkcja fitness głównego algorytmu ga2, której chromosomami są wartości odstępów pomiędzy ciągami "1" w wierszach
    print(chromosom)
    global test
    test = chromosom
    najlepszy=ga.run() # inicjacja algorytmu genetycznego ga
    for i in range(len(test)):
        test[i] = (test[i] + ga.best_individual()[1][i])
        if test[i] < 0:
            test[i] = 0
    return [ga.best_individual()[0],test] # funkcja zwraca również wartość chromosomu zsumowanego z najlepszym osobnikiem algorytmu genetycznego ga, wynik tego działania jest przypisywany temu chromosomowi

ga = pyeasyga.GeneticAlgorithm(data, #parametry algorytmu ga
                               population_size=50,
                               generations=20,
                               crossover_probability=0.8,
                               mutation_probability=1,
                               elitism=False,
                               maximise_fitness=True)
def create_individual(data): # funkcja generująca pierwsze pokolenie algorymu ga
    chromosom = []
    liczba=0
    for i in range(len(data[0])):
        liczba=liczba+len(data[0][i])
    for i in range(liczba):
        chromosom.append(np.random.choice([-1,0,1]))
    return chromosom
global test
print(data)
ga.create_individual = create_individual
ga.fitness_function = fitness1

ga2 = pyeasyga2.GeneticAlgorithm(data, #parametry algorytmu ga2
                               population_size=10,
                               generations=10,
                               crossover_probability=0.8,
                               mutation_probability=0.1,
                               elitism=True,
                               maximise_fitness=True)

def create_individual2(data): # funkcja generująca pierwsze pokolenie algorymu ga2
    chromosom = []
    liczba=0
    for i in range(len(data[0])):
        liczba=liczba+len(data[0][i])
    for i in range(liczba):
        chromosom.append(np.random.choice([1]))
    return chromosom
ga2.create_individual = create_individual2
ga2.fitness_function = fitness2
start = timer()
[fitnesy,best]=ga2.run()  #inicjacja algorytmu ga2
end = timer()

print("najlepszy:",ga2.best_individual())

print(data)
mac_chr=zamien(ga2.best_individual()[1],data)
plt.pcolor( np.flip(np.array(mac_chr).reshape(len(data[0]),len(data[1])),0) , cmap = 'Greys' )
plt.axis('equal')
plt.show()
fig = plt.figure()
print(fitnesy)
plt.plot(fitnesy)
plt.ylabel('Wartość maksymalna funkcji fitness')
plt.xlabel('Pokolenie')
plt.grid()
plt.show()
print(end-start)