import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
print("warming up")

def read_cell(csv_filename, row_index, column_index):
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == row_index:
                return row[column_index]

# Exemplu de utilizare:
csv_filename = 'points.csv'
row_index = 0  # Indicele rândului (poate fi modificat)
column_index = 0  # Indicele coloanei (0 pentru prima coloană, 1 pentru a doua, etc.)
#print(read_cell(csv_filename,1 , 1))

#Functia de calculare vector de coordonate ale unuei poze din csv
#Parametrii: nr= numarul liniei
def calculare_vector_fd_nr(nr):
    #nr este nr imaginii pe care o accesam
    #pentru nr=1, vom avea linia 2, sau row=1
    vec=([])
    for i in range (1,37,2):
        j=i+1
        vec.append([int(read_cell(csv_filename,nr,i )),int(read_cell(csv_filename,nr, j))])
        #salveaza coordonatele(x,y) ale  
    #vec.append([int(read_cell(csv_filename,nr,0 )),int(read_cell(csv_filename,nr, 0))])

    return vec



#print(calculare_vector_fd_nr(1))

#Functia de calculare a unghiului dintre 2 puncte si planul orizonntal
#Parametrii: p1(x1,y1),p2(x2,y2)
def calculare_unghi(p1, p2):
    #p1,p2 punctele pentru care 
    #trebuie calculat unghiul cu orizontala
    y=p2[1]-p1[1]
    x=p2[0]-p1[0]
    
    unghi_rad=math.atan2(x,y)
    
    #unghi_grade=math.degrees(unghi_rad)
    
    return unghi_rad


#Variabila in care se stocheaza vectorul cu unghiurile intre 2 puncte si orizontală
vec_antrenare=[]


#Functia de calcul al vectorului de antrenare/testare KNN 
# def vector_KNN():
#     #pentru primele n poze
#     for i in range (1,179,1):
#         #vectorul de caracteristici pentru poza i
#         #urmeaza prelucrarea lui prin aflarea unghiurilor
#         #1 cate unul
#         v=calculare_vector_fd_nr(i)
#        # print(len(v))
#         vec_aux=[]
#         for j in range (0,17,1):
#
#             p1=v[j]
#             p2=v[j+1]
#             unghi=calculare_unghi(p1, p2)
#             vec_aux.append(unghi)
#         vec_antrenare.append(vec_aux)
#
def vector_KNN():
    # pentru primele n poze
    for i in range(1, 179, 1):
        # vectorul de caracteristici pentru poza i
        # urmeaza prelucrarea lui prin aflarea unghiurilor
        # 1 cate unul
        v = calculare_vector_fd_nr(i)
        # print(len(v))
        vec_aux = []
        for j in range(0, 17, 1):
            p1 = v[j]
            p2 = v[j + 1]
            unghi = calculare_unghi(p1, p2)
            vec_aux.append(unghi)
        vec_antrenare.append(vec_aux)


#activarea functiei
vector_KNN()

vec_clase=[]
#vector_clase() - funcția ce retine clasa fiecarei poze din .csv
def vector_clase():
    for i in range (1,179):
        vec_clase.append(int(read_cell(csv_filename,i,0))-1)
    return 1

vector_clase()

#print(len(vec_antrenare))
#print(len(vec_clase))

#Functia de calculare a celui mai apropiat vecin al unei poze
#Impartim baza de date in antrenare 75 % si testare 25 %

vec_prob=[]

def KNN():
    
    X_train, X_test, y_train, y_test = train_test_split(vec_antrenare, vec_clase, test_size=0.25,random_state=42)
    
    medie=0
    minim=100
    maxim=0
    val_i=0
    nr_vecini=100
    # Crearea și antrenarea modelului KNN
    for i in range (1,nr_vecini,1):
        knn_model = KNeighborsClassifier(n_neighbors=i)  
        knn_model.fit(X_train, y_train)
        
        # Realizarea de predicții 
        predictions = knn_model.predict(X_test)
        #print(predictions)
        # Evaluarea acurateții
        accuracy = accuracy_score(y_test, predictions)
        vec_prob.append(accuracy)
        if maxim<=accuracy:
            maxim=accuracy
            print("Noul Maxim ",maxim," la i=",i)
            val_i=i
        if minim>accuracy:
            minim=accuracy
            
        medie= medie+accuracy
    
        #print(f"Acuratețe: {accuracy * 100:.2f}% ",i)
    knn_model = KNeighborsClassifier(n_neighbors=val_i)
    knn_model.fit(X_train, y_train)
    joblib.dump(knn_model, 'knn_model.pkl')

    medie=medie/nr_vecini
    print(f"Medie: {medie* 100:.2f}% ")
    print(f"Maxim: {maxim* 100:.2f}% ")    
    print(f"Minim: {minim* 100:.2f}% ")
    print("Acuratetea maxima la ",val_i," vecini")

#KNN_3 -acuratetea pentru primele 3 rezultate ca pondere
def KNN_3():
    
    # Separarea datelor în setul de antrenare și testare
    X_train, X_test, y_train, y_test = train_test_split(vec_antrenare, vec_clase, test_size=0.25,shuffle=True)
    
    # Crearea și antrenarea modelului KNN
    knn_model = KNeighborsClassifier(n_neighbors=13)
    knn_model.fit(X_train, y_train)
    
    # Realizarea de predicții pe setul de testare (cu probabilități)
    predictions_proba = knn_model.predict_proba(X_test)
    
    # Obținerea top 3 rezultate pentru fiecare predicție
    top3_indices = predictions_proba.argsort()
    #print(len(top3_indices))
    a=0
    for h in range (len(top3_indices)):
       #     print(top3_indices[h], y_test[h])
        a=a+1        

    # Calculul acurateții în funcție de condiția specificată
    numar_corecte = 0
    for i in range(len(y_test)):
        if y_test[i] in top3_indices[i][1:]:
            numar_corecte += 1
    
    acuratete = numar_corecte / len(y_test) * 100
    print(f"Acuratețe pe setul de testare: {acuratete:.2f}%")
    


print("Ce doriti sa vedeti?")
print("Press 0 - NN aplicat pe baza de date cu raport 75/25")   
print("Press 1 - primii 3 NN aplicat pe baza de date cu raport 75/25 ")   
pres=input()    


if pres == "1":
    for i in range(20):
        KNN_3()
elif pres == "0":
    KNN()
else:
    print("Tastare invalida")
x=[]
y=[]
for i in range (0,50,1):
    y.append(i+1)
    x.append(vec_prob[i]*100)

plt.plot(y, x)

# Add labels and title
plt.xlabel('K-vecini')
plt.ylabel('Acuratețea (%)')
plt.title('Evoluția vecinilor')

# Show the plot
plt.show()




#csv_filename = 'keypoints_all_images.csv'