import csv
import numpy as np
import math
import cv2
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import ttkbootstrap as ttk
from tkinter import filedialog
from tkinter.messagebox import showerror, askyesno

from src import model
from src import util
from src.body import Body
from src.hand import Hand

import joblib

#pentru interfata

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')




body_parts = [
    "Nose", "Neck", "R-Sho", "R-Elb", "R-Wr", "L-Sho",
    "L-Elb", "L-Wr", "R-Hip", "R-Knee", "R-Ank", "L-Hip",
    "L-Knee", "L-Ank", "R-Eye", "L-Eye", "R-Ear", "L-Ear", "none", "none2","none3",
    "none", "none2","none3","none", "none2","none3","none", "none2","none3"
]




# Functia de calculare a unghiului dintre 2 puncte si planul orizonntal
# Parametrii: p1(x1,y1),p2(x2,y2)
def calculare_unghi(p1, p2):
    # p1,p2 punctele pentru care
    # trebuie calculat unghiul cu orizontala
    y = p2[1] - p1[1]
    x = p2[0] - p1[0]

    unghi_rad = math.atan2(x, y)

    # unghi_grade=math.degrees(unghi_rad)

    return unghi_rad

file_path = ""

vector_testare=[]

vec_testare=[]


nr_de_puncte=0

# Funcție pentru a aplica KNN pe o imagine
def apply_open_pose_on_image():
    # Incarcarea imaginii
    global file_path,vector_testare,vec_testare,nr_de_puncte
    image_path=file_path
    if file_path:

        oriImg = cv2.imread(image_path)

        # Redimensionarea imaginii inițiale
        target_size = (200, 300)
        # oriImg = cv2.resize(oriImg, target_size)

        # Preprocesare imagine
        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Redimensionarea canvas-ului
        canvas = cv2.resize(canvas, target_size)

        # Convertirea imaginii la formatul corect pentru afișare în Tkinter
        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        oriImg = Image.fromarray(oriImg)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(canvas)

        # Afișarea imaginii inițiale în prima eticheta
        oriImg_photo = ImageTk.PhotoImage(oriImg.resize(target_size))
        #oriImg_photo = oriImg_photo.resize((200, 300))
        label_oriImg.config(image=oriImg_photo)
        label_oriImg.image = oriImg_photo

        # Afișarea canvas-ului în a doua etichetă

        canvas_photo = ImageTk.PhotoImage(canvas)
        label_canvas.config(image=canvas_photo)
        label_canvas.image = canvas_photo

        #pregatirea vectorului de testare
        for i in range (18):
            if subset[0][i]>=0 or candidate[i][2]>=0.1:
                index=int(subset[0][i])
                print(body_parts[i], " Punctul pe desen:",index, " ", candidate[index][:2], "Punctul perfect ",i)
                vector_testare.append([float(candidate[i][0]), float(candidate[i][1])])
                nr_de_puncte+=1
            else:
                print("Nu s-a gasit elementul", body_parts[i])
                vector_testare.append([0,0])
        print(vector_testare," da", vector_testare[0][1],vector_testare[0][0])
        vec_aux = []
        for j in range(0, 17, 2):
            p1 = vector_testare[j]
            p2 = vector_testare[j + 1]

            unghi = calculare_unghi(p1, p2)
            vec_aux.append(unghi)
        vec_testare.append(vec_aux)


# Funcție pentru a deschide și afișa imaginea selectată
def open_image():
    clear_output_text()
    global file_path
    file_path = filedialog.askopenfilename(title="Open Image File",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 300))  # Redimensionare pentru afișare
        photo = ImageTk.PhotoImage(image)
        label_oriImg.config(image=photo)
        label_oriImg.image = photo

# Funcție pentru a adăuga text la caseta de text
def add_text_to_output (text) :
    output_text.insert(tk.END, text)

def clear_output_text():
    output_text.delete(1.0, tk.END)

def display_probabilities(probabilities):
    #probabil.delete("1.0", tk.END)  # Șterge textul existent
    for i, probs in enumerate(probabilities, start=1):
        #add_text_to_output(f"\nObservația {i}:\n")
        add_text_to_output("\n")
        for j, prob in enumerate(probs):
            if j == 2:
                add_text_to_output(f"Starea de Fericire: {prob:.2f}\n")
            elif j == 0:
                add_text_to_output(f"Starea de Furie: {prob:.2f}\n")
            elif j == 1:
                add_text_to_output(f"Starea de Tristete: {prob:.2f}\n")
            else:
                add_text_to_output(f"Starea de Frica: {prob:.2f}\n")
            #add_text_to_output( f"Starea {j}: {prob:.2f}\n")
        add_text_to_output("\n")  # Adaugă o linie goală între observații

def nearest_neighbours ():

    #X_train, X_test, y_train, y_test = train_test_split(vec_antrenare, vec_clase, test_size=0.25, random_state=42)
    global vec_testare,vector_testare,nr_de_puncte


    # Crearea și antrenarea modelului KNN

    i=13

    knn_model = joblib.load('knn_model.pkl')

    if nr_de_puncte == 18:
        # Realizarea de predicții
        predictions = knn_model.predict(vec_testare)
        probabilitati=knn_model.predict_proba(vec_testare)
        print(predictions)
        if predictions==2:
            add_text_to_output("Starea detectata: Fericire")
        elif predictions==0:
            add_text_to_output("Starea detectata: Furie")
        elif predictions==1:
            add_text_to_output("Starea detectata: Tristete")
        else:
            add_text_to_output("Starea detectata: Frica")
            #reset la variabile

        display_probabilities(probabilitati)
    else:
        add_text_to_output("Nr de puncte "+str(nr_de_puncte)+"/18")

    vector_testare = []
    vec_testare = []
    nr_de_puncte=0

# Crearea interfeței grafice
root = tk.Tk()
root.title("Clasificator de emotie")


# Setarea dimensiunilor ferestrei principale
root.geometry("800x400")


# Etichetă pentru afișarea imaginii inițiale
label_oriImg = tk.Label(root)
label_oriImg.grid(row=0, column=2, padx=10, pady=10)

# Etichetă pentru afișarea canvas-ului
label_canvas = tk.Label(root)
label_canvas.grid(row=0, column=1, padx=10, pady=10)

# Crează un frame separat pentru butoane
button_frame = tk.Frame(root)
button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")  # poziționare în partea stângă

# Crează un frame separat pentru butoane și output
button_output_frame = tk.Frame(root)
button_output_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ns")  # poziționare în partea stângă

# Buton pentru deschiderea imaginii
open_button = tk.Button(button_frame, text="Încarcă Imaginea", command=open_image)
open_button.pack(fill="x", padx=10, pady=10)

# Buton pentru proiectarea punctelor cheie
open_button = tk.Button(button_frame, text="Aplica OpenPose", command=apply_open_pose_on_image)
open_button.pack(fill="x", padx=10, pady=10)

# Buton pentru aplicarea algoritmului KNN
knn_button = tk.Button(button_frame, text="NN Algorithm (13)", command=nearest_neighbours)
knn_button.pack(fill="x", padx=10, pady=10)

# Text box pentru output
output_text = tk.Text(button_frame, height=10, width=30)
output_text.pack(fill="x", padx=10, pady=10)




root.mainloop()