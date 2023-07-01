import tkinter as tk
import numpy as np
import ast
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
from string import punctuation
import re

class MyApp(tk.Tk):
    def __init__(self):
        print("Inicializando...")
        super().__init__()
        self.title("Sistema de recomendación")
        self.create_widgets()
        # Configura la geometría de la ventana para centrarla en la pantalla
        self.geometry("640x800")
        self.dic_tratado = {}
        with open('dic_tratado.txt', 'r') as f:
            contenido = f.read()
        # Convierte el contenido en un diccionario utilizando ast.literal_eval()
        self.dic_tratado = ast.literal_eval(contenido)
        self.generos={}
        with open('generos.txt', 'r') as f:
            contenido = f.read()
        # Convierte el contenido en un diccionario utilizando ast.literal_eval()
        self.generos = ast.literal_eval(contenido)
        self.emb_dic={}
        with open('embenddings.txt', 'r') as archivo:
            for linea in archivo:
                # Eliminar los espacios en blanco al principio y al final de la línea
                linea = linea.strip()
                # Separar la línea en el ID y el vector
                id, vector = linea.split(':')
                # Eliminar los espacios en blanco al principio y al final del ID y convertirlo a entero
                id = str(id.strip())
                # Convertir la cadena que representa al vector a una lista de números
                vector = [float(num) for num in vector.strip()[1:-1].split(',')]
                # Guardar el ID y el vector en el diccionario
                self.emb_dic[id] = vector
        self.dic={}
        with open('diccionario.txt', 'r') as f:
            contenido = f.read()
        # Convierte el contenido en un diccionario utilizando ast.literal_eval()
        self.dic = ast.literal_eval(contenido)
        # Cargar modelo pre-entrenado y tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        # Enunciado a representar
    def recomendados(self,sim, k):
        recomendaciones = []
        i=0
        for key in sim:
            if i<k:
                recomendaciones.append(self.dic[key]['title'])
                recomendaciones.append(self.dic[key]['genres'])
                recomendaciones.append(self.dic[key]['argumento'])
            else:
                break
            i=i+1
        return recomendaciones
    def comparar_enunciados(self, enunciado):
        # Crear embedding del nuevo enunciado
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        inputs = tokenizer(enunciado, return_tensors='pt')
        tokens_tensor = inputs['input_ids']
        segments_tensor = inputs['token_type_ids']
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensor)
            nuevo_emb = outputs[0][:, 0, :][0].numpy()
        # Comparar con embeddings existentes
        similitudes = {}
        maximo=0
        for key in self.emb_dic:
            i=0
            for palabra in enunciado.split():
                if palabra in self.dic_tratado[key]:
                    i=i+1
            if i>maximo:
                maximo=i
        for key in self.emb_dic:
            similitud = 1 - cosine(nuevo_emb, self.emb_dic[key])
            i=0
            for palabra in enunciado.split():
                if palabra in self.dic_tratado[key]:
                    i=i+1
            if i>0:
                similitudes[key] = (similitud/2)+(i/(2*maximo))
            else:
                similitudes[key] = similitud/2
        sim_ordenado = dict(sorted(similitudes.items(), key=lambda x: x[1], reverse=True))
        return sim_ordenado
    def procesar_enunciado(self, enunciado):
        # Tokenización
        tokens = nltk.word_tokenize(enunciado)

        # Eliminación de stopwords y puntuación
        stopwords_english = stopwords.words('english')
        punctuations = list(punctuation)
        clean_tokens = [token.lower() for token in tokens if token.lower() not in stopwords_english and token not in punctuations]

        # Eliminación de números
        clean_tokens = [re.sub('\d', '', token) for token in clean_tokens]

        # Unión de tokens limpios en un solo string
        clean_enunciado = ' '.join(clean_tokens)
        return clean_enunciado
    def create_widgets(self):
        # Crea el objeto Entry para ingresar la cadena
        self.entry = tk.Entry(self,width=50)
        self.entry.pack(pady=10)

        # Crea el botón para enviar la cadena
        self.send_button = tk.Button(self, text="Enviar", command=self.show_list)
        self.send_button.pack(pady=5)

        # Carga la imagen y la reduce a la mitad
        self.image = tk.PhotoImage(file="imagen.gif")

        # Crea el objeto Label para la imagen
        self.image_label = tk.Label(self, image=self.image)
        self.image_label.pack(pady=5)

    def show_list(self):
        print("Cargando recomendador...")
        #id=self.calcular_similitud()
        enunciado= self.entry.get()
        clean_enun=self.procesar_enunciado(enunciado)
        print("Calculando...")
        similitudes = self.comparar_enunciados(clean_enun)
        # Borra el texto en el objeto Entry
        self.entry.delete(0, tk.END)

        # Oculta el objeto Entry
        self.entry.pack_forget()

        # Oculta el botón Enviar
        self.send_button.pack_forget()

        # Borra la imagen
        self.image_label.pack_forget()

        # Carga la imagen y la reduce a la mitad
        self.image = tk.PhotoImage(file="im2.gif").subsample(2)

        # Crea el objeto Label para la imagen
        self.image_label = tk.Label(self, image=self.image)
        self.image_label.pack(pady=5)

        # Crea y muestra la lista de películas
        self.movies = self.recomendados(similitudes,7)
        print('Recomendaciones:')
        self.movie_list = tk.Listbox(self)
        for movie in self.movies:
            self.movie_list.insert(tk.END, movie)
        self.movie_list.pack(pady=5,fill="both", expand=True)

        # Crea el botón para regresar a la pantalla inicial
        self.home_button = tk.Button(self, text="Inicio", command=self.go_home)
        self.home_button.pack(pady=5)

    def go_home(self):
        # Borra la lista de películas
        self.movie_list.pack_forget()

        # Borra la imagen
        self.image_label.pack_forget()

        # Muestra el objeto Entry
        self.entry.pack(pady=10)

        # Muestra el botón Enviar
        self.send_button.pack(pady=5)

        # Borra el botón Inicio
        self.home_button.pack_forget()

        # Carga la imagen y la reduce a la mitad
        self.image = tk.PhotoImage(file="imagen.gif")

        # Crea el objeto Label para la imagen
        self.image_label = tk.Label(self, image=self.image)
        self.image_label.pack(pady=5)

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()