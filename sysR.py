import numpy as np
import ast
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
from string import punctuation
import re

class recomendacion:
    def __init__(self):
        print("Inicializando motor de busqueda")
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
    def calcular_similitud(self, enunciado):
        #Tokenizar enunciado y convertir a tensores
        inputs = self.tokenizer(enunciado, return_tensors='pt')
        tokens_tensor = inputs['input_ids']
        segments_tensor = inputs['token_type_ids']
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
            embeddings = outputs[0][:, 0, :]
            embeddings_list = [round(num.item(), 2) for num in embeddings[0]]
        # Asignar lista redondeada como valor del diccionario
        embending_enunciado = embeddings_list
        similitudes = {}
        #Calcular similitud
        #No borrar mas alla de este comentario!!!!!!!!!!1
        for key in self.emb_dic:
            similitud = np.dot(embending_enunciado, self.emb_dic[key]) / (np.linalg.norm(embending_enunciado) * np.linalg.norm(self.emb_dic[key]))
            similitud_re= round(similitud.item(), 2)
            similitudes[key]=similitud_re
        sim_ordenado = dict(sorted(similitudes.items(), key=lambda x: x[1], reverse=True))
        return  sim_ordenado
    def recomendados(self,sim):
        recomendaciones = []
        i=0
        for key in sim:
            if i<10:
                recomendaciones.append(key)
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
    def main(self):
        print("Calculando...")
        #id=self.calcular_similitud()
        enunciado="children movies like toy story"
        clean_enun=self.procesar_enunciado(enunciado)
        similitudes = self.comparar_enunciados(clean_enun)
        recomendaciones=self.recomendados(similitudes)
        #print(similitudes)

run=recomendacion()
run.main()
