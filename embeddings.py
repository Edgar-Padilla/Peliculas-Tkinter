import ast
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class embeddings:
    def __init__(self):
        pass
    def cargar_dic(self, path):
        # Abre el archivo en modo de lectura y lee su contenido
        with open(path, 'r') as f:
            contenido = f.read()
        # Convierte el contenido en un diccionario utilizando ast.literal_eval()
        diccionario = ast.literal_eval(contenido)
        return diccionario
    def cargar_voc(self, path):
        with open(path,'r') as f:
            contenido = f.read()
            vocabulario = set(contenido.split())
        return vocabulario
    def crear_embendding(self, dic):
        dic_em={}
        # Cargar modelo pre-entrenado y tokenizador
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        # Enunciado a representar
        for key in dic:
            enunciado = dic[key]
            # Tokenizar enunciado y convertir a tensores
            inputs = tokenizer(enunciado, return_tensors='pt')
            tokens_tensor = inputs['input_ids']
            segments_tensor = inputs['token_type_ids']
            # Obtener representación vectorial del enunciado
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensor)
                embeddings = outputs[0][:, 0, :]
                embeddings_list = [round(num.item(), 2) for num in embeddings[0]]
            # Asignar lista redondeada como valor del diccionario
            dic_em[key] = embeddings_list
        return dic_em
    def similitud_cosine(self,dic_emb):
        llaves = list(dic_emb.keys())
        # Crea una matriz vacía de similitud
        matriz_similitud = np.zeros((len(llaves), len(llaves)))
        # Calcula la similitud coseno para cada par de llaves
        for i in range(len(llaves)):
            for j in range(i, len(llaves)):
                similitud = torch.cosine_similarity(dic_emb[llaves[i]], dic_emb[llaves[j]], dim=0)
                # Redondea la similitud a dos dígitos
                similitud_redondeada = round(similitud.item(), 2)
                # Almacena el resultado en la matriz de similitud
                matriz_similitud[i][j] = similitud_redondeada
                matriz_similitud[j][i] = similitud_redondeada
        # Imprime la matriz de similitud
        return  matriz_similitud
    def main(self):
        dic=self.cargar_dic("dic_tratado.txt")
        dic_embeddings=self.crear_embendding(dic)
        #simi=self.similitud_cosine(dic_embeddings)
        for key in dic_embeddings:
            print(key,':',dic_embeddings[key])
run=embeddings()
run.main()
