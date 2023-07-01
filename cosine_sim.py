import torch
import numpy as np
import time


class simi_cosine:
    def __init__(self,path="embenddings.txt"):
        self.emb_dic={}
        with open(path, 'r') as archivo:
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
    def similitud(self):
        n = len(self.emb_dic)
        matriz_similitud = np.zeros((n, n))
        i=0
        # Calcular la similitud del coseno entre cada par de vectores
        np.set_printoptions(threshold=np.inf)
        for  key1 in self.emb_dic:
            j=0
            for key2 in  self.emb_dic:
                start_time = time.time()
                similitud = np.dot(self.emb_dic[key1], self.emb_dic[key2]) / (np.linalg.norm(self.emb_dic[key1]) * np.linalg.norm(self.emb_dic[key2]))
                similitud_re= round(similitud.item(), 2)
                matriz_similitud[i][j] = similitud_re
                matriz_similitud[j][i] = similitud_re
                end_time = time.time()
                j=j+1
            #print("Tiempo de ejecución: ", end_time - start_time, "segundos")
            print(i, ': ', matriz_similitud[i])
            i=i+1

        return matriz_similitud
    def main(self):
        ln=len(self.emb_dic)
        sim=self.similitud()

run=simi_cosine()
run.main()