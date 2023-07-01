import csv
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_es = stopwords.words('english')

class pretratamiento:
    def __init__(self):
        pass
    def crear_dicionario(self):
        dic={}
        with open('MovieLens.csv', newline='') as archivo_csv:
            lector_csv = csv.DictReader(archivo_csv, delimiter=',')
            i=0
            for fila in lector_csv:
                id = str(i)
                mid = fila['movieId']
                tit = fila['title']
                gen = fila['genres']
                arg= fila['argumento']
                dic[id] = {'movieId': mid, 'title': tit, 'genres': gen, 'argumento':arg}
                i=i+1
        return dic
    def delete_num(self, dic):
        for key in dic:
            s = dic[key].lower()
            s = re.sub(r"\d+", "", s)
            dic[key]=s
        return dic
    def create_vocab(self, dic):
        palabras = []
        for key in dic:
            gen = dic[key]
            palabras_gen = filter(lambda palabra: palabra not in stopwords_es, gen.split())
            palabras.extend(palabras_gen)
        vocabulario = set(palabras)
        return vocabulario
    def obtener_generos(self,dic):
        for key in dic:
            resultado = []
            palabras= dic[key]['genres'].split("|")
            for palabra in palabras:
                resultado.append(palabra)
            #for i in range(len(palabras)):
            #    for j in range(i+1, len(palabras)):
            #        combinacion = palabras[i] + '|' + palabras[j]
            #        resultado.append(combinacion)
            dic[key]['genres']=' '.join(resultado)
        return dic
    def dic_aumentado(self, dic):
        dic_tratado={}
        for key in dic:
            dic_tratado[key] =re.sub(r'[^\w\s]', '', dic[key]['title'])+' '+re.sub(r'[^\w\s]', '', dic[key]['genres'])+' '+re.sub(r'[^\w\s]', '', dic[key]['argumento'])
        return dic_tratado
    def main(self):
        dic=self.crear_dicionario()
        dic_gen=self.obtener_generos(dic)
        dic_au=self.dic_aumentado(dic_gen)
        dic_tratado=self.delete_num(dic_au)
        print(dic_tratado)
run=pretratamiento()
run.main()
