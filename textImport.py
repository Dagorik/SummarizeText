#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unicodedata
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem import SnowballStemmer
from gensim import corpora, models
import gensim
from random import shuffle

# delete utf-8 caracters
def elimina_tildes(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
def preproc(text):
    text=elimina_tildes(text.decode('utf-8'))
    text=text.lower()
    text=text.replace(",","")
    text=text.encode('ascii','ignore')
    text=text.split(".")
    return text
    
artificial = "La inteligencia artificial (IA), o mejor llamada inteligencia computacional, es la inteligencia exhibida por máquinas.\
 En ciencias de la computación, una máquina inteligente ideal es un agente racional\
 flexible que percibe su entorno y lleva a cabo acciones que maximicen sus posibilidades\
 de éxito en algún objetivo o tarea. Coloquialmente el término inteligencia artificial\
se aplica cuando una máquina imita las funciones cognitivas que los humanos asocian con otras\
 mentes humanas, como por ejemplo: aprender y resolver problemas.  A medida de que las máquinas\
se vuelven cada vez más capaces, tecnología que alguna vez se pensó que requería de inteligencia se\
elimina de la definición. Por ejemplo, el reconocimiento óptico de caracteres ya no se percibe como\
un ejemplo de la inteligencia artificial habiéndose convertido en una tecnología común.6 Avances\
tecnológicos todavía clasificados como inteligencia artificial son los sistemas capaces de jugar ajedrez, GO y manejar por si mismos."
artificial=preproc(artificial)
horror = "El cuento de terror (también conocido como cuento de horror o cuento de miedo, \
y en ciertos países de Sudamérica, cuento de suspenso), considerado en sentido estricto, es \
toda aquella composición literaria breve, generalmente de corte fantástico, cuyo principal objetivo \
parece ser provocar el escalofrío, la inquietud o el desasosiego en el lector, definición que no excluye \
en el autor otras pretensiones artísticas y literarias.\
El término horror suele referirse tanto a una emoción humana provocada por el miedo intenso, \
como a aquellos géneros de las artes narrativas —como la literatura, el cine, los videojuegos, \
la televisión y la historieta— que provocan dichas emociones. Ese género se divide en diferentes subgéneros."
horror=preproc(horror)
# compile sample documents into a list
text = artificial+horror
shuffle(text)

#tokenizer, stopwords, and stemmer
tokenizer = RegexpTokenizer(r'\w+')
sp_stop = get_stop_words('es')
sp_stemmer = SnowballStemmer("spanish")


# We'll save stemmed tokens in a list
texts = []
for line in text:
    # tokenizing
    tokens = tokenizer.tokenize(line)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in sp_stop]
    # stem tokens
    stemmed_tokens = [sp_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)

# tranform our tokenized set of documents into a id:word dictionary or hash table
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# Finally model find topics
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary)
print(ldamodel.print_topics(num_topics=2, num_words=4))