
# coding: utf-8

# # Rapport python
# 
# Yueyao Cheng
# 
# Zhiqiang Zhong
# 
# David-Alexandre Eklo
# 

# <a id='table'></a>
# # Sommaire
# 
#  * [Description de l'étude](#description)
#    * But du TP
#    * Importation des données
#  * [Partie 1: Le prétraitement](#pretraitement)
#  * [Partie 2: TF-IDF](#tfidf)
#    * Un peu de théorie
#    * Les fonctions
#  * [Partie 3: Le système de recommandation](#recommand)    
#  * [Conclusion](#conclusion)

# <a id='description'></a>
# ## Description de l'étude
# 
# ### But du TP
# 
# Le but du TP est de fournir des recommandations d'articles à l'utilisateur en fonction des mots clés ou centres d'intérets de ce dernier.
# Ils peuvent être entrés directement dans la console ou être fournis sous la forme d'un path d'un document.

# ### Importation des données
# 
# Pour simplifier l'étude nous allons importer nous-mêmes un simple corpus de textes de 4 documents en anglais mais le système fonctions avec des documents plus longs.
# On commence par importer les modules nécessaires et le texte

# In[1]:

import nltk
import math
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

text1 = "Python is a 2000 made-for-TV horror movie directed by Richard Clabaugh. The film features several cult favorite actors, including William Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy, Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean Whalen. The film concerns a genetically engineered snake, a python, that escapes and unleashes itself on a small town. It includes the classic finalgirl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,  California and Malibu, California. Python was followed by two sequels: Python  II (2002) and Boa vs. Python (2004), both also made-for-TV films."

text2 = "Python, from the Greek word, is a genus of nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are recognised.[2] A member of this genus, P. reticulatus, is among the longest snakes known."

text3 = "The Colt Python is a .357 Magnum caliber revolver formerly manufactured by Colt's Manufacturing Company of Hartford, Connecticut. It is sometimes referred to as a \"Combat Magnum\".[1] It was first introduced in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued Colt Python targeted the premium revolver market segment. Some firearm collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy Thompson, Renee Smeets and Martin Dougherty have described the Python as the finest production revolver ever made."

text4 = "Python is a widely used high-level, general-purpose, interpreted, dynamic programming language.[24][25] Its design philosophy emphasizes code readability,and its syntax allows programmers to express concepts in fewer lines of code thanpossible in languages such as C++ or Java.[26][27] The language provides constructsintended to enable writing clear programs on both a small and large scale.[28]"


# <a id='pretraitement'></a>
# ## Partie 1: Le prétraitement
# 
# Comme on peut le remarquer, ces 4 documents présentent des textes non formatés. On veut dire par là qu'il a encore la ponctuation, des mots non peu intéressants ne signifiant rien quand à la nature du texte (stopword), au pluriels donc pas sous forme de token, etc...
# 
# On présente les fonctions principales de prétraitement:
# 
# 1. **get_tokens** pour normaliser le texte
# 2. **stem_tokens(tokens, stemmer)**:
# 2. **stemmed_info(text)**: Ne retiens que les tokens qui ne sont pas des stopwords (mots fréquents qui ne signifient pas grand chose comme 'le' 'la' en anglais

# In[2]:

##############################
##############################
#Pretreatment
def get_tokens(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(None, string.punctuation)    
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def stemmed_info(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]  
    
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    
    return stemmed


# <a id='tfidf'></a>
# ## Partie 2: TF-IDF
# 
# ### Un peu de théorie
# La **fréquence inverse de document** (inverse document frequency) est une mesure de l'importance du terme dans l'ensemble du corpus.
# Dans le schéma TF-IDF, elle vise à **donner un poids plus important aux termes les moins fréquents**, considérés comme plus discriminants
# Elle consiste à calculer le logarithme de l'inverse de la proportion de documents du corpus qui contiennent le terme   
# 
# C'est cette mesure que nous allons utiliser pour ensuite la calculer dans la similarité.
# Pour plus de détails voir la page [ici](https://fr.wikipedia.org/wiki/TF-IDF).
# 
# ### Les fonctions
# 
# 
# On présente les fonctions pour calculer le TF-IDF:
# 
# 1. **tf** renvoie la fréquence du mot à partir d'un counter
# 2. **n_containing(word, liste_document)**: Le nombre de fois que le mot donné est dans la count_list qui est la liste des documents
# 3. **idf(word, liste_document)**: Calcule l'*Inverse Document Frequency*.  
# 4. Retourne le simple produit de **tf** et **idf**

# In[3]:

#Realise les fonctions TF-IDF
def tf(word, document):   
    return document.count(word) / (len(document) * 1.)

def n_containing(word, liste_document):   
    return sum(1. for document in liste_document if word in document)

def idf(word, liste_document):   
    return math.log(len(liste_document) / (1. + n_containing(word, liste_document)))

def tfidf(word, document, liste_document):   
    return tf(word, document) * idf(word, liste_document)


# On traite les 4 textes et on obtient les N premiers mots clés de chaque document.
# 
# A noter qu'il faudra certaines données pour *nltk*. Le plus simple est de rentrer dans la console python le code suivant:
# 
# **import nltk**
# 
# **nltk.download()**
# 
# Une boite de dialogue s'ouvrira, nous vous invitons à cliquer sur le bouton *Download*
# 

# In[4]:

document1 = stemmed_info(text1)
document2 = stemmed_info(text2)
document3 = stemmed_info(text3)
document4 = stemmed_info(text4)

#define the quantity of the output key words 
N = 10
documentlist = [document1, document2, document3, document4]
#define a list to save the key words
documentkeywords = list()
for i, document in enumerate(documentlist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, document, documentlist) for word in document}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for word, score in sorted_words[:N]:
        
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
    words = list()
    for i in range(N):
        words.append(sorted_words[i])
    
    documentkeywords.append(words)


# <a id='recommand'></a>
# ## Partie 3: Le système de recommandation
# 
# C'est la partie finale. Il s'agit de fournir à l'utilisateur les documents les plus suceptibles de l'intéresser (sous forme d'un vecteur *[numéro_document : similarité]* en fonction des mots clés qu'il entre OU des mots qui se trouvent dans un document qu'il a déjà lu par exemple.

# In[5]:

#############################
############################
#Recommandation part
users_interests = documentkeywords
#calcule the similarity and sort 
def most_similar_file(array):
    similarity = list()
    sim = 0.0
    for i in range(len(documentlist)):
        for j in range(len(array)):
            for k in range(1,10):
                
                if(array[j] == users_interests[i][k][0]):                  
                    sim = sim + 1.0*users_interests[i][k][1]
                
        similarity.append((i,sim))
        sim = 0.0
                 
    sorted_sim = sorted(similarity,
                  key = lambda x:x[1],
                  reverse = True)
    return sorted_sim
#According to the type of the input element, use different steps  
type = raw_input("Enter the type of input element(document ou mots):")
if(type == "mots"):          
    num_array = list()
    doc=""
    num = raw_input("Enter how many elements you want:")
    print 'Enter mots in array: '
    for i in range(int(num)):
        n = raw_input("num :")
        num_array.append(str(n))
        doc= doc + str(n) + " "
    print 'ARRAY: ',num_array
    print('par ordre decroissant de similarite (numero_document, similarite)')
    print(most_similar_file(stemmed_info(doc))) 
else:
    path = raw_input("Enter the path of document:")
    file = open(path, 'r')
    text = file.read().encode('utf-8')
    
    document = stemmed_info(text)
    
    print(most_similar_file(document))
    


# <a id='conclusion'></a>
# ## Conclusion
# 
# Cette étude nous a permis de construire un système de recommandation d'articles/documents en fonctions des intérêts de l'utilisateur. Ici, le deuxième document est le plus similaire.
