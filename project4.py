import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import PlaintextCorpusReader
import matplotlib.pyplot as plt
import copy
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D



# Get all text files and convert to dictionary with filename and respective tokens

files = '.*\.txt'
new_corpus = PlaintextCorpusReader("R:/ETSU/Artifical Intelligence/Rezwana_Project 4/speeches/speeches", files)

# corpus = nltk.Text(new_corpus.raw())

names = new_corpus.fileids()

def createDict(names):
        speech_dict = {}
        for counter, value in enumerate(names):
                speech_dict.update({value : new_corpus.raw(fileids = names[counter])})
        return speech_dict


# Remove Punctuation
def cleandata():
        # Create word tokens
        copy_of_speech_dict = copy.deepcopy(createDict(names))
        tokenizer = RegexpTokenizer(r'\w+')
        for key, value in copy_of_speech_dict.items():
                tokens = tokenizer.tokenize(value)
                tokens = [t.lower() for t in tokens]

        # Remove stop word tokens
                clean_tokens = tokens[:]
                for token in tokens:
                        if token in stopwords.words('english'):
                                clean_tokens.remove(token)
                copy_of_speech_dict.update({key : clean_tokens})     
        return copy_of_speech_dict


# # Lemmatization
def lemmatizeTokens():
        lemmatized_dict = {}
        lemmatized_dict = cleandata()
        for key, value in lemmatized_dict.items():
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(t) for t in value]
        lemmatized_dict.update({key : lemmatized_tokens})
        return lemmatized_dict

def createWordcloud():
        word_list = [value for value in lemmatizeTokens().values()]
        words = str(word_list)

        wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="white").generate(words)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


def createBigrams():
        current = {}
        current = lemmatizeTokens()
        for key, value in current.items():
                bigrams = ngrams(value, 2)
                bigram_list = []
                for gram in bigrams:
                        bigram_list.append(gram)
                if bigram_list not in current.values():
                        current.update({key : bigram_list})
        return  current


# # Method to plot frequency distribution
def plotFrequency():
        current = {}
        current = createBigrams()
        for value in current.values():
                freq = nltk.FreqDist(value)
                for key,val in freq.items():
                        print (str(key) + ':' + str(val))
                print("Length of Unique Items:", len(freq.items()))
        freq.plot(20, cumulative=False)

# Frequency of words used by single president
def plotFrequencyForEach():
        current = lemmatizeTokens().get('1965-Johnson.txt')
        freq = nltk.FreqDist(current)
        for key,val in freq.items():
                print (str(key) + ':' + str(val))
        print("Length of Unique Items:", len(freq.items()))
        freq.plot(30, cumulative=False)

# Document-Term Matrix
def createDTM():
        speeches = createDict(names)
        speech_list = []
        for key, value in speeches.items():
                speech_list.append(value)

        vec = CountVectorizer(stop_words='english', ngram_range=(0,2))
        X = vec.fit_transform(speech_list)
        df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        return df

# Generate elbow graph with k-means for optimal cluster number
def plotKmeansElbow():
        data = createDTM()
        num = range(2, 15)
        km = [KMeans(n_clusters=n, init='k-means++') for n in num]
        score = [km[n].fit(data).score(data) for n in range(len(km))]     
        plt.plot(num,score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        plt.show()

def createClusters():        
        data = createDTM()

        # calculate document similarity among each document in the DTM
        dist = 1 - cosine_similarity(data)

        # using cluster size of 4 according to elbow graph
        kmeans=KMeans(n_clusters=4)
        pos = kmeans.fit_transform(dist) 

        # plot points on the graph 
        xs, ys = pos[:,0], pos[:,0]
        clusters = kmeans.labels_.tolist()
        plt.figure('3 Cluster K-Means')
        plt.scatter(xs, ys, c=clusters, s=100)
        plt.title('3 Cluster K-Means')
        plt.show() 

plotKmeansElbow()









