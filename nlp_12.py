import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from nltk.stem.snowball import SnowballStemmer
import pickle
import joblib
from sklearn.preprocessing import normalize
from numpy import linalg as LA
def tfidf(i):
    Ngram_data=pd.DataFrame()
    feature_data={}
    vector_data={}
    #joblib.dump(i,'/test_tfidf.pkl',compress=True)
    for j in range(1,5):
        #print(j)
        temp= TfidfVectorizer(tokenizer=upper_tokenizer,lowercase=False,max_df=0.999,min_df=0.001,ngram_range=(j,j),max_features=int(3600/j),norm='l2').fit(i)
        tfidf_vectors_tf=temp.transform(i)
        count_temp= CountVectorizer(tokenizer=upper_tokenizer,lowercase=False,max_df=0.999,min_df=0.001,ngram_range=(j,j),max_features=int(3600/j)).fit(i)
        count_vectors=count_temp.transform(i)
        tfidf_vectors=np.array([])
        for m,v in enumerate(tfidf_vectors_tf):
            matrix=tfidf_vectors_tf[m].T*count_vectors[m]
            #matrix=np.array(matrix)
            matrix=np.array(matrix.toarray())
            rake=np.sum(matrix,axis=1)
            rake=rake/LA.norm(rake)
            #print(matrix.shape)
            #print(rake.shape)
            if m==0:
                tfidf_vectors=rake
            else:
                tfidf_vectors=np.vstack([tfidf_vectors,rake])
    
        tfidf_features=temp.get_feature_names()
        sorted_topic_modelling_array=np.array([])
        B_o=tfidf_vectors
        sorted_tfidf_fa=np.array([])
        vector_tfidf_fa=np.array([])
        #svd = TruncatedSVD(n_components=600, n_iter=50, random_state=42).fit(i).transform(i)
        for k,v in enumerate(B_o):
            B=B_o[k]
            sorted_tfidf=np.array([])
            sorted_tfidf_f=np.array([])
            sorted_topic_modelling=np.array([])
            B_sorted=np.argsort(B)
            B_sorted=np.flip(B_sorted)
            for l in B_sorted[:int(12/(j))]:
                sorted_tfidf=np.append(sorted_tfidf,B[l])
                sorted_tfidf_f=np.append(sorted_tfidf_f,tfidf_features[l])
                sorted_topic_modelling=np.append(sorted_topic_modelling,'{:.4f}*{}'.format(B[l],tfidf_features[l]))
            sorted_topic_modelling_a=' + '.join(sorted_topic_modelling)
            sorted_topic_modelling_array=np.append(sorted_topic_modelling_array,sorted_topic_modelling_a)
            if k==0:
                sorted_tfidf_fa=sorted_tfidf_f
                vector_tfidf_fa=sorted_tfidf
                #print(sorted_tfidf_fa.shape)
                #print(sorted_tfidf_f.shape)
            else:
                sorted_tfidf_fa=np.vstack([sorted_tfidf_fa,sorted_tfidf_f])
                vector_tfidf_fa=np.vstack([vector_tfidf_fa,sorted_tfidf])
        feature_data[j]=sorted_tfidf_fa
        Ngram_data[j]=sorted_topic_modelling_array
        #feature_data[j]=tfidf_features
        vector_data[j]=vector_tfidf_fa
        list_gram=[Ngram_data,feature_data,vector_data]
    return list_gram