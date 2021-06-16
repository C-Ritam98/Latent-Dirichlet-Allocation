import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
import pickle
import os
from nltk.stem import WordNetLemmatizer



class Preprocessing:

    def __init__(self):
        pass

    def Reduce_inflection(self,Corpus):
        lemmatizer = WordNetLemmatizer()

        lemmatized_data = []

        for doc in Corpus:
            doc_list = []
            for word in doc.split(' '):
                if len(word) <= 3 or any(map(str.isdigit, word)) == True:
                    continue
                lemma = lemmatizer.lemmatize(word)
                doc_list.append(lemma)
            lemmatized_data.append(' '.join(doc_list))

        return np.array(lemmatized_data)

    def _preprocess_(self,_data):

        data = self.Reduce_inflection(_data)

        # remove words which occur in more than 90% of the documents, remove stopwords and keep the rare words.
        doc_vectorizer = CountVectorizer(max_df = 0.9,stop_words='english')  # max_df=0.90, min_df=2, max_features = 10000
        tf_data = doc_vectorizer.fit_transform(data).toarray()
        self.tf_data = tf_data
        

        print(f'Vocabulary size = {len(doc_vectorizer.vocabulary_.keys())}')
        
        tokenized_data = []  # tf_data is in sparse matrix form. tokenized_data[] will only store the indices of words present, with repetations, for each doc in corpus.

        for doc_idx, doc in enumerate(tf_data):
            embedded_doc = []
            non_zero_words = np.where(doc > 0)[0].tolist()

            for word_idx,word in enumerate(non_zero_words):
                embedded_doc = embedded_doc + [word for _ in range(doc[word])]

            tokenized_data.append(embedded_doc)

        self.inv_mapping = { v: k for k, v in  doc_vectorizer.vocabulary_.items()}
        return tokenized_data, len(doc_vectorizer.vocabulary_.keys())


class LDA:

    def __init__(self) -> None:
        pass

    def _fit_(self,tokenized_data, n_words,n_iterations = 10, n_topics = 10):
        self.data = tokenized_data  # data
        self.n_topics = n_topics

        n_docs = len(tokenized_data) # number of documents

        # n_words = Vocabulary size
        # dirichlet parameters initialisation
        alpha = 0.1
        beta = 0.1

        # other initialisations

        phi_t_w = np.zeros((n_topics,n_words))  # per-topic word distributions
        theta_d_t = np.zeros((n_docs,n_topics)) # per-document topic distributions

        n_w_in_d = np.zeros((n_docs))  # number of tokens in documents --> self.data[i]
        n_t_in_c = np.zeros((n_topics)) # number of times the topics are present in the corpus --> self.data

        topic_d_w = [[0 for _ in range(len(doc))] for doc in self.data] # t_d_w[i,j] = topic of self.data[i,j], numpy initialisation not done due to presence of docs of varied length.
        
        for doc_indx,doc in enumerate(self.data):
            for w_indx, word in enumerate(doc):
                # assign a random topic to each word in the beginning
                topic = random.randint(0,n_topics-1)

                topic_d_w[doc_indx][w_indx] = topic

                theta_d_t[doc_indx, topic] += 1
                phi_t_w[topic ,word] += 1
                
                n_w_in_d[doc_indx] += 1
                n_t_in_c[topic] += 1

        # training phase
        for _ in range(n_iterations):
            print(f"Iteration : {_+1}")
            for doc_indx,doc in enumerate(self.data):
                for w_indx,word in enumerate(doc):

                    topic = topic_d_w[doc_indx][w_indx]

                    # remove topic and word from the existing distribution
                    n_t_in_c[topic] -= 1
                    n_w_in_d[doc_indx] -= 1
                    phi_t_w[topic,word] -= 1
                    theta_d_t[doc_indx,topic] -= 1

                    # sampling new topic from multinomial distribution

                    prob_d_t = (theta_d_t[doc_indx] + alpha)/(n_w_in_d[doc_indx] + alpha*n_topics)
                    prob_t_w = (phi_t_w[:,word] + beta) / (beta*n_words + n_t_in_c)

                    prob_topic = prob_d_t * prob_t_w

                    prob_topic = prob_topic / np.sum(prob_topic)
                    

                    new_topic = np.random.multinomial(1, prob_topic).argmax() # sample from multinomial distribution

                    # updation after sampling
                    n_t_in_c[new_topic] += 1
                    n_w_in_d[doc_indx] += 1
                    phi_t_w[new_topic,word] += 1
                    theta_d_t[doc_indx,new_topic] += 1
                    topic_d_w[doc_indx][w_indx] = new_topic

        self.theta = theta_d_t
        self.phi = phi_t_w


    def plot_distribution(self,doc_indx):

        plt.plot(self.theta[doc_indx]/np.sum(self.theta[doc_indx]))
        plt.title(f"Distribution of topics in document {doc_indx}")
        plt.xlabel("Topics")
        plt.ylabel("Probability values")
        #plt.show()
        plt.savefig(f't_100_Document_{doc_indx}.png')
        plt.clf()
        #plt.imsave(f'Outputs/epoch{epoch}_step{count}.jpg', img, cmap='gray')

    def show_top_(self,n_topics,k,index_to_token):
        temp = 0
        for i in range(self.n_topics):
            temp += 1
            print(f"Topic number {i}:")
            topic = self.phi[i]
            sorted_indices = topic.argsort()[::-1][:k]
            print([index_to_token[x] for x in sorted_indices])
            if temp == n_topics:
                break

   



if __name__ == '__main__':

    Data = pd.read_csv('DataBase.csv')
    Abstracts = Data['Abstract'].values
    
    print(f'Number of documents = {Abstracts.shape[0]}')

    preprocess = Preprocessing()
    tokenized_data, n_tokens = preprocess._preprocess_(Abstracts)

    number_of_topics = 100

    lda = LDA()
    lda._fit_(tokenized_data,n_tokens,n_iterations = 100,n_topics = number_of_topics)

    for i in range(10):
        lda.plot_distribution(i)

    lda.show_top_(100,10,preprocess.inv_mapping)

    store_objects = {0:preprocess, 1:lda}

    path = './Store_1000.pkl'

    with open(path, 'wb') as f:
        pickle.dump(store_objects,f,pickle.HIGHEST_PROTOCOL)


