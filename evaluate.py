import os
import pickle
from LDA import Preprocessing
from LDA import LDA
import numpy as np
import matplotlib.pyplot as plt

with open("Store.pkl",'rb') as f:
        stored_objs = pickle.load(f)

preprocess = stored_objs[0]
lda = stored_objs[1]

lda.show_top_(n_topics=30,k=20,index_to_token=preprocess.inv_mapping)
mapping = preprocess.inv_mapping

# rare words
data = np.array(preprocess.tf_data) > 0
indices = np.where(np.sum(1*data,axis=0) <= 25)[0].tolist()

print("Rare words:")
rare_map = dict()
for index in indices:
        s = mapping[index]
        topic = lda.phi[:,index].argmax()
        try:
                rare_map[topic].append(s)
        except:
                rare_map[topic] = []
                rare_map[topic].append(s)


print(f"Number of rare words in total: {len(indices)}")
# print rare words topicwise
for topic,s in rare_map.items():
        print(topic,end=':')
        print(s)

# topic distribution in corpus
n_topics = 100
topic_dist = [0]*n_topics



for i in range(4690):
        topic_dist[lda.phi[:,i].argmax()] += 1


plt.plot(topic_dist)
plt.title("Topic distribution in Corpus")
plt.xlabel("Topics")
plt.ylabel("Count")
plt.show()

