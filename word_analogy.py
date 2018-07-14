import os
import pickle
import numpy as np
import math
#from collections import defaultdict

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
def cos_sim(v1,v2):
    dot_prod = np.dot(v1,v2)
    v1_len = math.sqrt(sum([x*x for x in v1]))
    v2_len = math.sqrt(sum([x*x for x in v2]))
    return dot_prod / (v1_len*v2_len)
def simillar_words(words,embeddings, v2):
    list1={}
    for word in words:
        if word in dictionary:
            word_id=dictionary[word]
            vfind.append(embeddings[word_id])
    for i in range(len(vfind)):
        list1[i]=1-cos_sim(vfind[i],v2)
    value_list=list(list1.values())
    value_list.sort()
    if len(value_list)>21:
        value_list=value_list[:21]
    for key , value in list1.items():
        if value in value_list:
            if len(words)>key:
                print(words[key]),

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

temp=[]
example=[]
prediction=[]
example_temp=[]
prediction_temp=[]
read_data=[]

vfind=[]
vleft1=[]
vleft2=[]
vright2=[]
words=[]
left1=[]
left2=[]
right2=[]
similarity={}
with open('word_analogy_dev.txt') as f:
    read_data=list(map(lambda s: s.rstrip("\n"),f.readlines()))

for line in read_data:
    temp= line.split("||")

    example.append(temp[0])
    prediction.append(temp[1])

example_string = "\n".join(example)
prediction_string = "\n".join(prediction)

example_string=example_string.replace("\"","")
prediction_string=prediction_string.replace("\"","")

example_temp = example_string.split("\n")
prediction_temp = prediction_string.split("\n")


temp2 = []
temp3 = []
temp4 = []
temp5 = []
"""
##########################simillar_words#####################
word_list=['first','american','would']
for word in word_list:
    if word in dictionary:
        print("words similar to",word)
        word_id=dictionary[word]
        v2=embeddings[word_id]
        simillar_words(list(dictionary.keys()),embeddings,v2)
########################end###################################
"""
with open('word_analogy_dev_prediction.txt', 'a') as f:
    for line_no in range(len(example_temp)):
        #print(example_temp[line_no])
        #print(prediction_temp[line_no])
        vfind=[]
        vleft1=[]
        vleft2=[]
        vright2=[]
        words=[]
        left1=[]
        left2=[]
        right2=[]
        similarity={}
        temp2 = example_temp[line_no].split(",")
        temp4 = prediction_temp[line_no].split(",")
        for x in temp2:
            temp3=x.split(":")
            left1.append(temp3[0])
            left2.append(temp3[1])
        for y in temp4:
            temp5=y.split(":")
            words.append(temp5[0])
            right2.append(temp5[1])

        for word in words:
            if word in dictionary:
                word_id=dictionary[word]
                vfind.append(embeddings[word_id])
        for lword in left1:
            if lword in dictionary:
                word_id=dictionary[word]
                vleft1.append(embeddings[word_id])
        for lword in left2:
            if lword in dictionary:
                word_id=dictionary[lword]
                vleft2.append(embeddings[word_id])
        for rword in right2:
            if rword in dictionary:
                word_id=dictionary[lword]
                vright2.append(embeddings[word_id])
        i=0
        #print(vleft2)
        avg_difference_vectors=0
        while i<len(vleft1):
            avg_difference_vectors += np.subtract(vleft2[i],vleft1[i])
            i+=1
        #print(avg_difference_vectors)
        avg_difference_vectors = avg_difference_vectors/len(vleft1)
        #print(avg_difference_vectors)
        diff_vect_sum_analogy2=[]
        for i in range(len(right2)):
            diff_vect_sum_analogy2.append(np.subtract(vfind[i],vright2[i]))
        for i in range(len(diff_vect_sum_analogy2)):
            similarity[i]=1 - cos_sim(avg_difference_vectors,diff_vect_sum_analogy2[i])
        #print(similarity)
        maxsimilarity=0
        k1=tuple
        k2=tuple
        minsimilarity=float("inf")
        for key, value in similarity.items():
            if value>maxsimilarity:
                k1=(key,value)
                maxsimilarity=value
            if value<minsimilarity:
                k2=(key,value)
                minsimilarity=value



        for i in range(len(similarity)):
            f.write("\"")
            f.write(words[i])
            f.write(":")
            f.write(right2[i])
            f.write("\"")
            f.write(" ")

        f.write("\"")
        f.write(words[k1[0]])
        f.write(":")
        f.write(right2[k1[0]])
        f.write("\"")
        f.write(" ")
        f.write("\"")
        f.write(words[k2[0]])
        f.write(":")
        f.write(right2[k2[0]])
        f.write("\"")

        f.write("\n")
