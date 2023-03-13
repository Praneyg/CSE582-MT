#!/usr/bin/env python
# coding: utf-8

# In[33]:


import nltk
import random
import numpy as np
import pandas as pd
import pprint, time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

'''
Reading the input file and storing the 
values of the 3 columns of each row in
a tuple: (<Word>, <POS_TAG>, <CHUNK_TAG>)
'''
f = open("train.txt", "r")
sentence_corpus = []
sentence = []

for line in f:
    line = line.strip()
    if line == "":
        sentence_corpus.append(sentence)
        sentence = []
    else:
        word, pos_tag, _ = line.split(" ")
        #ignoring the chunk tag for this task
        sentence.append((word, pos_tag))
f.close()

# Add the last sentence (if any)
if sentence:
    sentence_corpus.append(sentence)


# In[91]:


print(sentence_corpus[:2])


# In[ ]:


'''
First implementation: Vertebi Algorithm from scratch
Note: Time consuming: Test data running for more than
      3 Hours.
'''


# In[5]:


#Splitting the corpus data into train_data and test_data (validadtion) (80/20 split)
train_set,test_set =train_test_split(sentence_corpus,train_size=0.80,test_size=0.20,random_state = 101)

# List of all the tags in the train and the test set (it may not be unique)
train_tag_corpus = [ t for sentence in train_set for t in sentence ]
test_tag_corpus = [ t for sentence in test_set for t in sentence ]
print(len(train_tag_corpus))
print(len(test_tag_corpus))


# In[6]:


print(train_tag_corpus[:20])


# In[7]:


# Finding number of unique tags and words (Vocabulary)
train_tag_set = {tag for word, tag in train_tag_corpus}
vocab = {word for word, tag in train_tag_corpus}


# In[8]:


#Methods to compute transition and emission

'''
prev_tag -> current_tag 
Pr(current_tag | prev_tag) = (# of prev_tag -> current_tag)/(# of prev_tag)
'''
def computeTransition(prev_tag, current_tag):
    tags = [tag for _, tag in train_tag_corpus]
    
    #Count of prev_tag
    cnt_prev_tag = len([tag for tag in tags if tag == prev_tag])
    cnt_prev_curr_tag = 0
    
    for i in range(1, len(tags)):
        if tags[i-1] == prev_tag and tags[i] == current_tag:
            cnt_prev_curr_tag += 1
    
    return cnt_prev_curr_tag / cnt_prev_tag


# In[9]:


#The crux of HMM is the emission and transition probabilities

#Transition
transition = np.zeros((len(train_tag_set), len(train_tag_set)), dtype='float32')
train_tag_list = list(train_tag_set)
for i in range(len(train_tag_list)):
    for j in range(len(train_tag_list)):
        transition[i,j] = computeTransition(train_tag_list[i], train_tag_list[j])


# In[19]:


# compute Emission Probability
def word_given_tag(word, tag, train_bag = train_tag_corpus):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
#now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
 
     
    return (count_w_given_tag, count_tag)


# In[20]:


tags_df = pd.DataFrame(transition, columns = list(train_tag_list), index=list(train_tag_list))


# In[37]:


def viterbi_memoization(words):
    train_bag = train_tag_corpus
    tags = list(set([pair[1] for pair in train_bag]))
    
    # initialize memoization dictionary
    memo = {}
    
    # initialize probability matrix
    T = len(words)
    prob_matrix = np.zeros((T, len(tags)))
    
    # fill in first column of probability matrix
    for i, tag in enumerate(tags):
        if (words[0], tag) in memo:
            emission_p = memo[(words[0], tag)]
        else:
            emission_p = word_given_tag(words[0], tag)[0] / word_given_tag(words[0], tag)[1]
            memo[(words[0], tag)] = emission_p
        prob_matrix[0][i] = tags_df.loc['.', tag] * emission_p
        
    # fill in remaining columns of probability matrix
    for i in range(1, T):
        for j, tag in enumerate(tags):
            max_prob = 0
            for k, prev_tag in enumerate(tags):
                transition_p = tags_df.loc[prev_tag, tag]
                prob = prob_matrix[i-1][k] * transition_p
                if prob > max_prob:
                    max_prob = prob
                    if (words[i], tag) in memo:
                        emission_p = memo[(words[i], tag)]
                    else:
                        emission_p = word_given_tag(words[i], tag)[0] / word_given_tag(words[i], tag)[1]
                        memo[(words[i], tag)] = emission_p
                    prob_matrix[i][j] = max_prob * emission_p
                    
    # backtrack to find optimal sequence of tags
    state = []
    max_prob = max(prob_matrix[-1])
    prev_tag = None
    for i in range(T-1, -1, -1):
        for j, tag in enumerate(tags):
            if prob_matrix[i][j] == max_prob:
                if prev_tag:
                    state.insert(0, prev_tag)
                max_prob /= memo[(words[i], tag)]
                max_prob /= tags_df.loc[prev_tag, tag]
                prev_tag = tag
                break
    
    state.insert(0, prev_tag)
    return list(zip(words, state))


# In[39]:


rndom = [random.randint(1,len(test_set)) for x in range(10)]
test_run = [test_set[i] for i in rndom]
test_run_base = [tup for sent in test_run for tup in sent]
test_tagged_words = [tup[0] for sent in test_run for tup in sent]


# In[40]:


tagged_seq = Viterbi_memoization(test_tagged_words)
  
# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
 
accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)

#Accuracy of random 10 sentences on the split test data set is 94% using Viterbi.


# In[ ]:


'''
Second Implementation: Using NLTK's hmm
Takes less time and easier to impement
'''


# In[43]:


#Creating HMM object
HmmModel = nltk.HiddenMarkovModelTagger.train(train_set)

true_pos_tags = [tag for sentences in test_run for word, tag in sentences]

predicted_pos_tags=[]
for sentences in test_run:
    predicted_pos_tags += [tag for _, tag in HmmModel.tag([word for word, _ in sentences])]


# In[44]:


#Accuracy
print (classification_report(true_pos_tags, predicted_pos_tags))
#Accuracy of random 10 sentences on the split test data set is 95% using nltk's hmm


# In[45]:


true_pos_tags = [tag for sentences in test_set for word, tag in sentences]

predicted_pos_tags=[]
for sentences in test_set:
    predicted_pos_tags += [tag for _, tag in HmmModel.tag([word for word, _ in sentences])]


# In[46]:


#Accuracy
print (classification_report(true_pos_tags, predicted_pos_tags))


# In[ ]:


#Accuracy on the split test data set is 93%

