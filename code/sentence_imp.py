import glob
import os 
import re
import string
import math
import nltk
import gensim.downloader as api


from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

k = 5
doc_array = []
doc_order = []
tf_of_words_in_all_docs = []


st = StanfordNERTagger('/home/sunitha/Desktop/6th sem/IR/Multi-document-extraction-based-Summarization/english.all.3class.distsim.crf.ser.gz',
					   '/home/sunitha/Desktop/6th sem/IR/Multi-document-extraction-based-Summarization/stanford-ner.jar',
					   encoding='utf-8')

root_dir = '../../Multi-document-extraction-based-Summarization/dataset/Cluster_of_Docs'

def clean_sentence(sentence):
    return ''.join(e for e in sentence if e.isalnum() or e in [' ','-','\'',',','\''])
    

doc_array = []
doc_order = []
#dir = '/home/sunitha/Desktop/6th sem/IR/Multi-document-extraction-based-Summarization/Cluster_of_Docs/d30001t'
#dir = '/home/rosa31/Desktop/6thSem/IR/project/Multi-document-extraction-based-Summarization/Cluster_of_Docs/d30001t'


def get_doc_order():
    return doc_order

def get_documents():
    return doc_array


#Returns sentence length
def length(text,document):
    sentences=tokenize.sent_tokenize(document)
    return len(text)/len(sentences)

#Counts number of verbs
def verbs(text):
    count=0
    text = nltk.word_tokenize(text)
    result = nltk.pos_tag(text)
    for i in result:
        if i[1]=='VB':
            count+=1
    return count

#Calculates sentence position in corpus
def sentencePos(document,sentence):
    sentences=tokenize.sent_tokenize(document)
    pos=0
    result = 0
    for i in sentences:
        if(i==sentence):
            result = pos
            break
        pos+=1
    return result/len(sentences)

#Counts the number of named entities
def count_named_entities(text):
    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    count=0
    
    for i in classified_text:
        if i[1]!='O':
            count+=1

    return count

#Counts the number of digits in a text
def count_digits(text):
    words = text.split(" ")
    count=0
    for word in words:
        if word.isdigit() == True:
            count+=1
    return count

#Calculate number of adjectives in a sentence
def adjectives_count(sentence):
    text = nltk.word_tokenize(sentence)
    result = nltk.pos_tag(text)
    count = 0
    for i in result:
        if(i[1] == 'JJ'):
            count = count + 1

    return count
    

#Calculate upper case words
def upper_case_words(sentence):
    words = sentence.split(" ")
    count = 0
    for word in words:
        if word.lower() == word:
            count = count
        else:
            count = count +1

    return count

# Cleaning words to remove unnecessary punctuations
def cleaned_words(sentence):
    words = re.split(r'\W+', sentence)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    final_words = []
    for word in stripped:
        if(word is not ''):
            final_words.append(word.lower())
    return final_words

# Creating frequency of words for a sentence and all sentences from a document are passed through this function
def create_frequency_dict(words,words_dict):	
    for word in words:
        word = word.lower()
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1
    
#Calculate tf of all words in a document
def get_tf_docs(document):
    words_dict = dict()
    total_words_in_doc = 0
    sentences = document[0].split(".")
    tf_of_words = dict()

    for sentence in sentences:
        words_in_sentence = cleaned_words(sentence)
        total_words_in_doc = total_words_in_doc + len(words_in_sentence)
        create_frequency_dict(words_in_sentence,words_dict)

    for key,value in words_dict.items():
        tf_of_words[key] = value/total_words_in_doc

    return tf_of_words

#Calculate tf of all words in all documents so we know which word exists in which doc and what its importance is
def calculate_tf_all_docs():
    for i in range(len(doc_array)):
        get_tf_of_words_in_doc = get_tf_docs(doc_array[i])
        tf_of_words_in_all_docs.append(get_tf_of_words_in_doc)
    return tf_of_words_in_all_docs

#Calculate number of top k words present in sentence
def top_k_tfidf_words(sentence,doc_no):
    tf_allwords = calculate_tf_all_docs()
    tokens = cleaned_words(sentence)
    #doc_no = doc_order.index(doc_no)
    sorted_k_tfidf = sorted(tf_allwords[doc_no].items(), key=lambda x: x[1],reverse = True)
    count = 0
    for i in range(k):
        for word in tokens:
            if(sorted_k_tfidf[i][0]==word):
                count = count + 1
                break
    
    return count


#Calculate tf-idf of words in a sentence and then sum them up 
def tf_idf_sentence(sentence,doc_no):
    tf_allwords = calculate_tf_all_docs()
    words_of_sentence = cleaned_words(sentence)
    tf_idf_sum = 0
    for word in words_of_sentence:
        word = word.lower()
        tf_word = tf_allwords[doc_no][word]
        doc_count = 0
        for doc in tf_allwords:
            if word in doc.keys():
                doc_count = doc_count + 1
        idf_word = math.log(len(tf_allwords)/doc_count)
        tf_idf_sum = tf_idf_sum + (tf_word*idf_word)

    #print(tf_idf_sum)
    top_k_words = top_k_tfidf_words(sentence,doc_no)
    #print(top_k_words)
    upper_case = upper_case_words(sentence)
    #print(upper_case)
    adjectives = adjectives_count(sentence)
    #print(adjectives)
    digit_count = count_digits(sentence)
    #print(digit_count)
    ner_count = count_named_entities(sentence)
    #print(ner_count)
    sentence_pos = sentencePos(doc_array[doc_no][0],sentence)
    #print(sentence_pos)
    verb_count = verbs(sentence)
    #print(verb_count)
    sentence_len = length(sentence,doc_array[doc_no][0])
    #print(sentence_len)
    feature_vector_for_one_sentence = [tf_idf_sum,top_k_words,upper_case,adjectives,digit_count,ner_count,sentence_pos,verb_count,sentence_len]
    return feature_vector_for_one_sentence

#cosine similarity- returns the similarity matrix
def cosine_similarity(sentences,sentence_vectors):
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    return sim_mat


file=open("sentence_array.txt","w")
for subdir, dirs, files in os.walk(root_dir):
    doc_array=[]
    doc_order=[]
    for file in files:
        doc_order.append(int(file[1:]))	
        with open(os.path.join(subdir,file)) as f:
            #print(os.path.join(subdir,file))
            para = f.readlines()
            #print(para)
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            lines = tokenizer.tokenize(para[0])
            sentences_dir = []

            for line in lines:
                sentences_dir.append(clean_sentence(line))
            doc_array.append(['.'.join(sentences_dir)])
    
    sentences_array=[]
    for doc in range(10):
        if(doc<len(doc_array) and len(doc_array[doc])>= 1):
            sentences = doc_array[doc][0].split(".")
            for group in sentences:
                feature_vec = tf_idf_sentence(group,doc)
                mapping = [group,sum(feature_vec)]

                sentences_array.append(mapping)
    