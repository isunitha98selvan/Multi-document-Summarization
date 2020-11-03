
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_arrays.sentence_array44 import sentence_array_final

def document_generate():
    doc_array = sentence_array_final
    documents = []
    for doc in range(len(doc_array)):
        sentences = doc_array[doc][0]
        documents.append(sentences)
    return documents


def extract_word_vec():
        # Extract word vectors from glove embedding
        word_embeddings = {}
        emb = open('glove.6B.100d.txt', encoding='utf-8')
        for line in emb:
                word_values = line.split()
                word = word_values[0]
                coefs = np.asarray(word_values[1:], dtype='float32')
                word_embeddings[word] = coefs
        emb.close()
        return word_embeddings

def page_rank():
        sentences = document_generate()
        word_embeddings = extract_word_vec()
        sentence_vectors = []
        for sentence in sentences:
                if (len(sentence) != 0):
                        vec = sum([word_embeddings.get(w, np.zeros((100,))) for w in sentence.split()])/(len(sentence.split())+0.001)
                else:
                        vec = np.zeros((100,))
                sentence_vectors.append(vec)
        
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
                for j in range(len(sentences)):
                        if (i != j):
                                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        sn = 3

        for i in range(sn):
                print(ranked_sentences[i][1])
                



page_rank()