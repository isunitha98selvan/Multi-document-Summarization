from sentence_array import sentence_array_final
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import collections


def document_generate():
    doc_array = sentence_array_final
    documents = []
    for doc in range(len(doc_array)):
        sentences = doc_array[doc][0]
        documents.append(sentences)
    return documents

def k_means():

    documents = document_generate()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters, random_state = 42)
    km.fit(tfidf)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km.labels_):
        clustering[label].append(idx)

    summary = []
    for cl in range(len(clustering)):
        summary.append(documents[clustering[cl][0]])
    return summary

def main():
    Summary = k_means()
    print("The summary is:")
    print('.'.join(Summary))
    print("Word count:")
    print(len(''.join(Summary).split(' ')))

if __name__=='__main__':
    main()