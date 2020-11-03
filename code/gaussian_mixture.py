from sentence_array import sentence_array_final
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import collections
from sklearn.mixture import GaussianMixture

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
    gm = GaussianMixture(n_components=num_clusters, random_state = 42)
    labels = gm.fit(tfidf.toarray()).predict(tfidf.toarray())
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(labels):
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