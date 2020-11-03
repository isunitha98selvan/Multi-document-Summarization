import sentence_imp as fv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_array import sentence_array_final
maxLen = 101
decoder_stacks = [[ ] for i in range(maxLen +1)]

def create_sentence_val():
    doc_array = fv.get_documents()
    doc_order = fv.get_doc_order()
    for doc in range(10):
        if(doc<len(doc_array) and len(doc_array[doc])>= 1):
            sentences = doc_array[doc][0].split(".")
            #doc_num = doc_order[doc]
            for group in sentences:
                print(group,doc)
                feature_vec = fv.tf_idf_sentence(group,doc)
                mapping = [group,sum(feature_vec)]
                sentences_array.append(mapping)
    print(sentences_array)

def importance(sentence_stack):
    summ =0
    for i in sentence_stack:
        summ+=i[1]
    return summ

def stack_decoder():
    threshold=0.5
    stack=[] #priority queue
    for i in range(maxLen):
        for j in decoder_stacks:
            for s in sentences_array:
                newLen=maxLen
                #decoder_stacks[num].append(0)
                sent_vec = []
                for val in j:
                    sent_vec.append(val[0])
                if i + len(s[0].split(" "))<maxLen:
                    newLen=i+len(s[0].split(" "))
                    if len(j)==0:
                        j.append(s)
                    else:
                        count = 0
                        for sentence in sent_vec:
                            if(cosine_similarity(s[0],sentence) < threshold):
                                count = count + 1

                        if(count == len(set_vec)):
                            j.append(s)
                    score=importance(j)
                    print(newLen)
                    decoder_stacks[newLen].append([j,score])
                    print(decoder_stacks)

create_sentence_val()
