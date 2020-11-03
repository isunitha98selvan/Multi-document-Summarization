from sentence_array import sentence_array_final
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import math

def sort_by_imp(sentences):
    sentences.sort(key = lambda x: x[1],reverse=True) 
    return sentences

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     WORD = re.compile(r'\w+')
     words = WORD.findall(text)
     return Counter(words)

def score_imp_summary(threshold):

    summary=[]
    maxWords=100
    sentence_array=sort_by_imp(sentence_array_final)
    #print(sentence_array)
    summary.append(sentence_array[0][0])
    curr_num_words = 0
    top = 0
    for i in range(1,len(sentence_array)):
        count = len(sentence_array[i][0].split(' '))
        if curr_num_words+count> maxWords and i==len(sentence_array)-1:
            break
        s1=text_to_vector(sentence_array[i][0])
        s2=text_to_vector(sentence_array[top][0])
        if get_cosine(s1,s2) < threshold:
           continue
        curr_num_words+=count                       
        summary.append(sentence_array[i][0])
        top+=1
    return summary

def main():
    Summary = score_imp_summary(0.5) 
    print("The summary is:")
    output = '.'.join(Summary)
    print(output)
    file=open('summary.txt',"w")
    file.write(output)
    print("Word count:")
    print(len(''.join(Summary).split(' ')))


if __name__=='__main__':
    main()   