import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

bk = pd.read_csv('Books.csv')

# Create a colume to combine important features
def comb_feature(data):
    features = []
    for i in range(0, data.shape[0]):
        features.append(str(data['BookTitle'][i])+' '+ str(data['BookAuthors'][i] +' '+str(data['description'][i])))
    return features

# Create a colume to store the combined features
bk['comb_feature'] = comb_feature(bk)

cm = CountVectorizer().fit_transform(bk['comb_feature'])

cosSim = cosine_similarity(cm)
inpTitle = input("Enter the book name: \t")

bookId = bk[bk.BookTitle == inpTitle]['BookId'].values[0]

# Create a list of (BookId, Similarity score)
score = list(enumerate(cosSim[bookId]))


#Short the list of Similar Books in decensing order 

sorted_score = sorted(score, key = lambda x:x[1] , reverse = True)
sorted_score = sorted_score[1:]
sorted_score


j=0
print("The 10 rocomended books are: \n")
for item in sorted_score:
    book_title = bk[bk.BookId == item[0]]['BookTitle'].values[0]
    print(j+1,  book_title)
    j = j+1
    if j>=10:
        break
