import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from collections import defaultdict, Counter

# Loaing Dataset
df1 = pd.read_csv("Youtube01-Psy.csv")
df2 = pd.read_csv("Youtube02-KatyPerry.csv")
df3 = pd.read_csv("Youtube03-LMFAO.csv")
df4 = pd.read_csv("Youtube04-Eminem.csv")
df5 = pd.read_csv("Youtube05-Shakira.csv")
                
frames=[df1,df2,df3,df4,df5]
dataframe = pd.concat(frames)
                
dataframe.head()

print(dataframe.columns)

plt.figure(figsize = (16,8))

# Data Exploratoion

print("Count of Classes")
count =(dataframe.groupby("CLASS")["CLASS"].count())
print(count)
print()
print(dataframe.info())

plt.figure(figsize = (10,5))
sns.barplot(x = count.index, y = count)
plt.ylabel('Relative Frequency')
plt.xlabel('Classes')

#%%   
   
stop_words =set(stopwords.words('english'))

#this part gives us the topmost used stop words in the document
new= dataframe['CONTENT'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

dic=defaultdict(int)
for word in corpus:
  if word in stop_words:
        dic[word]+=1
        
#plotting the topmost used stop words in a graph            
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)
plt.bar(x,y)
#%%
# Most used words in the spam comments in a bar graph
spam_comment_words=dataframe[dataframe['CLASS']==1]
spam_comment_words['CONTENT']=spam_comment_words['CONTENT'].str.lower()

most=Counter(" ".join(spam_comment_words['CONTENT']).split()).most_common()
x, y= [], []
for word,count in most[:40]:
  if (word not in stop_words):
    x.append(word)
    y.append(count)
    
sns.barplot(x=y,y=x)

#%%
# Most used words in the ham comments in a bar graph
genuine_comment_words=dataframe[dataframe['CLASS']==0]
genuine_comment_words['CONTENT']=genuine_comment_words['CONTENT'].str.lower()

most=Counter(" ".join(genuine_comment_words['CONTENT']).split()).most_common()
x, y= [], []
for word,count in most[:40]:
  if (word not in stop_words):
    x.append(word)
    y.append(count)
    
sns.barplot(x=y,y=x)
   
#%%
# Data Pre-Processing

# converting the data set to lower case
dataframe['CONTENT'] =dataframe['CONTENT'].str.lower()

# removing stop words
dataframe['CONTENT'] = dataframe['CONTENT'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

dic={"/":" ",
     ".":" ",
     "-":" ",
     ",":" ",
     "=":" ",
     "  ":" ",
     ";":" ",
     "%":" ",
     "!":" ",
     ":" :" ",
     "?":" " }

dataframe['CONTENT'].apply(lambda x: replace_all(x, dic)) 


#shuffle the dataset
dataframe=dataframe.sample(frac=1, replace=True)

count_vect = CountVectorizer()
final_data = count_vect.fit_transform(dataframe['CONTENT'].values).toarray()
type(final_data)

tfidf = TfidfTransformer()
data_after_tfidf = tfidf.fit_transform(final_data).toarray()
print("Data Processed")

#%% 
# Model Training
#Set the split proportion

X =data_after_tfidf
Y =dataframe['CLASS']

split_index = int(0.75 * len(X))

#Splitting the dataframe into 75% train and 25% test
X_train = X[:split_index]
X_test = X[split_index:]

Y_train = Y[:split_index]
Y_test = Y[split_index:]

np.random.seed(6)

GB = GaussianNB()
GB.fit(X_train, Y_train)

MNB = MultinomialNB()
MNB.fit(X_train, Y_train)

#corss validation scores for training data

pred = GB.predict(X_train)
print("Cross-Validation scores: \n",cross_val_score(GB,X_train,Y_train,cv=5))
print("Mean is : ",cross_val_score(GB,X_train,Y_train).mean())
# print("Accuracy is:",accuracy_score(Y_train, pred))

pred_mnb = MNB.predict(X_train)
print("Cross-Validation scores: \n",cross_val_score(MNB,X_train,Y_train,cv=5))
print("Mean is : ",cross_val_score(MNB,X_train,Y_train).mean())
# print("Accuracy is:",accuracy_score(Y_train, pred))

#%%
# Testing the model on test data

pred_test = GB.predict(X_test)
print("\nModel Accuracy on Test Data set: ",round(accuracy_score(Y_test,pred_test)*100,2))
confusion_matrix_GB=confusion_matrix(Y_test, pred_test)
print(confusion_matrix_GB)

pred_test_mnb = MNB.predict(X_test)
print("\nModel Accuracy on Test Data set MNB: ",round(accuracy_score(Y_test,pred_test)*100,2))
confusion_matrix_MNB=confusion_matrix(Y_test, pred_test_mnb)
print(confusion_matrix_MNB)

#Plot the confusion Matrix
class_names=[0,1]
fig, ax = plt.subplots(figsize=(10,6))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(confusion_matrix_GB, annot=True,cmap="PuBu" ,fmt='g')
ax.set_ylim([0,2])
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix - Gaussian Naive Bayes',fontsize=20)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

fig, ax = plt.subplots(figsize=(10,6))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(confusion_matrix_MNB, annot=True,cmap="PuBu" ,fmt='g')
ax.set_ylim([0,2])
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix - Multinomial Naive Bayes',fontsize=20)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


test_data = ["Visit my channel for more videos&*^*(*",
           "Like,comment share and subscribe to my channel.",
           "http i love this ong",
           "I LOVE YOOU!!!!",
           "Katy is underrated",
           #"I am a big fan of you and your music!!"
           "Never heard a song as shitty as this one"]

final_vector = count_vect.transform(test_data).toarray()
input_data = tfidf.transform(final_vector).toarray()
result = GB.predict(input_data)
print(result)
    
