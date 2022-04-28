# Load EDA Pkgs
import pandas as pd
import numpy as np
# Load Text Cleaning Pkgs
import neattext.functions as nfx
import sklearn
# Load ML Pkgs

# Estimators
from sklearn.linear_model import LogisticRegression

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report,confusion_matrix

# Load Dataset
df = pd.read_csv(r"C:\Users\S K Pachauri\AppData\Local\Programs\Python\Python38\Lib\emotion_dataset_raw-yes.csv")
# Value Counts
df['Emotion'].value_counts()



#sentiment analyis
from textblob import TextBlob
def get_sentiment(text):
    blob=TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
       result = 'Positive'
    elif sentiment < 0:
       result = 'neagative'
    else:
       result = 'Neutral'
    return result

df['Sentiment']=df['Text'].apply(get_sentiment)
df.head()
df.groupby(['Emotion','Sentiment'])
# Data Cleaning
dir(nfx)

# User handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
# Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
#remove special character
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_special_characters)

# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

cv = CountVectorizer()
cv.fit_transform(Xfeatures)
#  Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)
x_test.value_counts()
x_train.value_counts()
# Build Pipeline
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
# Train and Fit Data
pipe_lr.fit(x_train,y_train)
pipe_lr
#Checking accuracy of training data
pipe_lr.score(x_train,y_train)
# Check Accuracy of test data
pipe_lr.score(x_test,y_test)

y_pred_for_lr =pipe_lr.predict(x_test)
y_pred_for_lr
# Make A Prediction
ex1 = 'I am angry with you'
predicted_emotion=pipe_lr.predict([ex1])
print("Predicted emotion is :  ",predicted_emotion)
# Prediction Prob
pipe_lr.predict_proba([ex1])
# To Know the classes
pipe_lr.classes_
print ("prediction score of predicted emotion  ")
np.max(pipe_lr.predict_proba([ex1]))


def predict_emotion(sample_text, model):
    myvect = sample_text

    prediction = pipe_lr.predict(myvect)
    pred_proba = np.max(pipe_lr.predict_proba(myvect))
    pred_percentage_for_all = dict(zip(model.classes_, model.predict_proba(myvect)))
    print("Predicted Emotion: {}, \nPrediction Score: {}".format(prediction[0], np.max(pred_proba)))
    b = pipe_lr.predict_proba(myvect)

    return pred_percentage_for_all


import matplotlib.pyplot as plt


def emotion_bar_graph(a):
    y = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']
    # for changing dimendion of array
    x = a.ravel()
    # getting values against each value of y
    plt.barh(y, x)
    # setting label of y-axis
    plt.ylabel("Emotions ")
    # setting label of x-axis\
    plt.xlabel("strength of emotion depicted ")
    plt.title("Horizontal bar graph")
    plt.show()


def emotion_predictor():
   print("enter the input text\n\n ")
   str1= input()
   p= [str1]
   print("Predicted Emotion with their prediction score is :\n ")
   a = predict_emotion(p, pipe_lr)
   emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']


y = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']
# for changing dimension of array
# getting values against each value of y
# setting label of y-axis
plt.ylabel("Emotions ")
# setting label of x-axis\
plt.xlabel("strength of emotion depicted ")
plt.title("Horizontal bar graph")
plt.show()

emotion_predictor()
# classification

print(classification_report(y_test, y_pred_for_lr))
#confusion
confusion_matrix(y_test,y_pred_for_lr)
# plot confusion matrix
plot_confusion_matrix(pipe_lr,x_test,y_test)


