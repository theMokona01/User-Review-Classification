import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords') #irrelevant words (the, a, this, and, she etc)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #transform words to present tense (Ex. loved = love)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from colorama import Fore
from colorama import Style

class Review_Detection:
    
    def __init__(self):
        self.dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3) #ignore quotes on sentences (quoting=3)
   
    def cleanText(self, text):
        #cleaning the texts      
        review = re.sub('[^a-zA-z]', ' ', text) #remove all punctuations, anything that is not a letter, will be replaced by a space
        #^ = not, everything that is not a-z or A-Z
        review = review.lower() #make all words of review lowercased
        review = review.split() #split words
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not') #not include the not word in stopwords
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #function to remove all english stopwords, if the word is not a stopword, then include it in review list
        #apply steming for words that are not stopwords and add them to review()
        review = ' '.join(review) #join splited/edited words together with spaces to get string
        return review
    
    def predict(self, new_review):
        corpus = []
        for i in range(0,len(self.dataset.index.values)): #number of reviews in dataset, loop to clean each review
            corpus.append(self.cleanText(self.dataset['Review'][i]))
            
        #creating Bag of Words model
        cv = CountVectorizer(max_features = 1500) #get rid of remaining irrelevant words that stopwords function didnt remove
        X = cv.fit_transform(corpus).toarray()
        y = self.dataset.iloc[:, -1].values
        
        #len(x[0]) # number of words, try before cv
        
        #train,test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
        
        #XGBoost model
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_test)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        
        #confusion matrix
        #cm = confusion_matrix(y_test, y_pred)
        #print(cm)
        ac = float(accuracy_score(y_test, y_pred)) * 100
        
        #new prediction
        new_corpus = [self.cleanText(new_review)]
        new_X_test = cv.transform(new_corpus).toarray()
        new_y_pred = classifier.predict(new_X_test)
        
        if new_y_pred[0] == 0:
            review_pred = 'Negative'
            color = Fore.RED
        else:
            review_pred = 'Positive'
            color = Fore.GREEN
            
        print(f'\n\nUser left a {color}{review_pred}{Style.RESET_ALL} review!\nReview: "{new_review}"\n\nAccuracy Level={ac}%')
        
review_1 = 'We also ordered the spinach and avocado salad, the ingredients were sad and the dressing literally had zero taste.'
review_2 = 'Great food for the price, which is very high quality and house made.'

rd = Review_Detection()
rd.predict(review_1)
rd.predict(review_2)
