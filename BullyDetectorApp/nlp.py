from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import dill
import re
from nltk.tokenize import word_tokenize
import string
from difflib import SequenceMatcher
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tweepy
import string
from collections import Counter
import operator

RE_D = re.compile('\d')
clf = dill.load(open("assets/nlp/bully_linearsvm", 'rb'))
vocab = dill.load(open("assets/nlp/vocab_linearsvm", 'rb'))
fnoswearing = open("assets/nlp/noswearing.txt", "r")

def tfidf_fit(list_doc):
    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, 
                                    smooth_idf=True, sublinear_tf=False, 
                                    tokenizer=tokenize)
    model = sklearn_tfidf.fit(list_doc)
    return model

def tfidf_transform(list_doc, model):
    return model.transform(list_doc)

def remove_hyperlinks(list_doc):
    temp = []
    for text in list_doc:
        temp_text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        temp.append(temp_text)
    return temp

def remove_certain_char(list_doc):
    temp = []
    for text in list_doc:
        text = text.replace('?', ' ')
        text = text.replace('.', ' ')
        temp.append(text)
    return temp

def tokenize(list_doc):
    temp = []
    for text in list_doc:
        s = re.sub(r'[^\w\s]','',text.lower())
        list_word = word_tokenize(s)
        temp.append(list_word)
    return temp

def change_second_pronoun(list_doc):
    temp = []
    second_pronoun = ['you', 'your', 'youre', 'yours', 'u', 'ur', 'ure', 'urs']
    
    for text in list_doc:
        temp_2 = []
        for word in text:
            if word in second_pronoun:
                word = 'secondpronoun'
            temp_2.append(word)
        temp.append(temp_2)
    return temp

def change_third_pronoun(list_doc):
    temp = []
    third_pronoun = ['he', 'hes', 'his', 'she', 'shes', 'her', 'hers', 'they', 'theyre', 
                     'their', 'theirs', 'it', 'its']
    
    for text in list_doc:
        temp_2 = []
        for word in text:
            if word in third_pronoun:
                word = 'thirdpronoun'
            temp_2.append(word)
        temp.append(temp_2)
    return temp

def hasNumbers(string):
    return RE_D.search(string)

def remove_unused(list_doc):
    temp = []
    for text in list_doc:
        remove_temp = []
        for word in text:
            if (hasNumbers(word)) or (len(word)<3) or (len(word)>20):
                remove_temp.append(word)
        for word in remove_temp:
            text.remove(word)
        temp.append(text)
    return temp

def remove_repeat_occ(word):
    pre_char = ""
    temp = ""
    ctr = 0
    for char in word:
        if char == pre_char:
            if ctr < 1:
                ctr += 1
                temp += char
        else:
            pre_char = char
            ctr = 0
            temp += char
    return temp

from nltk.corpus import stopwords

def remove_stop_words(list_doc):
    stop = set(stopwords.words('english'))
    temp = []
    for text in list_doc:
        list_word = [i for i in text if i not in stop]
        temp.append(list_word)
    return temp
    
def normalize_bad_words(list_doc):
    noswearing = fnoswearing.read().split('\n')
    foul_words = []
    for word in noswearing:
        foul_words.append(word.split(' ', 1)[0])
    temp = []
    for text in list_doc:
        temp_2 = []
        for word in text:
            for word2 in foul_words:
                if word == word2:
                    word = 'foulword'
                    break
            temp_2.append(word)
        temp.append(temp_2)
    return temp

def lemmatize(list_doc):
    temp = []
    lemmatizer = WordNetLemmatizer()
    for text in list_doc:
        temp_2 = []
        for word in text:
            lemma = lemmatizer.lemmatize(word)
            temp_2.append(lemma)
        temp.append(temp_2)
    return temp

def conjugate(list_doc):
    temp = []
    for text in list_doc:
        s = ""
        for word in text:
            s += word + " "
        s = s.rstrip(' ')
        temp.append(s)
    return temp

def preprocess(list_doc):
    X = remove_hyperlinks(list_doc)
    X = remove_certain_char(X)
    X = tokenize(X)
    X = change_second_pronoun(X)
    X = change_third_pronoun(X)
    X = remove_unused(X)
    temp_1 = []
    for list_word in X:
        temp_2 = []
        for word in list_word:
            temp_2.append(remove_repeat_occ(word))
        temp_1.append(temp_2)
    X = temp_1
    X = remove_stop_words(X)
    #X = normalize_bad_words(X)
    X = lemmatize(X)
    X = conjugate(X)
    return X

def get_tweets(username):
    # Authenticate
    consumer_key= 'ftDI7PRSrrr4ARs043ZZmIRh3'
    consumer_secret= '1SRJkRNYmVacusTGgjVsTSrGEE56xIoYUT4cTnUqxyGFq3TR08'

    access_token='480263154-ILFmjSYffr5j06Sj6xBRzjrYetLFSHcge0k8ioRq'
    access_token_secret='ljcBtKGWl7U8c2JpoDxfFkoSex8oaaeaFcHLVSLycu5qa'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    public_tweets = api.user_timeline(screen_name = username, count = 100)
    
    instances = []
    for tweet in public_tweets:
        instances.append(tweet.text)

    return instances

def evaluate(instances):
    processed = preprocess(instances)
    model_tfidf = tfidf_fit(vocab)
    X_test = tfidf_transform(processed, model_tfidf)
    predicted = clf.predict(X_test)
    results = {
        "bully_tweets": [],
        "isbully": 0,
        "bully_words": [],
        "original_tweets": instances
    }
    coef = clf.coef_.ravel()
    feature_names = model_tfidf.get_feature_names()
    for i in range(0, len(predicted)):
        if (predicted[i] == 1):
            print instances[i]
            results['bully_tweets'].append(instances[i])
            print processed[i]
            masukan = processed[i].split(" ")
            d = dict()
            for word in masukan:
                if word in feature_names:
                    idx = feature_names.index(word)
                    d[word] = coef[idx]
            sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
            bully_word = []
            for data in sorted_d:
                if data[1] >= 1:
                    bully_word.append(data[0])
            #print bully_word
            results['bully_words'].append(bully_word)


    resultcounter = Counter(predicted)
    isbully = resultcounter.most_common(1)[0]
    results['isbully'] = isbully

    return results