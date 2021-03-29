import csv
import random
import pandas as pd
import numpy as np
import re
import string
import itertools
import nltk
nltk.download('averaged_perceptron_tagger')
from sklearn import svm
from sklearn.metrics import f1_score, classification_report,confusion_matrix, accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk import PorterStemmer as Stemmer
from nltk import pos_tag
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
#print(len(stopWords))
#print(stopWords)

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(500) 
airline = pd.read_csv('Tweets.csv', encoding = 'utf8')
airline.shape 
print(airline.head(5))
airline_sub = airline.loc[:, ['airline_sentiment', 'airline', 'name', 'negativereason', 'text']]
airline_sub.isnull().sum() 
print(airline_sub.shape) 
print(airline_sub['text'].head(100))
sentiment_count = airline_sub['airline_sentiment'].value_counts()  # negative 9178, neutral 3099, positive 2363
print("airline_sub sentiment_count: ", sentiment_count)
airline_sub['label'] = airline_sub['airline_sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
airline_sub['text_length'] = airline_sub['text'].apply(lambda x: len(word_tokenize(x)))  
    
"""def airline_EDA():
    sns.set_style("darkgrid")
    sns.countplot(x = 'airline_sentiment', data = airline_sub, order = airline_sub['airline_sentiment'].value_counts().index, palette = 'Set1')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()   
    sns.set_style("darkgrid")
    g=sns.countplot(x = 'airline', data = airline_sub, hue = 'airline_sentiment', order = airline_sub['airline'].value_counts().index, palette = 'Set2')
    plt.xlabel('Airline')
    plt.ylabel('Frequency')
    plt.legend().set_title('Sentiment')
    for index, row in airline_sub.iterrows():
        g.text(row.neutral,row.positive,row.negative,color='black', ha="center")
    plt.show()
    countt= airline_sub['airline'].value_counts()
    print("count",countt)
            
"""
    
"""def stemmer_text(text):
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer    
    st = Stemmer()  
    text = [st.stem(t) for t in text]
    return (' '.join(text))
    print(text)"""
    
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def lemmatize_text(text):
    lmtzr = WordNetLemmatizer().lemmatize
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer    
    word_pos = pos_tag(text)    
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]
    return (' '.join(lemm_words))
def pre_process(text):      
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)    
    text = emoji_pattern.sub(r'', text)                             
    text = text.lower() 
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text) 
    white_list = ["not", "no", "won't", "isn't", "couldn't", "wasn't", "didn't", "shouldn't", 
                  "hasn't", "wouldn't", "haven't", "weren'-1ikt", "hadn't", "shan't", "doesn't",
                  "mightn't", "mustn't", "needn't", "don't", "aren't", "won't"]
    words = text.split()
    text = ' '.join([t for t in words if (t not in stopwords.words('english') or t in white_list)])  # remove stopwords        
    text = ''.join([t for t in text if t not in string.punctuation])      
    text = ''.join([t for t in text if not t.isdigit()])        
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    text = lemmatize_text(text)      
    return text

      
def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """
    print("Loading Glove Model")
    f = open(glove_file,'r', encoding = "utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model  
count_total = 0   # How many words are in processed data, including duplicate words
count_in = 0      # How many words are in Glove pretrained data
count_out = 0     # How many words are not in Glove pretrained data
out_words_list = []    # words list that not in Glove pretrained data
def tweet_to_vec(tweet, g_model, num_features):    
    
    global count_total, count_in, count_out
    
    word_count = 0
    feature_vec = np.zeros((num_features), dtype = "float32")
   
    for word in tweet.split(' '):
        count_total += 1
        if word in g_model.keys():   
            count_in += 1
            word_count += 1
            feature_vec += g_model[word]
        else:
            count_out += 1
            out_words_list.append(word)
    if (word_count != 0):
        feature_vec /= word_count
    return feature_vec
    print(feature_vec)
    

def gen_tweet_vecs(tweets, g_model, num_features):    
    curr_index = 0
    tweet_feature_vecs = np.zeros((len(tweets), num_features), dtype = "float32")
    
    for tweet in tweets:
        if curr_index % 2000 == 0:
            print('Word2vec vectorizing tweet %d of %d' %(curr_index, len(tweets)))
        tweet_feature_vecs[curr_index] = tweet_to_vec(tweet, g_model, num_features)
        curr_index += 1
    return tweet_feature_vecs

    print(tweet_feature_vecs)

def airline_word2vec_model(df, classifier, g_model):
        
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)   
    global count_total, count_in, count_out
    global out_words_list
    count_total, count_in, count_out = 0, 0, 0 
    out_words_list = []    
    train_vec = gen_tweet_vecs(train_X['processed_text'], g_model, 100)
    test_vec = gen_tweet_vecs(test_X['processed_text'], g_model, 100)
   
    print("Glove word embedding statistic\n", "count_total: %d/" %count_total, "count_in: %d/" %count_in, "count_out: %d/" %count_out)
    print("Number of unique words without embedding: %d" %len(set(out_words_list)))
   # print("Words without embedding: \n", set(out_words_list))
    
    if classifier == "SVM":      
        pipe = make_pipeline(svm.SVC(kernel = 'linear', probability = True, random_state = 101))
        clf = pipe.fit(train_vec, train_X['label'])                 
        test_y_hat = pipe.predict(test_vec)
        
        file_name = 'SVM_word2vec_'                  
   
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_vec), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)

    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    if classifier == "SVM": 
        print("SVM classification report -- ") 
    #elif classifier == "RF":
       # print("RF word2vec classification report -- ")
        
    print(pd.crosstab(test_y.ravel(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))       
    print("-----------------------------------------")
    #print(classification_report(test_y, test_y_hat))
    print("Accuracy -- ")
    print(accuracy_score(test_y, test_y_hat))
   
def main():
    
   # airline_EDA()  
    airline_sub['processed_text'] =  airline_sub['text'].apply(pre_process)  
    airline_sub['processed_text_length'] = airline_sub['processed_text'].apply(lambda x: len(word_tokenize(x)))
    df = pd.DataFrame( airline_sub['processed_text'] ) 
  
      # saving the dataframe 
    df.to_csv('file2.csv', header=False, index=False) 
      
    # with open('processedtxt.csv', 'w', newline='') as file:
       # writer = csv.writer(file)
        #writer.writerows(airline_sub['processed_text'])
    print(airline_sub['processed_text'])
    print(airline_sub['processed_text_length'])     
    
    airline_model = airline_sub.loc[:, ['airline_sentiment', 'text', 'processed_text', 'label']]
    airline_model.to_csv('airline_model.csv')  
    g_model = load_glove_model("glove.6B.100d.txt") 
    airline_word2vec_model(airline_model, "SVM", g_model) 
   
      
   
    

if __name__ == "__main__":
    main() 


    


        