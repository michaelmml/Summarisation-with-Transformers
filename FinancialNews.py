import keras
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy

nlp = spacy.load('en_core_web_sm')
import warnings
# from Contractions import contraction_mapping
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


data = pd.read_csv("NewsSum.csv")
print(data.head())

data.drop_duplicates(subset=['Text'], inplace=True)  # dropping duplicates
data.dropna(axis=0, inplace=True)  # dropping na

stop_words = set(stopwords.words('english'))

"""contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}"""

contraction_mapping = {"ain’t": "is not", "aren’t": "are not","can’t": "cannot", "‘cause": "because", "could’ve": "could have", "couldn’t": "could not",
                           "didn’t": "did not",  "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hasn’t": "has not", "haven’t": "have not",
                           "he’d": "he would","he’ll": "he will", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "how’s": "how is",
                           "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have","I’m": "I am", "I’ve": "I have", "i’d": "i would",
                           "i’d’ve": "i would have", "i’ll": "i will",  "i’ll’ve": "i will have","i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would",
                           "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have","it’s": "it is", "let’s": "let us", "ma’am": "madam",
                           "mayn’t": "may not", "might’ve": "might have","mightn’t": "might not","mightn’t’ve": "might not have", "must’ve": "must have",
                           "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have","o’clock": "of the clock",
                           "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",
                           "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",
                           "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have","so’s": "so as",
                           "this’s": "this is","that’d": "that would", "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would",
                           "there’d’ve": "there would have", "there’s": "there is", "here’s": "here is","they’d": "they would", "they’d’ve": "they would have",
                           "they’ll": "they will", "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have",
                           "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are",
                           "we’ve": "we have", "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are",
                           "what’s": "what is", "what’ve": "what have", "when’s": "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
                           "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’s": "who is", "who’ve": "who have",
                           "why’s": "why is", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have",
                           "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all",
                           "y’all’d": "you all would","y’all’d’ve": "you all would have","y’all’re": "you all are","y’all’ve": "you all have",
                           "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have",
                           "you’re": "you are", "you’ve": "you have"}


def prog_sent(text):
    patterns = [r'\b(?i)' + 'financial' + r'\b',
                r'\b(?i)' + 'finance' + r'\b',
                r'\b(?i)' + 'business' + r'\b',
                r'\b(?i)' + 'economy' + r'\b',
                r'\b(?i)' + 'economics' + r'\b']

    output = []
    flag = 0
    for pat in patterns:
        if re.search(pat, text) is not None:
            flag = 1
            break
    return flag


data['filtered'] = data['Text'].apply(prog_sent)
# data = data[data['filtered'] == 1]
data.info()
# data.to_csv('financialnewssum.csv', index=True)


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# 1.Convert everything to lowercase
# 2.Remove HTML tags
# 3.Contraction mapping
# 4.Remove (‘s)
# 5.Remove any text inside the parenthesis ( )
# 6.Eliminate punctuations and special characters
# 7.Remove stopwords
# 8.Remove short words

def text_cleaner(text):
    newString = text.lower()
    newString = re.sub('"', '', newString)
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(' ')])
    newString = re.sub(r"’s\b", "", newString)
    newString = re.sub('[^a-zA-Z\s\-]', '', newString)
    newString = re.sub('[^a-zA-Z\s]', ' ', newString)
    tokens = [w for w in newString.split()]  # if not w in stop_words
    long_words = []
    prev_word = []
    for i in tokens:
        if i not in prev_word and len(i) >= 3:  # removing short word
            long_words.append(i)
            prev_word = [i]
    return (" ".join(long_words)).strip()


cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t))


def summary_cleaner(text):
    newString = text.lower()
    newString = re.sub('"', '', newString)
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(' ')])
    newString = re.sub("’s\b", '', newString)
    newString = re.sub('[^a-zA-Z\s\-]', '', newString)
    newString = re.sub('[^a-zA-Z\s]', ' ', newString)
    # natlanpro = nlp(newString)
    # newString = " ".join([t.text if not t.ent_type_ else t.ent_type_ for t in natlanpro])
    tokens = [w for w in newString.split()]
    long_words = []
    prev_word = []
    for i in tokens:
        if i not in prev_word and len(i) >= 2:  # removing short word
            long_words.append(i)
            prev_word = [i]
    return (" ".join(long_words)).strip()

# Call the above function
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(summary_cleaner(t))

data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary

data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

###################################################################################################################

data.info()
print(data.head())
data.to_csv('financialnewssum.csv', index=True)

# Understanding distribution of summary and text lengths #############################################################
import matplotlib.pyplot as plt

# data = pd.read_csv("financialnewssum.csv", nrows=50000)
data = data[data['filtered'] == 1]

max_len_text = 100
max_len_summary = 20

#################################################################################################################

cleaned_text = np.array(data['cleaned_text'])
cleaned_summary = np.array(data['cleaned_summary'])

for i in range(len(cleaned_text)):
    if len(cleaned_text[i].split()) > max_len_text:
        cleaned_text[i] = ' '.join(cleaned_text[i].split()[:max_len_text])

for i in range(len(cleaned_summary)):
    if len(cleaned_summary[i].split()) > max_len_summary:
        cleaned_summary[i] = ' '.join(cleaned_summary[i].split()[:max_len_summary])

df = pd.DataFrame({'text': cleaned_text, 'summary': cleaned_summary})
# df['summary'] = df['summary'].apply(lambda x: '_START_ ' + x + ' _END_')

text_word_count = []
summary_word_count = []

# populate the lists with sentence lengths
for i in df['text']:
    text_word_count.append(len(i.split()))

for i in df['summary']:
    summary_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})
length_df.hist(bins=50)
plt.show()
print(max(text_word_count), max(summary_word_count))

df.to_csv('financialnewssum2022.csv', index=True)
