# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:02:06 2019

@author: Administrator

Spam Ham Email Classifier
"""
########################### Downlaod data from internet ############################
import os
import tarfile
from six.moves import urllib
import time

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    time1=time.time()
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()
    time2=time.time()
    print("Download successfully,use time %.2fs."%(time2-time1))
    
################################## load data #######################################
def get_email():
    print(">> Starting get email,please wait some seconds...")
    time1=time.time()
    if os.path.exists(SPAM_PATH):
        pass
    else:
        print("No email locally, need to download from internet, please wait...")
        fetch_spam_data()
    
    HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
    SPAM_DIR = os.path.join(SPAM_PATH, "spam")
    ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
    
    import email
    import email.policy
    
    def load_email(is_spam, filename, spam_path=SPAM_PATH):
        directory = "spam" if is_spam else "easy_ham"
        with open(os.path.join(spam_path, directory, filename), "rb") as f:
            return email.parser.BytesParser(policy=email.policy.default).parse(f)
        
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    time2=time.time()
    print("Finished! use time %.2fs,ham_email: %d, spam_email: %d"%(time2-time1,
                                        len(ham_filenames),len(spam_filenames)))
    print("-"*40)
    return ham_emails,spam_emails
# print the email content
ham_emails,spam_emails=get_email()
#def print_email(email,name="ham-email"):
#    print("="*35+name+"="*35+"\n")
#    print(email.get_content().strip())
#    print("="*80+"\n")
#print_email(ham_emails[1])
#print_email(spam_emails[6],"spam-email")

############################# check the emails structures ##########################
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()
    
from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

#print("\n>> Ham-email structures:\n"+"-"*40)
#from pprint import pprint
#pprint(structures_counter(ham_emails).most_common())
#print("\n>> Spam-email structures:\n"+"-"*40)
#pprint(structures_counter(spam_emails).most_common())
#
## check out the email header
#print("\n>> Check out a spam-email headers:\n"+"-"*40)
#for header, value in spam_emails[0].items():
#    print(header,":",value)
#print("Subject header:","《",spam_emails[0]["Subject"],"》")
#del header;del value

############################# Split train data and test data #######################
print(">> Starting split the data to train&test...")
import numpy as np
from sklearn.model_selection import train_test_split

X=np.array(ham_emails+spam_emails)
y=np.array([0]*len(ham_emails)+[1]*len(spam_emails))
del ham_emails;del spam_emails
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X;del y
print("finished!")

################################## clean the email #################################
import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

#html_spam_emails = [email for email in X_train[y_train==1]
#                    if get_email_structure(email) == "text/html"]
#sample_html_spam = html_spam_emails[7]
#print("Raw Html:\n",sample_html_spam.get_content().strip()[:1000], "...")
#print("Cleaned content:\n",html_to_plain_text(sample_html_spam.get_content())[:1000], "...")

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)
# Final clean function: can transform html to plain-text regardless of what the type is    
#print("The final contest:\n",email_to_text(sample_html_spam)[:100], "...")

# remove the word suffix
try:
    import nltk

    stemmer = nltk.PorterStemmer()
#    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
#        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None
    
# get the url from text
try:
    import urlextract # may require an Internet connection to download root domain names
    
    url_extractor = urlextract.URLExtract()
#    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None
    
############################ Transform the email to counter ########################
from sklearn.base import BaseEstimator, TransformerMixin

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)
    
#X_few = X_train[:3]
#X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
#print("Email to Counter:\n",X_few_wordcounts)

####################### transform the counter to sparse matrix #####################
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

#vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
#X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
#print("sparse matrix result:\n",X_few_vectors.toarray())
#print("Vacubularies:\n",vocab_transformer.vocabulary_)

####################### Pipeline: Prepare the datasets #############################
print("\n>> Starting processing the raw data...")
time.sleep(0.5)
time1=time.time()
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

from sklearn.externals import joblib    
if os.path.exists("./model_set/chapter03_spam_classifier.pkl"):
    log_clf=joblib.load("./model_set/chapter03_spam_classifier.pkl")
    print("Load the local LogisticRegression model sucessfully!")
else:
    X_train_transformed = preprocess_pipeline.fit_transform(X_train)
    time2=time.time()
    print("Finished! Use time %.2fs."%(time2-time1))
    ############################### train the model ####################################
    print("\n>> Starting training the model...")
    time.sleep(0.5)
    time1=time.time()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    log_clf = LogisticRegression(solver="liblinear", random_state=42)
    score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
    time2=time.time()
    print("Finished! Use time %.2fs,The LogisticRegression mean-score:"%(time2-time1),score.mean())
    
    #log_clf = LogisticRegression(solver="liblinear", random_state=42)
    log_clf.fit(X_train_transformed, y_train)
    # save model
    joblib.dump(log_clf,"./model_set/chapter03_spam_classifier.pkl")

X_test_transformed = preprocess_pipeline.transform(X_test)
y_pred = log_clf.predict(X_test_transformed)

########################### check the model usefull or not #########################
from sklearn.metrics import precision_score, recall_score
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

################################# confusion matrix #################################
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
import seaborn as sns
sns.set(font="SimHei")

from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_test, y_pred)
def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.grid(False)
    
# 绘制混淆矩阵
plot_confusion_matrix(conf_mx)
    
    
    