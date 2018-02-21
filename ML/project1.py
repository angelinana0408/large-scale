import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import pos_tag
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


##  (a) 
graphics = fetch_20newsgroups(subset='train', categories=['comp.graphics'], shuffle=True, random_state=42)
mswindows = fetch_20newsgroups(subset='train', categories=['comp.os.ms-windows.misc'], shuffle=True, random_state=42)
pchardware = fetch_20newsgroups(subset='train', categories=['comp.sys.ibm.pc.hardware'], shuffle=True, random_state=42)
machardware = fetch_20newsgroups(subset='train', categories=['comp.sys.mac.hardware'], shuffle=True, random_state=42)
autos = fetch_20newsgroups(subset='train', categories=['rec.autos'], shuffle=True, random_state=42)
motorcycles = fetch_20newsgroups(subset='train', categories=['rec.motorcycles'], shuffle=True, random_state=42)
baseball = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball'], shuffle=True, random_state=42)
hockey = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey'],shuffle=True, random_state=42)

frequency=[len(graphics.data),len(mswindows.data),len(pchardware.data),len(machardware.data),len(autos.data),len(motorcycles.data),len(baseball.data),len(hockey.data)]
names=('graphics','ms-windows','pc.hardware','mac.hardware','autos','motorcycles','baseball','hockey')
y_pos=np.arange(len(names))
plt.bar(y_pos,frequency, align='center',alpha=1)
plt.xticks(y_pos,names,fontsize = 6)
plt.ylabel('number of tracking doc')
plt.title('number of tracking doc in 8 classes')
plt.figure(1)

##  (b)
print('(b)------------------------------------------------------------')
stop_words_skt = text.ENGLISH_STOP_WORDS
catergories=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos'
             ,'rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
all_classes=fetch_20newsgroups(subset='train', categories=catergories, shuffle=True, random_state=42)


combined_stopwords = set.union(set(punctuation),set(stop_words_skt))

## function for tokenize and stemming and remove punctuation 
def stemmed_words(doc):
    stemmer = SnowballStemmer("english")
    return (stemmer.stem(w) for w in doc)
def stem_rmv_punc(doc):
    token=""
    for char in doc:
        if (char not in punctuation):token+=char
    tokens=token.split()
    return (word for word in stemmed_words(tokens) if word not in combined_stopwords and not word.isdigit())

## min_df=2
count_vect1 =  CountVectorizer(tokenizer=stem_rmv_punc,min_df=2)
X_train_counts1 = count_vect1.fit_transform(all_classes.data)
tfidf_transformer = TfidfTransformer()
X_train_counts1_tfid=tfidf_transformer.fit_transform(X_train_counts1)
print ('with min_df=2: \n')
print (X_train_counts1_tfid.shape)

## min_df=5 
count_vect2 = CountVectorizer(min_df=5,analyzer=stem_rmv_punc)
X_train_counts2 = count_vect2.fit_transform(all_classes.data)
tfidf_transformer2 = TfidfTransformer()
X_train_counts2_tfid=tfidf_transformer2.fit_transform(X_train_counts2)
print ('with min_df=5: \n')
print (X_train_counts2_tfid.shape)

##(c)
print('(c)------------------------------------------------------------')
categories = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
for i in categories:
    all_group = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)
    all_group_vector= TfidfVectorizer(analyzer=stem_rmv_punc,max_features=10)
    sub_group = fetch_20newsgroups(subset='train',categories= [i],shuffle=True,random_state=42)
    tsub_group_tfidf = all_group_vector.fit_transform(sub_group.data)
    print(all_group_vector.vocabulary_.keys())

##(d)
print('(d)------------------------------------------------------------')
model_LSI1 = TruncatedSVD(n_components=50,random_state=0,algorithm='arpack')
model_LSI2 = TruncatedSVD(n_components=50,random_state=0,algorithm='arpack')
LSI1_train = model_LSI1.fit_transform(X_train_counts1_tfid)
LSI2_train = model_LSI2.fit_transform(X_train_counts2_tfid)
print ('LSI with min_df=2 \n')
print (LSI1_train)
print ('LSI with min_df=5 \n')
print (LSI2_train)


NMF1_model = NMF(n_components=50,random_state=0)
NMF2_model = NMF(n_components=50,random_state=0)
NMF1_train = NMF1_model.fit_transform(X_train_counts1_tfid)
NMF2_train = NMF2_model.fit_transform(X_train_counts2_tfid)
print('NMF with min_df=2 \n')
print (NMF1_train)
print('NMF with min_df=5 \n')
print (NMF2_train)

## (e)
## function used to plot ROC curve, calculate confusion matrix, print out matrix, accuracy, recall, precision
def evaluate(test_label,svc_score): 
    a, b, _ = roc_curve(test_label,svc_score)
    roc_area = auc(a,b)
    plt.figure()
    plt.plot(a,b,lw=2,label='area = %0.4f' % roc_area)
    plt.xlim([0.0,1.3])
    plt.ylim([0.0,1.3])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
 
    confusion = confusion_matrix(test_label,svc_prediction)
    print ('\tcomp\trec')
    print ('comp\t',confusion[0][0],'\t',confusion[0][1])
    print ('rec\t',confusion[1][0],'\t',confusion[1][1])
    accuracy = accuracy_score(test_label,svc_prediction)
    recall = recall_score(test_label,svc_prediction)
    precision = precision_score(test_label,svc_prediction)
    print("accuracy: " ,accuracy)
    print("recall: " ,recall)
    print("precision: " , precision)


test=fetch_20newsgroups(subset='test', categories=catergories, shuffle=True, random_state=42)
#min_df=2
test_counts1 = count_vect1.transform(test.data)
test_tfid1 = tfidf_transformer.transform(test_counts1)
LSI1_test=model_LSI1 .transform(test_tfid1)
#min_df=5
test_counts2 = count_vect2.transform(test.data)
test_tfid2 = tfidf_transformer2.transform(test_counts2)
LSI2_test=model_LSI2 .transform(test_tfid2)


train_label=[int (x/4) for x in all_classes.target]
test_label=[int(y/4) for y in test.target]

## hard-margin with min_df=2
linear_svc = svm.SVC(kernel='linear',probability=True,C=1000)
linear_svc.fit(LSI1_train,train_label)
svc_prediction = linear_svc.predict(LSI1_test)
svc_score = linear_svc.decision_function(LSI1_test)
print('\n hard-margin with min_df=2: \n')
evaluate(test_label,svc_score)

## hard-margin with min_df=5
linear_svc = svm.SVC(kernel='linear',probability=True,C=1000)
linear_svc.fit(LSI2_train,train_label)
svc_prediction = linear_svc.predict(LSI2_test)
svc_score = linear_svc.decision_function(LSI2_test)
print('\n hard-margin with min_df=5: \n')
evaluate(test_label,svc_score)

## soft-margin with min_df=2
linear_svc = svm.SVC(kernel='linear',probability=True,C=0.001)
linear_svc.fit(LSI1_train,train_label)
svc_prediction = linear_svc.predict(LSI1_test)
svc_score = linear_svc.decision_function(LSI1_test)
print('\n soft-margin with min_df=2: \n')
evaluate(test_label,svc_score)

## soft-margin with min_df=5
linear_svc = svm.SVC(kernel='linear',probability=True,C=0.001)
linear_svc.fit(LSI2_train,train_label)
svc_prediction = linear_svc.predict(LSI2_test)
svc_score = linear_svc.decision_function(LSI2_test)
print('\n soft-margin with min_df=5: \n')
evaluate(test_label,svc_score)

## (f)
cross_list = []
for c in [0.001,0.01,0.1,1,10,100,1000]:
    linear_svc = svm.LinearSVC(C=c)
    linear_svc.fit(LSI1_train,train_label)
    svc_prediction = linear_svc.predict(LSI1_test)
    svc_score = linear_svc.decision_function(LSI1_test)
    fcv_scores = cross_val_score(linear_svc, LSI1_test, test_label, cv=5)
    average_score = sum(fcv_scores)/len(fcv_scores)
    cross_list.append([average_score,c])
cross_list = sorted(cross_list)

best_c=cross_list[len(cross_list)-1][1]
print ('\n best value is: ',best_c)
print('\ncontruct model with best value')
linear_svc = svm.LinearSVC(C=c)
linear_svc.fit(LSI1_train,train_label)
svc_prediction = linear_svc.predict(LSI1_test)
svc_score = linear_svc.decision_function(LSI1_test)
evaluate(test_label,svc_score)

##(g)
## train the test data with NMF
print ('(g)---------------------------------------------')
NMF1_test = NMF1_model.transform(test_tfid1)
NMF2_test = NMF2_model.transform(test_tfid2)
##min_df=2
clf = GaussianNB().fit(NMF1_train,train_label)
svc_prediction = clf.predict(NMF1_test)
svc_score = clf.predict(NMF1_test)
evaluate(test_label,svc_score)

##min_df=5
clf = GaussianNB().fit(NMF2_train,train_label)
svc_prediction = clf.predict(NMF2_test)
svc_score = clf.predict(NMF2_test)
evaluate(test_label,svc_score)

## (h)
print('(h)------------------------------------------')
print('Logistic Regression with min_df=2: \n')
clf= linear_model.LogisticRegression().fit(LSI1_train,train_label)
svc_prediction = clf.predict(LSI1_test)
svc_score = clf.decision_function(LSI1_test)
evaluate(test_label,svc_score)

print('Logistic Regression with min_df=5: \n')
clf= linear_model.LogisticRegression().fit(LSI2_train,train_label)
svc_prediction = clf.predict(LSI2_test)
svc_score = clf.decision_function(LSI2_test)
evaluate(test_label,svc_score)

## (i)
cross_list = []
for c in [0.001,0.01,0.1,1,10,100,1000]:
    linear_svc = linear_model.LogisticRegression(penalty='l1',C=c).fit(LSI1_train,train_label)
    linear_svc.fit(LSI1_train,train_label)
    svc_prediction = linear_svc.predict(LSI1_test)
    svc_score = linear_svc.decision_function(LSI1_test)
    fcv_scores = cross_val_score(linear_svc, LSI1_test, test_label, cv=5)
    average_score = sum(fcv_scores)/len(fcv_scores)
    cross_list.append([average_score,c])
cross_list = sorted(cross_list)

best_c=cross_list[len(cross_list)-1][1]
print ("best value is : ", best_c)
print ('construct model with best_c ,l1 and min_df=2')
linear_svc = linear_model.LogisticRegression(penalty='l1',C=best_c).fit(LSI1_train,train_label)
linear_svc.fit(LSI1_train,train_label)
svc_prediction = linear_svc.predict(LSI1_test)
svc_score = linear_svc.decision_function(LSI1_test)
evaluate(test_label,svc_score)
print ('construct model with best_c ,l2 and min_df=2')
linear_svc = linear_model.LogisticRegression(penalty='l2',C=best_c).fit(LSI1_train,train_label)
linear_svc.fit(LSI1_train,train_label)
svc_prediction = linear_svc.predict(LSI1_test)
svc_score = linear_svc.decision_function(LSI1_test)
evaluate(test_label,svc_score)

## (i) Naive bayes
categories = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
train=fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test=fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
vect = CountVectorizer(min_df=2,tokenizer=stem_rmv_punc)
train_counts = vect.fit_transform(train.data)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
LSI= TruncatedSVD(n_components=50, algorithm='arpack')
LSI_train=LSI.fit_transform(train_tfidf)

NMF_model = NMF(n_components=50,random_state=0)
NMF_train = NMF_model.fit_transform(train_tfidf)


test_counts = vect.transform(test.data)
test_tfidf = tfidf_transformer.transform(test_counts)
LSI_test=LSI.transform(test_tfidf)

NMF_test = NMF_model.transform(test_tfidf)

clf=GaussianNB().fit(NMF_train,train.target)
svc_prediction = clf.predict(NMF_test)
svc_score = clf.predict(NMF_test)

def multi_evaluate(test_label,svc_prediction): 
 
    confusion = confusion_matrix(test_label,svc_prediction)
    print (confusion)
    accuracy = accuracy_score(test_label,svc_prediction)
    recall = recall_score(test_label,svc_prediction,average='weighted')
    precision = precision_score(test_label,svc_prediction,average='weighted')
    print("accuracy: " ,accuracy)
    print("recall: " ,recall)
    print("precision: " , precision)

multi_evaluate(test.target,svc_prediction)

## (i) SVM one vs one
print('SVM one vs one with LSI: \n')
clf = OneVsOneClassifier(LinearSVC(C=100, random_state=42)).fit(LSI_train,train.target)
svc_prediction = clf.predict(LSI_test)
svc_score = clf.predict(LSI_test)
multi_evaluate(test.target,svc_prediction)

print('SVM one vs one with NMF: \n')
clf = OneVsOneClassifier(LinearSVC(C=100, random_state=42)).fit(NMF_train,train.target)
svc_prediction = clf.predict(NMF_test)
svc_score = clf.predict(NMF_test)
multi_evaluate(test.target,svc_prediction)

## (i) SVM one vs more

print('SVM one vs rest with LSI: \n')
clf = OneVsRestClassifier(LinearSVC(C=100, random_state=42)).fit(LSI_train,train.target)
svc_prediction = clf.predict(LSI_test)
svc_score = clf.predict(LSI_test)
multi_evaluate(test.target,svc_prediction)

print('SVM one vs rest with NMF: \n')
clf = OneVsRestClassifier(LinearSVC(C=100, random_state=42)).fit(NMF_train,train.target)
svc_prediction = clf.predict(NMF_test)
svc_score = clf.predict(NMF_test)
multi_evaluate(test.target,svc_prediction)

plt.show()
