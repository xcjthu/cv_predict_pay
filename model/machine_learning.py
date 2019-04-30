import json
import numpy as np
import sklearn 
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
import re
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib

# mode = 'real' 
mode = 'request'

def turnToWordList(sentence):
	#print("text1:", text)
	text = re.sub("[^a-zA-Z]", " ", sentence)
	#print("text2:", text)
	words = text.lower().split()
	return words

def train_tfidf():
	skills, data = read()
	corpus = [' '.join(turnToWordList(d['description'])) for d in data]
	features = 5000
	tfidf = TFIDF(min_df=2, max_features=features, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1,3), use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words="english")
	tfidf.fit(corpus)
	joblib.dump(tfidf, '../../model/tfidf.pkl')


def read():
	f = open('../data/skills.txt', 'r')
	skills = {}
	for s in f.readlines():
		skills[s.strip()] = len(skills)

	f.close()
	f = open('../data/data.json', 'r')
	data = [json.loads(d) for d in f.readlines()]
	f.close()

	return skills, data

def read_tfidf():
	tfidf = joblib.load('../../model/tfidf.pkl')
	return tfidf


def get_y(num):
	try:
		ret = int(num)
		if mode == 'real':
			if ret <= 0 or ret > 3:
				ret = None
			else:
				ret = ret - 1
		else:
			if ret <= 0 or ret > 4:
					ret = None
			else:
				ret = ret - 1
			
	except:
		ret = None

	return ret



def vectorlize(line, skills, tfidf):
	x = np.zeros(len(skills))
	for v in line['skills']:
		x[skills[v]] = 1
	tfidf_vec = tfidf.transform([' '.join(turnToWordList(line['description']))])
	tfidf_vec = tfidf_vec.toarray()
	# print(tfidf_vec.shape)
	# tfidf_vec = tfidf_vec.reshape((tfidf_vec.shape[1]))
	x = np.append(tfidf_vec, x)
	# x = tfidf_vec
	if mode == 'real':
		y = get_y(line['log_realized_wage'])
	else:
		y = get_y(line['log_requested_wage'])
	return x, y



if __name__ == '__main__':
	# train_tfidf()
	print('loading')
	skills, data = read()
	tfidf = read_tfidf()

	X = []
	Y = []
	for v in data:
		x, y =vectorlize(v, skills, tfidf)
		if y is None:
			continue
		X.append(x)
		Y.append(y)

	print('training')

	X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.2,random_state=33)
	# model = SVC(kernel='rbf', class_weight='balanced')
	model = sklearn.svm.LinearSVC()
	# model = MNB()

	model.fit(X_train,y_train)
	y_predict = model.predict(X_test)

	print(classification_report(y_test, y_predict))

	print('microPrecision:', metrics.precision_score(y_test, y_predict, average='micro'))
	print('microRecall:', metrics.recall_score(y_test, y_predict, average='micro'))
	print('microF1:', metrics.f1_score(y_test, y_predict, average='micro'))
	print('macroPrecision:', metrics.precision_score(y_test, y_predict, average='macro'))
	print('macroRecall:', metrics.recall_score(y_test, y_predict, average='macro'))
	print('macroF1:', metrics.f1_score(y_test, y_predict, average='macro'))


