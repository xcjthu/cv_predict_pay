import json

data_path = '/data1/private/xcj/cv_pay/data/data.json'
word_vec_path = '/data2/private/gty/glove.840B.300d.txt'

f = open(data_path, 'r')
datas = [json.loads(d) for d in f.readlines()]

words = []
for d in datas:
    words += d['description'].split()
    for s in d['skills']:
        words += s.split('-')

words = set(words)
words.add('UNK')
words.add('PAD')
word2id = {}
#for w in words:
#    word2id[w] = len(word2id)

print(len(word2id), len(words))


c = 0
fout = open('/data1/private/xcj/cv_pay/data/wordvec/word2vec_new.txt', 'w')
for line in open(word_vec_path, 'r'):
    w = line.split()[0]
    if w in words:
        if w in word2id.keys():
            continue
        print(line.strip(), file = fout)
        word2id[w] = len(word2id)
        c += 1

print(len(word2id), c)
print(json.dumps(word2id), file = open('/data1/private/xcj/cv_pay/data/wordvec/word2id_new.json', 'w'))





