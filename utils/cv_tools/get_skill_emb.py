import json
import numpy as np

skill_path = '/data1/private/xcj/cv_pay/data/skills.txt'
skills = []
for l in open(skill_path, 'r'):
    skills.append(l.split()[0].split('-'))

word_list = json.load(open('/data1/private/xcj/cv_pay/data/wordvec/word2id.json', "r"))
embs = np.zeros([len(word_list), 300], dtype=np.float32)

fin = open('/data1/private/xcj/cv_pay/data/wordvec/word2vec.txt', 'r')
# line = fin.readline()
for line in fin:
    line = line.split()
    word = line[0]
    line = line[1:]
    # print(line)
    try:
        for i in range(len(line)):
            embs[word_list[word]][i] = float(line[i])
    except Exception as err:
        print(err)
        # print(line)


skills_embedding = {}
for s in skills:
    a = np.zeros(300)
    for w in s:
        try:
            a += embs[word_list[w]]
        except:
            pass
    a /= len(s)
    skills_embedding['-'.join(s)] = a.tolist()

print(json.dumps(skills_embedding, indent = 2), file = open('/data1/private/xcj/cv_pay/data/skill_emb.json', 'w'))




