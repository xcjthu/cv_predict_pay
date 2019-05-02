import json
import torch
import numpy as np
import re


class CV_formatter:
    def __init__(self, config):
        super(CV_formatter, self).__init__()
        
        path = config.get('data', 'skill_list')
        f = open(path, 'r')
        
        self.th = config.getint('data', 'skills_th')
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))
        self.max_len = config.getint("data", "sent_max_len")

        self.skills = {}
        for l in f:
            l = l.strip().split('\t')
            if int(l[1]) >= self.th:
                self.skills[l[0]] = len(self.skills)


    def lookup(self, data, max_len):
        lookup_id = []
        #print(data)
        for word in data:
            #print(word)
            try:
                lookup_id.append(self.word2id[word])
            except:
                lookup_id.append(self.word2id["UNK"])
        #print('max_len:', max_len)
        while len(lookup_id) < max_len:
            lookup_id.append(self.word2id["PAD"])
        lookup_id = lookup_id[:max_len]

        return lookup_id


    def turnTowordlist(self, sentence):
        text = re.sub("[^a-zA-Z]", " ", sentence)
        words = text.lower().split()
        return words


    def check(self, data, config):
        data = json.loads(data)
        if len(data['description']) == 0:
            return None
        skills = []
        for v in data['skills']:
            if v in self.skills:
                skills.append(self.skills[v])

        if len(skills) == 0:
            return None
        try:
            data['label'] = self.get_label(data, config)
            if (data['label'] is None):
                return None
            # request = float(data['log_realized_wage'])
            # real = float(data['log_requested_wage'])
        except Exception as err:
            print(err)
            return None
        
        data['skills'] = skills

        return data
 
    
    def get_label(self, data, config):
        try:
            if config.get('data', 'label') == 'request':
                y = data['log_requested_wage']
                y = int(y)
                if y <= 0 or y > 4:
                    return None
                else:
                    return y - 1
            else:
                y = data['log_realized_wage']
                y = int(y)
                if y <= 0 or y > 3:
                    return None
                else:
                    return y - 1
        except:
            return None


    def pad_skill(self, l, length):
        while len(l) < length:
            l.append(len(self.skills))
        l = l[:length]
        return l


    def format(self, data, config, transformer, mode):
        desc = []
        skills = []

        label = []

        for d in data:
            desc.append(self.lookup(self.turnTowordlist(d['description']), self.max_len))
            
            skills.append(self.pad_skill(d['skills'], config.getint('data', 'skill_num_per_data')))
            label.append(d['label'])

        desc = torch.tensor(desc, dtype = torch.long)
        skills = torch.tensor(skills, dtype = torch.long)
        label = torch.tensor(label, dtype = torch.long)

        return {'description': desc, 'skills': skills, 'label': label}
            
