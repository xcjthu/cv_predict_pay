import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from utils.util import calc_accuracy, generate_embedding


class SkillsEncoder(nn.Module):
    def __init__(self, config):
        super(SkillsEncoder, self).__init__()
        torch.manual_seed(1)
        self.skill2id = json.load(open(config.get('data', 'skill2id'), 'r'))
        #self.th = config.getint('data', 'skill_num_th')
        self.embs = nn.Embedding(len(self.skill2id) + 1, config.getint('data', 'vec_size'))
        self.init_skill_emb(config)
        
        self.hidden = config.getint('model', 'hidden_size')
        self.vec_size = config.getint('data', 'vec_size')
        self.W = nn.Parameter(torch.Tensor(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size')))

        self.work_skill = nn.Parameter(torch.Tensor(self.hidden + self.vec_size, self.hidden))
        torch.nn.init.xavier_normal(self.W, gain=1)
        torch.nn.init.xavier_normal(self.work_skill, gain = 1)
    
    def init_skill_emb(self, config):
        skill2id = json.load(open(config.get('data', 'skill2id'), 'r'))
        skill2vec = json.load(open(config.get('data', 'skill2vec'), 'r'))
        emb = np.zeros((len(skill2id) + 1, config.getint('data', 'vec_size')))
        for s in skill2id.keys():
            emb[skill2id[s]] = np.array(skill2vec[s])
        self.embs.weight.data.copy_(torch.from_numpy(emb))


    def forward(self, skills, desc, work):
        # print(skills.shape)

        skills = self.embs(skills)
        # skills: batch, skill_num, skill_dim
        # desc: batch, len, hidden_size
        #print(skills)

        # print(self.W)
        # exit(0)
        out = skills.matmul(self.W)
        out = torch.bmm(out, torch.transpose(desc, 1, 2))
        out = torch.softmax(out, dim = 2)

        out = torch.bmm(out, desc) # batch, skill_num, hidden_size
        out = torch.cat([skills, out], dim = 2) # batch, skill_num, hidden_size + vec_size

        rel_ma = out.matmul(self.work_skill)
        rel_ma = torch.bmm(work, torch.transpose(rel_ma, 1, 2))
        rel_ma_skill, _ = torch.max(rel_ma, dim = 1)
        rel_ma_work, _ = torch.max(torch.transpose(rel_ma, 1, 2), dim = 1)

        rel_ma_work = torch.softmax(rel_ma_work, dim = 1)

        rel_ma_skill = torch.softmax(rel_ma_skill, dim = 1)
        
        out = torch.bmm(rel_ma_skill.unsqueeze(1), out).squeeze(1)
        work_out = torch.bmm(rel_ma_work.unsqueeze(1), work).squeeze(1)
        out = torch.cat([out, work_out], dim = 1)
        # print(out.shape)
        #return out.view(out.shape[0], -1)
        # return out
        return out
        # return torch.max(out, dim = 1)[0]




class CV_pay(nn.Module):
    def __init__(self, config):
        super(CV_pay, self).__init__()
        
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.emb_dim = config.getint("data", "vec_size")  # 300
        self.embs = nn.Embedding(self.word_num, self.emb_dim)

        self.work_num = config.getint('data', 'work_history_num')

        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)

        # feature_size = config.getint('model', 'hidden_size') * (config.getint('data', 'skill_num_per_data') + 1) + config.getint('data', 'vec_size') * config.getint('data', 'skill_num_per_data')
        feature_size = config.getint('model', 'hidden_size') * 3 + config.getint('data', 'vec_size')
        self.desc_encoder = nn.LSTM(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)
        self.work_encoder = nn.LSTM(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)


        self.skill_encoder = SkillsEncoder(config)

        self.fc = nn.Linear(feature_size, config.getint('data', 'out_dim'))
        

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        # print(data.keys())
        description = data['description']
        skills = data['skills']
        label = data['label']
        work = data['work_history']

        desc = self.embs(description)
        # print(desc)
        # skills = self.skill_encoder(skills)
        work = work.view(work.shape[0] * self.work_num, work.shape[2])
        work = self.embs(work)

        work, _ = self.work_encoder(work)
        work, _ = torch.max(work, dim = 1) # batch_size * work_num, hidden_size
        work = work.view(desc.shape[0], self.work_num, config.getint('model', 'hidden_size'))

        desc, _ = self.desc_encoder(desc)
        skills = self.skill_encoder(skills, desc, work)
        
        desc = torch.max(desc, dim = 1)[0]
        #print('desc', desc.shape)
        feature = torch.cat([desc, skills], dim = 1)
        y = self.fc(feature)
        
        loss = criterion(y, label)
        accu, acc_result = calc_accuracy(y, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                        "accuracy_result": acc_result}



