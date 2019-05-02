import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from utils.util import calc_accuracy, generate_embedding


class SkillsEncoder(nn.Module):
    def __init__(self, config):
        super(SkillsEncoder, self).__init__()
        
        #self.th = config.getint('data', 'skill_num_th')
        self.embs = nn.Embedding(2001, config.getint('model', 'skill_dim'))
    
        self.W = nn.Parameter(torch.Tensor(config.getint('model', 'skill_dim'), config.getint('model', 'hidden_size')))


    def forward(self, skills, desc):
        skills = self.embs(skills)
        # skills: batch, skill_num, skill_dim
        # desc: batch, len, hidden_size
        out = skills.matmul(self.W)
        out = torch.bmm(out, torch.transpose(desc, 1, 2))
        out = torch.softmax(out, dim = 2)

        out = torch.bmm(out, desc) # batch, skill_num, hidden_size
        out = torch.cat([skills, out], dim = 2)

        return out.view(out.shape[0], -1)



class CV_pay(nn.Module):
    def __init__(self, config):
        super(CV_pay, self).__init__()
        
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.emb_dim = config.getint("data", "vec_size")  # 300
        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)

        feature_size = config.getint('model', 'hidden_size') * (config.getint('data', 'skill_num_per_data') + 1) + config.getint('model', 'skill_dim') * config.getint('data', 'skill_num_per_data')
        self.desc_encoder = nn.LSTM(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)
        self.skill_encoder = SkillsEncoder(config)

        self.fc = nn.Linear(feature_size, config.getint('data', 'out_dim'))
        

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        description = data['description']
        skills = data['skills']
        label = data['label']

        desc = self.embs(description)

        # skills = self.skill_encoder(skills)
        desc, _ = self.desc_encoder(desc)
        skills = self.skill_encoder(skills, desc)
        
        desc = torch.max(desc, dim = 1)[0]

        feature = torch.cat([desc, skills], dim = 1)
        y = self.fc(feature)
        
        loss = criterion(y, label)
        accu, acc_result = calc_accuracy(y, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                        "accuracy_result": acc_result}



