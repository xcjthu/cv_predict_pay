import os
import json
import random
import jieba
import jieba.posseg as psg

#pre_path = "/data/disk3/private/zhx/exam/data/origin_data/学法"
pre_path = "/home/zhx/"
file_list = ["all.json"]
#file_list = ["xuefa_data_1.json", "xuefa_data_2.json", "xuefa_data_3.json"]
#output_path = "/data/disk3/private/zhx/exam/data/origin_data/format"
#output_path = "/data/disk1/private/xcj/exam/data/origin_data/"
output_path = "/data/disk1/private/xcj/exam/data/all_data/gen/"

def contain_name(sentence):
    ret = psg.cut(sentence)
    ret = [x.flag for x in ret]
    if 'nr' in ret:
        return True
    return False


def check(d):
    if contain_name(d['statement']):
        return True

    for x in ["甲", "乙", "丙", "丁", "某", "某某", "A", "B", "C", "D", "O", "P", "S", "Q", "张三", "李四"]:
        if d["statement"].find(x) != -1:
            return True
        for option in d["option_list"].keys():
            if d["option_list"][option].find(x) != -1:
                return True
    return False


def dump(data, filename):
    print(filename, len(data))
    f = open(os.path.join(output_path, filename), "w")
    for d in data:
        print(json.dumps(d, ensure_ascii=False, sort_keys=True), file=f)


map_dic = {
    "国际法": 4,
    "刑法": 2,
    "刑事诉讼法【最新更新】": 2,
    "司法制度和法律职业道德": 1,
    "法制史": 5,
    "民法": 3,
    "民诉与仲裁【最新更新】": 3,
    "国际经济法": 4,
    "法理学": 1,
    "法考冲刺试题": 0,
    "法考真题(按年度)": 0,
    "国际私法": 4,
    "社会主义法治理念": 1,
    "商法": 3,
    "民诉与仲裁【更新中】": 3,
    "行政法与行政诉讼法": 2,
    "宪法": 1,
    "经济法": 4,
}

if __name__ == "__main__":
    data = [[[], []], [[], []]]
    for filename in file_list:
        f = open(os.path.join(pre_path, filename), "r")

        for line in f:
            d = json.loads(line)
            if check(d):
                type2 = 1
            else:
                type2 = 0

            if random.randint(1, 5) == 1:
                type3 = 1
            else:
                type3 = 0

            data[type2][type3].append(d)


    for b in range(0, 2):
        for c in range(0, 2):
            if c == 0:
                x = "train"
            else:
                x = "test"
            dump(data[b][c], "%d_%s.json" % (b, x))


