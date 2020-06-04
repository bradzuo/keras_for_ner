# coding:utf-8
import numpy as np
import os,sys,json
curPath = os.path.abspath(os.path.dirname(__file__))
print('curPath',curPath)
sys.path.append(curPath)
from Proc_data import process_data
from Ner import ner_model

def entity_result(per,loc,org,tm):

    ent_person = {'tag':'PER','name':'人物','entity':per}
    ent_location = {'tag': 'LOC', 'name': '地点', 'entity': loc}
    ent_organ = {'tag':'ORG','name':'机构','entity':org}
    ent_time = {'tag':'TM','name':'时间','entity:':tm}
    ent_results = {'Person':ent_person,'Organ':ent_organ,'Time':ent_time,'Location':ent_location}
    for i,j in zip([per,loc,org,tm],['Person','Location','Organ','Time']):
        if i == '':
            ent_results.pop(j)
    return json.dumps(ent_results)

model, num_words= ner_model(Train=False)
# model.load_weights,一定不能model=model.load_weights,该处错误导致模型无法预测,debug耗费了很多时间
model.load_weights(curPath + '/Model/crfApril.h5')
entity_tags = ['O', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', "B_ORG", "I_ORG",'B_TIME','I_TIME']

try:
    params = sys.argv[1]
    # print(params, type(params))
    params = json.loads(params)

    pred_text = params.get('text')
except IndexError:
    pred_text = '抽水蓄能电站项目获批,近日，由中国南方电网有限责任公司全资建设的梅州（五华）抽水蓄能电站获得国家发展和改革委的批准，同意该项目开展前期建设工作。这是五华招商引资的重大突破，对于推动该县向低碳、环保型产业转变，打造低碳电力产业基地，壮大该县县域经济，加快推动绿色的经济崛起将起到重大作用。|该工程规划装机240万千瓦，分期建设，首期建设4台30万千瓦的立轴单级混流可逆式机组，总装机容量120万千瓦。该项目总投资978102万元，首台机组可望于2015年建成投产，全部机组将于2020年建成投产。|梅州抽水蓄能电站站址地处五华县南部的龙村镇黄狮村境内，省委省政府在粤北山区工作会议上将其列入省主要重大项目，省十一届人大三次会议上正式列入省低碳能源经济发展项目，是实现省委省政府区域协调发展战略的优秀项目。建成投产后，年产值将达12.864亿元，必将大大增强五华县的经济实力并促进其经济结构调整。该项目还具有旅游、科普教育价值。电站所在的地方沿途山势蜿蜒起伏、层峦叠嶂，山上树木郁郁葱葱，空气清新怡人，是天然避暑休闲胜地。电站全部建成后，将会增添“高峡出平湖”的胜景，将极大提升五华的生态旅游和投资环境，使电站和地方实现“双赢”。（万自明廖伟军）|420,江北截流输污工程进展顺利,近日从'
# print('测试文本',pred_text)
print(len(pred_text))

str, length = process_data().pred_pad(pred_text[:5154])
# print(str.shape)
# 如果预测数据未达到maxlen,就会存在padding的情况,导致预测结果中存在不是预测数据的预测结果,而采用的pre的padding方式,故
# 要[-len(预测数据):]来提取预测结果
raw = model.predict(str)[0][-length:]
# print(raw)
result = [np.argmax(row) for row in raw]
result_tags = [entity_tags[i] for i in result]
# print(result_tags)
per, loc, org,tm= '', '', '', ''

cur_index = -1
loc_index = -1
changed_index = [0]
for s, t in zip(pred_text, result_tags):
    index_caps = cur_index - loc_index
    # print(index_caps)
    cur_index += 1
    if t in ('B_PER', 'I_PER'):
        per += ' ' + s if (t == 'B_PER') else s
    if t in ('B_ORG', 'I_ORG'):
        org += ' ' + s if (t == 'B_ORG') else s
    if t in ('B_LOC', 'I_LOC'):
        loc_index += 1
        assemble_entity_index = cur_index - loc_index
        changed_index.append(assemble_entity_index)
        # print(assemble_entity_index)
        if index_caps == assemble_entity_index and assemble_entity_index == changed_index[-2]:
            # print('here1',s)
            loc += s
        else:
            # print('here2', s)
            loc += ' ' + s
    if t in ('B_TIME', 'I_TIME'):
        tm += ' ' + s if (t == 'B_TIME') else s

ent_result= entity_result(per,loc,org,tm)
print(ent_result)
print(json.loads(ent_result))