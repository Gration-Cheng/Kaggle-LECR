import pandas as pd
from Tips import Topic,ContentItem
from tqdm.auto import tqdm
from tqdm.contrib import tzip

# from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
# losses.ContrastiveLoss(model=model)
data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")
content = pd.read_csv("../data/content.csv")
topic_trans = pd.read_csv("./topic_trans.csv")
content_trans = pd.read_csv("./content_trans.csv")
content.set_index("id",inplace=True)
content_trans.set_index("id", inplace=True)
print(data)
test_topic_id = test_data["topic_id"]
test_topic_title = test_data["title"]
test_Y = test_data["content_ids"]
test_predictions = test_data["predictions"]

train_topic_id = data["topic_id"]
train_topic_title = data["title"]
train_Y = data["content_ids"]
train_predictions = data["predictions"]

#
# topics_id = []
# content_id = []
# pair1 = []
# pair2 = []
# label = []
# text = []
# #######################stage2#######################
# for id,title,y,prediction in tzip(train_topic_id,train_topic_title,train_Y,train_predictions):
#     prediction = prediction.split(" ")
#     y = y.split(" ")
#     for i in prediction:
#         if i in y:
#             continue
#         # # text = Topic(id).title + '[SEP]' + Topic(id).get_breadcrumbs() + '[SEP]' + Topic(id).language + '[SEP]' + Topic(id).description
#         topics_id.append(id)
#         content_id.append(i)
#         # pair1.append(title)
#         # # text = ContentItem(i).title + '[SEP]' + "".join(ContentItem(i).get_all_breadcrumbs())+ '[SEP]' + ContentItem(i).language + '[SEP]' + ContentItem(i).description
#         # pair2.append(content_trans.loc[i]['title'])
#         text.append(title + '[SEP]' + content_trans.loc[i]['title'])
#         label.append(0)
#     for i in y:
#     #     text = Topic(id).title + '[SEP]' + Topic(id).get_breadcrumbs() + '[SEP]' + Topic(id).language + '[SEP]' + Topic(id).description
#         topics_id.append(id)
#         content_id.append(i)
#         # pair1.append(title)
#     #     text = ContentItem(i).title + '[SEP]' + "".join(ContentItem(i).get_all_breadcrumbs())+ '[SEP]' + ContentItem(i).language + '[SEP]' + ContentItem(i).description
#     #     pair2.append(content_trans.loc[i]['title'])
#         text.append(title + '[SEP]' + content_trans.loc[i]['title'])
#         label.append(1)
#
# stage2_train = pd.DataFrame(
#         {'topics_ids': topics_id,
#          'content_ids': content_id,
#          'text':text,
#          'target': label}
#     )
# stage2_train.to_csv("./stage2_train.csv")
#
#
topics_id = []
content_id = []
pair1 = []
pair2 = []
text = []
label = []

for id,title,y,prediction in tzip(test_topic_id,test_topic_title,test_Y,test_predictions):
    prediction = prediction.split(" ")
    y = y.split(" ")
    for i in prediction:
        if i in y:
            label.append(1)
        else:
            label.append(0)
        # # text = Topic(id).title + '[SEP]' + Topic(id).get_breadcrumbs() + '[SEP]' + Topic(id).language + '[SEP]' + Topic(id).description
        topics_id.append(id)
        content_id.append(i)
        # pair1.append(title)
        # # text = ContentItem(i).title + '[SEP]' + "".join(ContentItem(i).get_all_breadcrumbs())+ '[SEP]' + ContentItem(i).language + '[SEP]' + ContentItem(i).description
        # pair2.append(content_trans.loc[i]['title'])
        text.append(title + '[SEP]' + content_trans.loc[i]['title'])

stage2_val = pd.DataFrame(
        {'topics_ids': topics_id,
         'content_ids': content_id,
         'text': text,
         'target': label}
    )


stage2_val.to_csv("./stage2_val.csv")

# ####stage1####
topics_id = []
content_id = []
pair1 = []
pair2 = []
label = []


####stage1####
for id,title,y,prediction in tzip(train_topic_id,train_topic_title,train_Y,train_predictions):
    prediction = prediction.split(" ")
    y = y.split(" ")
    # for i in prediction:
    #     if i in y:
    #         continue
    #     # # text = Topic(id).title + '[SEP]' + Topic(id).get_breadcrumbs() + '[SEP]' + Topic(id).language + '[SEP]' + Topic(id).description
    #     topics_id.append(id)
    #     content_id.append(i)
    #     pair1.append(title)
    #     # # text = ContentItem(i).title + '[SEP]' + "".join(ContentItem(i).get_all_breadcrumbs())+ '[SEP]' + ContentItem(i).language + '[SEP]' + ContentItem(i).description
    #     pair2.append(content_trans.loc[i]['title'])
    #     label.append(0)
    for i in y:
    #     text = Topic(id).title + '[SEP]' + Topic(id).get_breadcrumbs() + '[SEP]' + Topic(id).language + '[SEP]' + Topic(id).description
        topics_id.append(id)
        content_id.append(i)
        pair1.append(title)
    #     text = ContentItem(i).title + '[SEP]' + "".join(ContentItem(i).get_all_breadcrumbs())+ '[SEP]' + ContentItem(i).language + '[SEP]' + ContentItem(i).description
        pair2.append(content_trans.loc[i]['title'])
        label.append(1)


stage1 = pd.DataFrame(
        {'topics_ids': topics_id,
         'content_ids': content_id,
         'title1': pair1,
         'title2': pair2,}
         # 'target': label}
    )
stage1.to_csv("./stage1_train.csv")
####stage1####