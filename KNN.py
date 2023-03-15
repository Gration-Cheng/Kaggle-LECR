import os
import gc
import time
import math
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib import tzip
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
import cupy as cp
# from cuml.metrics import pairwise_distances
# from cuml.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import util
from sklearn.metrics import pairwise_distances
from Tips import Topic
# %env TOKENIZERS_PARALLELISM = false
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CFG1:
    # model = "./model/all-MiniLM-L6-v2/"
    model = "./Mutiligual_104"
    tokenizer = AutoTokenizer.from_pretrained(model)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    Neighbor = 5
    batch_size = 128
    num_workers = 0
    top_n = 50
def read_data(cfg):
    # topics = pd.read_csv('../data/topics.csv')
    # content = pd.read_csv('../data/content.csv')
    topics = pd.read_csv('./topic_trans.csv')
    content = pd.read_csv('./content_trans.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    correlations = pd.read_csv('../data/correlations.csv')
    # topics = topics.merge(sample_submission, how='inner', left_on='id', right_on='topic_id')
    print(len(correlations))
    topic_sources = pd.read_csv('../data/topics.csv')
    topic_sources.drop(['title','description','channel','level','parent','has_content'],axis=1, inplace=True)
    content_lg = pd.read_csv('E:/PyCharm Community Edition 2021.2.3/project/Learning_Equality/data/content.csv')

    corr_topic = correlations.merge(topic_sources,how='inner',left_on='topic_id',right_on='id')
    corr_topic.set_index("topic_id")
    train = corr_topic[corr_topic['category']=='source']
    test = corr_topic[corr_topic['category']!='source']
    # test = test[0:len(corr_topic)*0.90]
    train = pd.concat([train,test[0:int(len(test)*0.90)]],axis=0)
    test = test[int(len(test)*0.90):]

    # train = correlations[:int(len(correlations)*0.85)]
    # test = correlations[int(len(correlations)*0.85):]
    # train_topics = topics.merge(train,how='inner',left_on='id',right_on='topic_id').drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis=1,
    #              inplace=True)
    # test_topics = topics.merge(test,how='inner',left_on='id',right_on='topic_id').drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis=1,
    #              inplace=True)
    # Fillna titles
    topics['title'].fillna(" ", inplace=True)
    content['title'].fillna(" ", inplace=True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    # topics.sort_values('length', inplace=True)
    # content.sort_values('length', inplace=True)
    # Drop cols
    # topics.drop(
    #     ['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length'], axis=1, inplace=True)
    topics.drop(
        ['length'], axis=1,
        inplace=True)
    # content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis=1,
    #              inplace=True)
    content.drop(['length'], axis=1,
                 inplace=True)
    # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    content = pd.concat([content, content_lg['language']], axis=1)

    # train = correlations[:int(len(correlations)*0.85)]
    # test = correlations[int(len(correlations)*0.85):]
    train_topics = topics.merge(train,how='inner',left_on='id',right_on='topic_id')
    test_topics = topics.merge(test,how='inner',left_on='id',right_on='topic_id')
    print(' ')
    print('-' * 50)
    print(f"trian_topics.shape: {train_topics.shape}")
    print(f"test_topics: {test_topics.shape}")
    languages = content['language'].unique().tolist()
    content_dict = {}
    for lang in languages:
        content_dict[lang] = content[content['language'] == lang].reset_index(drop=True)
    return train_topics,test_topics,content_dict,languages

def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=256
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class topic_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        # state = torch.load("softmax_topic_0.pth", map_location=torch.device('cuda'))
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        # self.model.load_state_dict(state['model'])
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature

class content_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        # state = torch.load("softmax_content_0.pth", map_location=torch.device('cuda'))
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        # self.model.load_state_dict(state['model'])
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature

class unsup_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs



def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    print(round(recall.mean(),4))
    return round(f2.mean(), 4)


train,test,content,languages= read_data(CFG1)
# train_text = train["title"]
# test_text = train["title"]
# content_text = content["title"]
test = test[0:int(len(test)*0.1)]
train_topic = unsup_dataset(train,CFG1)
test_topic= unsup_dataset(test,CFG1)
# content_text = unsup_dataset(content,CFG1)
content_text_dict = {}
for lang in languages:
    content_text = unsup_dataset(content[lang],CFG1)
    content_text_dict[lang] = content_text
train_topics_loader = DataLoader(
    train_topic,
    batch_size=CFG1.batch_size,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=CFG1.tokenizer, padding='longest'),
    num_workers=CFG1.num_workers,
    pin_memory=True,
    drop_last=False
)
train_language = train['language']
test_language = test['language']

test_topics_loader = DataLoader(
    test_topic,
    batch_size=CFG1.batch_size,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=CFG1.tokenizer, padding='longest'),
    num_workers=CFG1.num_workers,
    pin_memory=True,
    drop_last=False
)
content_loader_dict = {}
for lang in languages:
    content_loader = DataLoader(
        content_text_dict[lang],
        batch_size=CFG1.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG1.tokenizer, padding='longest'),
        num_workers=CFG1.num_workers,
        pin_memory=True,
        drop_last=False
)
    content_loader_dict[lang] = content_loader

# content_loader = DataLoader(
#     content_text,
#     batch_size=CFG1.batch_size,
#     shuffle=False,
#     collate_fn=DataCollatorWithPadding(tokenizer=CFG1.tokenizer, padding='longest'),
#     num_workers=CFG1.num_workers,
#     pin_memory=True,
#     drop_last=False
# )

t_model = topic_model(CFG1)
c_model = content_model(CFG1)
# state = torch.load(".-model-all-MiniLM-L6-v2-_epoch.pth", map_location=torch.device('cpu'))
# model.load_state_dict(state['model'])
t_model.to(device)
c_model.to(device)
train_topics_preds = get_embeddings(train_topics_loader, t_model, device)
test_topics_preds = get_embeddings(test_topics_loader, t_model, device)
# content_preds = get_embeddings(content_loader, c_model, device)
content_preds_dict = {}
for lang in languages:
    content_preds = get_embeddings(content_loader_dict[lang], c_model, device)
    # content_preds = pd.DataFrame(content_preds)
    # content_preds = pd.concat([content[lang]['id'],content_preds],axis=1)
    # content_preds = content_preds.set_index('id')
    # content_preds.drop(['id'],axis=1, inplace=True)

    content_preds_dict[lang] = content_preds

for i in [50]:
    top_n = i
    print('Training KNN model...')
        # tokenizers = CFG1.tokenizer
    # neighbors_model = NearestNeighbors(n_neighbors=top_n, metric='cosine',)
    # neighbors_model.fit(content_preds)
    # train_indices = neighbors_model.kneighbors(train_topics_preds, return_distance=False)
    # test_indices = neighbors_model.kneighbors(test_topics_preds, return_distance=False)
    # train_indices = []
    # for topic_embedding,lang in tzip(train_topics_preds,train_language):
    #     cos_scores = util.cos_sim(topic_embedding, content_preds_dict[lang])[0]
    #     top_results = torch.topk(cos_scores, k=top_n)
    #     train_indic = top_results[1].cpu().numpy()
    #     predict_id = content[lang]['id'][train_indic]
    #     train_indices.append(' '.join(predict_id))
    for thres in np.arange(0.65, 0.75, 0.01):
        test_indices = []
        for topic_embedding, lang in tzip(test_topics_preds, test_language):
            # cos_scores = util.cos_sim(topic_embedding, content_preds_dict[lang])[0]
            cos_scores = util.cos_sim(topic_embedding, content_preds_dict[lang])[0]
            # top_results = torch.topk(cos_scores, k=top_n)
            top_results = (cos_scores > thres).nonzero()
            top_results = torch.reshape(top_results,[-1])
            # 0.72
            test_indic = top_results.cpu().numpy()
            predict_id = content[lang]['id'][test_indic]
            test_indices.append(' '.join(predict_id))
        test['predictions'] = test_indices
        test_f2_score = f2_score(test['content_ids'],test['predictions'])
        print("test_f2_score: ",test_f2_score)

        # train_predictions = []
        # for k in tqdm(range(len(train_indices))):
        #     pred = train_indices[k]
        #     p = ' '.join([content.loc[ind, 'id'] for ind in pred])
        #     train_predictions.append(p)

        # train['predictions'] = train_indices
        # train.to_csv("./train.csv")
        # test_predictions = []
        # for k in tqdm(range(len(test_indices))):
        #     pred = test_indices[k]
        #     p = ' '.join([content.loc[ind, 'id'] for ind in pred])
        #     test_predictions.append(p)
        test['predictions'] = test_indices
        # train_test = pd.concat([train,test],axis=0)
        # train_test.to_csv("./train.csv")
        # train.to_csv("./train.csv")
        test.to_csv("./test.csv")

        # train_f2_score = f2_score(train['content_ids'],train['predictions'])
        test_f2_score = f2_score(test['content_ids'],test['predictions'])
        # print("train_f2_score: ",train_f2_score)
        print("test_f2_score: ",test_f2_score)


