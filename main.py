# =========================================================================================
# Libraries
# =========================================================================================
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
from sentence_transformers import util
from tqdm.contrib import tzip

# from cuml.metrics import pairwise_distances
# from cuml.neighbors import NearestNeighbors
# % env
# TOKENIZERS_PARALLELISM = false
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG1:
    uns_model = "./Mutiligual_13"
    sup_model = "./Mutiligual_13"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model)
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model)


CFG_list = [CFG1]

topics_df = pd.read_csv('../data/topics.csv', index_col=0).fillna({"title": "", "description": ""})
content_df = pd.read_csv('../data/content.csv', index_col=0).fillna("")
correlations_df = pd.read_csv('../data/correlations.csv', index_col=0)
class Topic:
    def __init__(self, topic_id):
        self.id = topic_id

    @property
    def parent(self):
        parent_id = topics_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    @property
    def content(self):
        if self.id in correlations_df.index:
            return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
        else:
            return tuple([]) if self.has_content else []

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))

    @property
    def children(self):
        return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]

    def subtree_markdown(self, depth=0):
        markdown = "  " * depth + "- " + self.title + "\n"
        for child in self.children:
            markdown += child.subtree_markdown(depth=depth + 1)
        for content in self.content:
            markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
        return markdown

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topics_df.loc[self.id][name]

    def __str__(self):
        return self.title

    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"


class ContentItem:
    def __init__(self, content_id):
        self.id = content_id

    @property
    def topics(self):
        return [Topic(topic_id) for topic_id in
                topics_df.loc[correlations_df[correlations_df.content_ids.str.contains(self.id)].index].index]

    def __getattr__(self, name):
        return content_df.loc[self.id][name]

    def __str__(self):
        return self.title

    def __repr__(self):
        return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"

    def __eq__(self, other):
        if not isinstance(other, ContentItem):
            return False
        return self.id == other.id

    def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
        breadcrumbs = []
        for topic in self.topics:
            new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
            if new_breadcrumb:
                new_breadcrumb = new_breadcrumb + separator + self.title
            else:
                new_breadcrumb = self.title
            breadcrumbs.append(new_breadcrumb)
        return breadcrumbs


def read_data(cfg):
    # topics = pd.read_csv('../data/topics.csv')
    # content = pd.read_csv('../data/content.csv')
    topics = pd.read_csv('../data/topics.csv')
    content = pd.read_csv('../data/content.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    # Merge topics with sample submission to only infer test topics
    topics = topics.merge(sample_submission, how='inner', left_on='id', right_on='topic_id')
    # Fillna titles
    topics['title'].fillna("", inplace=True)
    content['title'].fillna("", inplace=True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace=True)
    content.sort_values('length', inplace=True)
    # Drop cols
    # topics.drop(
    #     ['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length', 'topic_id',
    #      'content_ids'], axis=1, inplace=True)
    # # topics.drop([ 'topic_id','content_ids','length'], axis=1, inplace=True)
    # content.drop(['length','title',''], axis=1,
    #             inplace=True)
    # # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    topics_id = topics["id"]
    topics_text = []
    content_id = content['id']
    content_text = []
    for i in tqdm(topics_id):
        text = '[' + str(Topic(i).level) + ']' + '[SEP]' + Topic(i).title + '[SEP]' + Topic(
            i).get_breadcrumbs().replace(">>", ",") + '[SEP]' + Topic(i).description[0:100]
        topics_text.append(text)
    topics = pd.DataFrame(
        {'id': topics_id,
         'title': topics_text,
         'language':topics['language']}
    )
    for i in tqdm(content_id):
        text =  ContentItem(i).title  + '[SEP]' + ContentItem(i).kind + '[SEP]' +ContentItem(i).text.split("\n")[0][0:200]+'[SEP]'+ContentItem(i).description[0:100]
        content_text.append(text)
    content = pd.DataFrame(
        {'id': content_id,
         'title': content_text,
         'language': content['language']}
    )
    languages = content['language'].unique().tolist()

    content_dict = {}
    for lang in languages:
        content_dict[lang] = content[content['language'] == lang].reset_index(drop=True)

    return topics, content_dict,languages


def prepare_uns_input(text, cfg):
    inputs = cfg.uns_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=256
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

def prepare_uns_input(text, cfg):
    inputs = cfg.uns_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=256
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_uns_input(self.texts[item], self.cfg)
        return inputs



# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_sup_input(text, cfg):
    inputs = cfg.sup_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=256
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# Supervised dataset
# =========================================================================================
class sup_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], self.cfg)
        return inputs


# =========================================================================================
# Mean pooling class
# =========================================================================================
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


# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.uns_model)
        self.model = AutoModel.from_pretrained(cfg.uns_model, config=self.config)
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature


# =========================================================================================
# Get embeddings
# =========================================================================================
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


# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_socre(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


# =========================================================================================
# Build our inference set
# =========================================================================================
def build_inference_set(tmp_topics, tmp_content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    # Iterate over each topic
    for k in tqdm(range(len(tmp_topics))):
        row = tmp_topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title']
        topics_language = row['language']
        predictions = row['predictions'].split(' ')
        for pred in predictions:
            content_title = tmp_content[topics_language].loc[pred, 'title']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
    # Build training dataset
    tmp_test = pd.DataFrame(
        {'topics_ids': topics_ids,
         'content_ids': content_ids,
         'title1': title1,
         'title2': title2
         }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2
    gc.collect()
    torch.cuda.empty_cache()
    return tmp_test


# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(tmp_topics, tmp_content,languages, cfg):
    # Create topics dataset
    topics = tmp_topics
    languages = languages
    topics_dataset = uns_dataset(topics, cfg)
    # Create content dataset
    content_text_dict = {}
    for lang in languages:
        content_text = uns_dataset(tmp_content[lang], CFG1)
        content_text_dict[lang] = content_text
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.uns_tokenizer, padding='longest'),
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    content_loader_dict = {}
    for lang in languages:
        content_loader = DataLoader(
            content_text_dict[lang],
            batch_size=32,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=cfg.uns_tokenizer, padding='longest'),
            num_workers=0,
            pin_memory=True,
            drop_last=False
             )
        content_loader_dict[lang] = content_loader
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds_dict = {}
    for lang in languages:
        content_preds = get_embeddings(content_loader_dict[lang],model, device)
        content_preds_dict[lang] = content_preds
    # Transfer predictions to gpu
    # topics_preds_gpu = cp.array(topics_preds)
    # content_preds_gpu = cp.array(content_preds)
    # Release memory
    topics_language = tmp_topics['language']

    del topics_dataset, content_text_dict , topics_loader, content_loader_dict, content_preds
    gc.collect()
    torch.cuda.empty_cache()
    # KNN model
    print(' ')
    print('Training KNN model...')
    top_n = 100
    train_indices = []
    for topic_embedding,lang in tzip(topics_preds,topics_language):
        cos_scores = util.cos_sim(topic_embedding, content_preds_dict[lang])[0]
        top_results = torch.topk(cos_scores, k=top_n)
        train_indic = top_results[1].cpu().numpy()
        predict_id = tmp_content[lang]['id'][train_indic]
        train_indices.append(' '.join(predict_id))
    predictions = train_indices
    # for k in range(len(indices)):
    #     pred = indices[k]
    #     p = ' '.join([tmp_content.loc[ind, 'id'] for ind in pred.get()])
    #     predictions.append(p)
    topics['predictions'] = predictions
    # Release memory
    del   predictions, train_indices, model
    gc.collect()
    torch.cuda.empty_cache()
    return topics, tmp_content


# =========================================================================================
# Process test
# =========================================================================================
def preprocess_test(tmp_test):
    tmp_test['title1'].fillna("Title does not exist", inplace=True)
    tmp_test['title2'].fillna("Title does not exist", inplace=True)
    # Create feature column
    tmp_test['text'] = tmp_test['title1'] + '[SEP]' + tmp_test['title2']
    # Drop titles
    tmp_test.drop(['title1', 'title2'], axis=1, inplace=True)
    # Sort so inference is faster
    tmp_test['length'] = tmp_test['text'].apply(lambda x: len(x))
    tmp_test.sort_values('length', inplace=True)
    tmp_test.drop(['length'], axis=1, inplace=True)
    tmp_test.reset_index(drop=True, inplace=True)
    gc.collect()
    torch.cuda.empty_cache()
    return tmp_test


# =========================================================================================
# Model
# =========================================================================================
class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.sup_model, output_hidden_states=True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.sup_model, config=self.config)
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# =========================================================================================
# Inference function loop
# =========================================================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


# =========================================================================================
# Inference
# =========================================================================================
def inference(tmp_test, cfg, _idx):
    # Create dataset and loader
    test_dataset = sup_dataset(tmp_test, cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.sup_tokenizer, padding='longest'),
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    # Get model
    model = custom_model(cfg)
    # Load weights
    # state = torch.load("/kaggle/input/reranker-model/.-L12pretrain_23_fold0_42_50_2_0.07370000000000002.pth",
    #                    map_location=torch.device('cpu'))
    state = torch.load("./.-Mutiligual_13_fold0_32_100_2_0.029500000000000005.pth",
                                          map_location=torch.device('cpu'))

    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    # Release memory
    del test_dataset, test_loader, model, state
    gc.collect()
    torch.cuda.empty_cache()
    # Use threshold
    tmp_test['predictions'] = np.where(prediction > 0.0295, 1, 0)
    tmp_test1 = tmp_test[tmp_test['predictions'] == 1]
    tmp_test1 = tmp_test1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
    tmp_test1['content_ids'] = tmp_test1['content_ids'].apply(lambda x: ' '.join(x))
    tmp_test1.columns = ['topic_id', 'content_ids']
    tmp_test0 = pd.Series(tmp_test['topics_ids'].unique())
    tmp_test0 = tmp_test0[~tmp_test0.isin(tmp_test1['topic_id'])]
    tmp_test0 = pd.DataFrame({'topic_id': tmp_test0.values, 'content_ids': ""})
    tmp_test_r = pd.concat([tmp_test1, tmp_test0], axis=0, ignore_index=True)
    tmp_test_r.to_csv(f'submission_{_idx + 1}.csv', index=False)
    print(tmp_test_r.head())

    del tmp_test, tmp_test1, tmp_test0, tmp_test_r
    gc.collect()
    torch.cuda.empty_cache()
if __name__ == '__main__':
    for _idx, CFG in enumerate(CFG_list):
        # Read data
        tmp_topics, tmp_content,language = read_data(CFG)
        # Run nearest neighbors
        tmp_topics, tmp_content = get_neighbors(tmp_topics, tmp_content,language, CFG)
        gc.collect()
        torch.cuda.empty_cache()
        # Set id as index for content
        for key,value in tmp_content.items():
            value.set_index('id', inplace=True)
        # Build training set
        tmp_test = build_inference_set(tmp_topics, tmp_content, CFG)
        # Process test set
        tmp_test = preprocess_test(tmp_test)
        # Inference
        inference(tmp_test, CFG, _idx)
        del tmp_topics, tmp_content, tmp_test
        gc.collect()
        torch.cuda.empty_cache()
    df_test = pd.concat([pd.read_csv(f'submission_{_idx + 1}.csv') for _idx in range(len(CFG_list))])
    df_test.fillna("", inplace=True)
    df_test['content_ids'] = df_test['content_ids'].apply(lambda c: c.split(' '))
    df_test = df_test.explode('content_ids').groupby(['topic_id'])['content_ids'].unique().reset_index()
    df_test['content_ids'] = df_test['content_ids'].apply(lambda c: ' '.join(c))

    df_test.to_csv('submission.csv', index=False)
    df_test.head()