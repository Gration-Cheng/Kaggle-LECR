from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
df = pd.read_csv("stage1_train.csv")
topics_train= df['title1'].values
contents_train = df['title2'].values
# labels = df['target'].values
# topics_train, topics_val, contents_train, contents_val = train_test_split(topics,contents, test_size=0.00001)


model = SentenceTransformer("./Mutiligual_79")
# train_examples = [InputExample(texts=[topics_train[i],contents_train[i]], label=int(labels_train[i])) for i in range(int(len(topics_train)))]
train_examples = [InputExample(texts=[topics_train[i],contents_train[i]] ) for i in range(int(len(topics_train)))]
# val_examples = [InputExample(texts=[topics_val[i],contents_val[i]], label=int(labels_val[i])) for i in range(len(topics_val))]
# class NoDuplicateDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
#         super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
#         self.selected_indices = set()
#
#     def __iter__(self):
#         for batch in super().__iter__():
#             batch_indices = set(batch.indices)
#             if len(self.selected_indices.intersection(batch_indices)) == 0:
#                 self.selected_indices.update(batch_indices)
#                 yield batch
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
# train_dataloader = NoDuplicateDataLoader(train_examples, batch_size=64, shuffle=True)

# dataset.
# train_loss = losses.OnlineContrastiveLoss(model=model)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
for i in range(50):
    model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    model.save(f"./Mutiligual_{i+80}/")



