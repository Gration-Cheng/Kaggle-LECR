

#  Kaggle-LECR
 
Rank:68/1058,

Bronze medal solution.
# Introduction
The goal of this competition is to streamline the process of matching educational content to specific topics in a curriculum. You will develop an accurate and efficient model trained on a library of K-12 educational materials that have been organized into a variety of topic taxonomies. These materials are in diverse languages, and cover a wide range of topics, particularly in STEM (Science, Technology, Engineering, and Mathematics).


## Basic Usage

### Requirements

The code was tested with `python 3.9.7`,`PyToch 1.13.1`,  `Sentence-transformers 2.2.2`,`transformers 4.20.1` 

```
### Run the code
```shell
# Getting text input of topics and contents.  
python Tips.py
##Example:
    Topic: [level]+[SEP]+title+[SEP]+breadcrumbs+[SEP]+Description(max_len:100 for string level)
    Content: title+[SEP]+kind+[SEP]+text(The first sentence and max_len=200 for string level) + [SEP] + description(max_len=100 for string level)

# Generate stage1 pair sentence for training
python stage1_pair4train.py

# Finetune stage1 retriver model using MultiNegative Loss
python stage1_SB_finetune.py

# Using KNN to validate stage1 model. Focus on the Recall@50
python KNN.py

# Generate stage2 pair sentence for training, Using the KNN results as hard negative sample and all positive sample in correlation.csv
python stage2_pair4train.py

#Finetune Stage2 reranker model
python stage2.py
```

