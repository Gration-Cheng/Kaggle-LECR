import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from IPython.display import display, Markdown
from pathlib import Path

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

topic = Topic("t_c78b75536f2c")
print("Content title:\t'" + topic.content[0].title + "' [kind: " + topic.content[0].kind + "]")
print("Topic title:\t'" + topic.title + "'")
print("Breadcrumbs:\t" + topic.get_breadcrumbs())
print(topic.content[0].get_all_breadcrumbs())

# matching = 0
# nonmatching = 0
# for topic_id in topics_df.query("has_content").sample(n=10000).index:
#     topic = Topic(topic_id)
#     if any(topic.language != content.language for content in topic.content):
#         nonmatching += 1
#     else:
#         matching += 1

# print("Matching:", matching)
# print("Nonmatching:", nonmatching)
# print("Percent matching: {:.2f}%".format(100 * matching / (matching + nonmatching)))
# #
# topic_id = topics_df.index
# content_id = content_df.index
# # #
# topic_text = []
# for i in tqdm(topic_id):
#     text = '['+str(Topic(i).level)+']'+'[SEP]'+Topic(i).title + '[SEP]' + Topic(i).get_breadcrumbs().replace(">>",",") +  '[SEP]' + Topic(
#         i).description[0:100]
#     topic_text.append(text)
# #
# content_text = []
# for i in tqdm(content_id):
#     text =  ContentItem(i).title  + '[SEP]' + ContentItem(i).kind + '[SEP]' +ContentItem(i).text.split("\n")[0][0:200]+'[SEP]'+ContentItem(i).description[0:100]
#     content_text.append(text)
#
# topic = pd.DataFrame(
#         {'id': topic_id,
#          'title': topic_text,}
#     )
# topic.to_csv('topic_trans.csv', index=False)
# #
# content = pd.DataFrame(
#         {'id': content_id,
#          'title': content_text,}
#     )
# content.to_csv('content_trans.csv', index=False)

