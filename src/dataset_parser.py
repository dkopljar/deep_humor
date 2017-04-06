import os
import re

TRAIN_PATH = "/Users/filipgulan/apt_project/dataset/train_data"

def read_file_by_line_and_tokenize(file_path):
    tweets = [line.rstrip('\n') for line in open(file_path, 'r', encoding="utf8")]
    tokenized = [re.split('\t', tweet) for tweet in tweets]
    return tokenized
 
def filterText(text):  # remove unnecesary data from tweet, such as extra hashtags, links
    tweets = []
    for tweet in text:
        filtered = []
        for token in tweet:
            if(not(token.startswith('#') or token.startswith('@') or token.startswith('.@') or token.startswith('http'))):
                filtered.append(token)
        while filtered.__contains__(''):
            filtered.remove('')
        tweets.append(filtered)
    return tweets

def combinantorial(lst):
    count = 0
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            if element1[2] == element2[2]:
                continue
            pairs.append((element1, element2))
        index += 1
    return pairs

all_combs = []
for filename in os.listdir(TRAIN_PATH):
    if filename.endswith(".tsv"):
        current_rows = read_file_by_line_and_tokenize(os.path.join(TRAIN_PATH, filename))
        comb = combinantorial(current_rows)
        all_combs.extend(comb)

print(len(all_combs))