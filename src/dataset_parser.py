import os
import re
import numpy as np
import pickle
import pdb

data_path = "/Users/Dkopljar/Desktop/Faks/Diplomski/APT/apt_project/dataset/train_data"
glove_file = "/Users/Dkopljar/Desktop/Faks/Diplomski/APT/apt_project/src/resources/glove/glove.twitter.27B.100d.txt"


def filterText(tweets):  # remove unnecesary data from tweet, such as extra hashtags, links
    result = []
    for tweet in tweets:
        filtered = []
        for token in tweet:
            if(not(token.startswith('#') or token.startswith('@') or token.startswith('.@') or token.startswith('http'))):
                filtered.append(token)
        while filtered.__contains__(''):
            filtered.remove('')
        result.append(filtered)
    return result

def read_file_by_line_and_tokenize(file_path):
    tweets = [line.rstrip('\n') for line in open(file_path, 'r', encoding="utf8")]
    tokenized = [re.split('\t', tweet) for tweet in tweets]
    return tokenized

def combinantorial(lst):
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            if element1[1] == element2[1]:
                continue
            if element1[1] < element2[1]:
                if np.random.random() > 0.5:
                    concatMatrix = np.concatenate((element1[0],element2[0]),axis=1)
                    pairs.append((concatMatrix,0))
                else:
                    concatMatrix = np.concatenate((element2[0],element1[0]),axis=1)
                    pairs.append((concatMatrix,1))
            else:
                if np.random.random() > 0.5:
                    concatMatrix = np.concatenate((element2[0],element1[0]),axis=1)
                    pairs.append((concatMatrix,0))
                else:
                    concatMatrix = np.concatenate((element1[0],element2[0]),axis=1)
                    pairs.append((concatMatrix,1))
        index += 1
    return pairs

def parse_data(glove_file,data_path,pickleDir):
    embed_dict={}

    print("Loading glove fle")
    with open(glove_file) as f:
            for line in f:
                split = line.split()
                token = split[0]
                vec = np.array([np.float(x) for x in split[1:]])
                embed_dict[token] = vec


    all_combs = []
    numberOfRows = 25
    rowSize = 100
    topicsMatrix=[]

    paddingRow = np.zeros(rowSize)
    for i,filename in enumerate(os.listdir(data_path)):
        print("Parsing file number:",i)

        if filename.endswith(".tsv"):
            tweetsMatrix = []
            current_rows = read_file_by_line_and_tokenize(os.path.join(data_path, filename))
            ranks = [word[2] for word in current_rows]
            tokenized = [re.split('\s| ', word[1]) for word in current_rows]
            tokenized = filterText(tokenized)
            for k,tweet in enumerate(tokenized):
                sentenceRow = np.zeros((rowSize,numberOfRows))
                for j,token in enumerate(tweet[:numberOfRows]):
                    if(token in embed_dict):

                        sentenceRow[:,j]=embed_dict[token.lower()]
                tweetsMatrix.append((sentenceRow, ranks[k]))
            topicsMatrix.append(tweetsMatrix)


    for topic in topicsMatrix:
        comb = combinantorial(topic)
        all_combs.extend(comb)

    with open(pickleDir,"wb") as f:
        pickle.dump(all_combs, f)
    


parse_data(glove_file, data_path)