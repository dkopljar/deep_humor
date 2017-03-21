import os

ROOT = os.path.dirname(__file__)

RSC = os.path.join(ROOT, "resources")

DATA = os.path.join(ROOT, "data")

# DATA URLS
GLOVE_TWITTER = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
TRAIN = "http://alt.qcri.org/semeval2017/task6/data/uploads/train_data.zip"
VALIDATION = "http://alt.qcri.org/semeval2017/task6/data/uploads/gold_data.zip"
VALIDATION_NO_LABELS = "http://alt.qcri.org/semeval2017/task6/data/uploads/evaluation_dir.zip"
