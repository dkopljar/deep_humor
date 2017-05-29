import os

ROOT = os.path.dirname(__file__)

RSC = os.path.join(ROOT, "resources")

DATA = os.path.join(ROOT, "data")

TEMP_OUTPUT = os.path.join(ROOT, "temp-output")

GLOVE_PATH = os.path.join(ROOT, "./resources/glove/glove.twitter.27B.100d.txt")

# Data and resources URLs
GLOVE_TWITTER = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
TRAIN = "http://alt.qcri.org/semeval2017/task6/data/uploads/train_data.zip"
VALIDATION = "http://alt.qcri.org/semeval2017/task6/data/uploads/gold_data.zip"
VALIDATION_NO_LABELS = "http://alt.qcri.org/semeval2017/task6/data/uploads/evaluation_dir.zip"


TF_WEIGHTS = os.path.join(RSC, "weights")
LOGS = os.path.join(RSC, "logs")

CONFIGS = os.path.join(ROOT, "configs")

TWEET_CHARACTER_COUNT = 70