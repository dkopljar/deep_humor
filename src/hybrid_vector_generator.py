import os
import sys
import dataset_parser

# tweets = dataset_parser.read_file_by_line_and_tokenize("../dataset/evaluation_data/Bad_Job_In_5_Words.tsv")
# for tw in tweets:
    # print(tw)
# print(dataset_parser.tokenize_and_filter_tweet("Ola, como estas you prick"))
print(dataset_parser.tweet_to_integer_vector("Ahe sfa dsaf dsaf dsfds sdfjdude #fsd  sa f dsafa ds ds  fdsaf ds fdsaf adfd fdfdfd fdfdsfdsfads ds fdsafdsfasdfdsf sadfdsf asdf fadskako @ste!!!"))

def generate(train_dir, output_dir):
    print()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', __file__, '<train_data> <output_dir>')
        print('train_data directory contains train tsv files for each theme.')
        print('output directory for storing vector pickles.')
        print('Example call:')
        print('python3 evaluator.py ../dataset/train_data output')
        sys.exit(1)

    _, train_dir, output_dir = sys.argv
    generate(train_dir, output_dir)

# glove = loadGlove("./resources/glove/glove.twitter.27B.100d.txt")
# createGlovefromTweet(glove, "Gugi is smart boy")