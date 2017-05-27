import sys
import os

def evaluate_pipe(evaluation_dir, gold_dir):
    print(evaluation_dir, gold_dir)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', __file__, '<evaluation_dir> <gold_dir>')
        print('Input directory contains tsv files for each theme.')
        sys.exit(1)

    _, evaluation_dir, gold_dir = sys.argv
    evaluate_pipe(evaluation_dir, gold_dir)
