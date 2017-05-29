import os
import sys

import constants
import evala
import evala_generator
import evalb
import evalb_generator


def evaluate_pipe(evaluation_dir, gold_dir, task):
    if not os.path.exists(constants.TEMP_OUTPUT):
        os.makedirs(constants.TEMP_OUTPUT)
    if task == "A":
        evala_generator.generate(evaluation_dir, constants.TEMP_OUTPUT)
        evala.evaluate(constants.TEMP_OUTPUT, gold_dir)
    else:
        evalb_generator.generate(evaluation_dir, constants.TEMP_OUTPUT)
        evalb.evaluate(constants.TEMP_OUTPUT, gold_dir)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', __file__, '<evaluation_dir> <gold_dir> <task>')
        print(
            'evaluation_dir directory contains evaluation tsv files for each theme.')
        print('gold_dir directory contains gold tsv files for each theme.')
        print('<task> represent evaluation task: "A" or "B"')
        print('Example call:')
        print(
            'python3 evaluator.py ../dataset/evaluation_data ../dataset/gold_data A')
        sys.exit(1)

    _, evaluation_dir, gold_dir, task = sys.argv
    evaluate_pipe(evaluation_dir, gold_dir, task)
