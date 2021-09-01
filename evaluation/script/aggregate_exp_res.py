import os
import argparse


def parse_log_file(fpath):
    global log_info

    with open(fpath, 'r') as fin:
        data = fin.readlines()

    header_str = ','.join(data[1].split(' | ')[2].split(',')[:4])
    variants = header_str.split(',')
    for variant in variants:
        if variant not in log_info:
            log_info[variant] = []

    del data[:2]
    for row in data:
        row = row.strip()
        score_str = ','.join(row.split(' | ')[2].split(',')[:4])
        scores = list(map(lambda x: float(x), score_str.split(',')))
        log_info["STANDARD_IG"].append(scores[0])
        log_info["ASCENDING_DEPENDENCY_GUIDED_IG"].append(scores[1])
        log_info["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"].append(
            scores[2])
        log_info["DESCENDING_DEPENDENCY_GUIDED_IG"].append(scores[3])


parser = argparse.ArgumentParser(description='Aggregate attr accu logs.')
parser.add_argument(
    "--task", type=str, help="specify the task that log aggregation should be applied to.")
parser.add_argument("--dir", type=str, help="specify log directory.")
args = parser.parse_args()

log_info = {}

log_filenames = os.listdir(args.dir)
for fname in log_filenames:
    if args.task in fname:
        parse_log_file(args.dir + '/' + fname)

for variant, scores in log_info.items():
    mean_score = sum(scores) / len(scores)
    print("Variant: %s | # Samples: %d | Mean score: %f" %
          (variant, len(scores), mean_score))
