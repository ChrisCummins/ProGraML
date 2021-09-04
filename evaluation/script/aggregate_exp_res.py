import os
import argparse


def parse_log_file(fpath):
    global log_info

    with open(fpath, 'r') as fin:
        data = fin.readlines()

    header_str = ','.join(data[1].strip().split(' | ')[1].split(',')[1:])
    variants = header_str.split(',')
    for variant in variants:
        if variant not in log_info:
            log_info[variant] = []

    del data[:2]
    for row in data:
        row = row.strip()
        score_str = ','.join(row.split(' | ')[1].split(',')[1:])
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

print("Analyzing task name: %s..." % args.task)
print("In directory: %s" % args.dir)

assert len(log_info["STANDARD_IG"]) == \
    len(log_info["ASCENDING_DEPENDENCY_GUIDED_IG"]) == \
    len(log_info["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"]) == \
    len(log_info["DESCENDING_DEPENDENCY_GUIDED_IG"]
        ), "[ERROR] Uneven lengths of lists."

print("====== Mean Attribution Score ======")
for variant, scores in log_info.items():
    mean_score = sum(scores) / len(scores)
    if variant != "STANDARD_IG":
        std_mean_score = sum(log_info["STANDARD_IG"]) / \
            len(log_info["STANDARD_IG"])
        margin = (mean_score - std_mean_score) / std_mean_score
        print("Variant: %s | # Samples: %d | Mean score: %f (%s)" %
              (variant, len(scores), mean_score, "{:.2f}".format(margin * 100) + "%"))
    else:
        print("Variant: %s | # Samples: %d | Mean score: %f" %
              (variant, len(scores), mean_score))

running_ranks = {}
for variant, _ in log_info.items():
    running_ranks[variant] = []

for i in range(len(log_info["STANDARD_IG"])):
    attr_acc_std_ig = log_info["STANDARD_IG"][i]
    attr_acc_dep_guided_ig = log_info["ASCENDING_DEPENDENCY_GUIDED_IG"][i]
    attr_acc_dep_guided_ig_unaccumulated = log_info["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"][i]
    attr_acc_reverse_dep_guided_ig = log_info["DESCENDING_DEPENDENCY_GUIDED_IG"][i]
    sorted_acc_scores = sorted([attr_acc_std_ig, attr_acc_dep_guided_ig,
                               attr_acc_dep_guided_ig_unaccumulated, attr_acc_reverse_dep_guided_ig])
    variant_rank = list(map(lambda x: sorted_acc_scores.index(x), [
        attr_acc_std_ig,
        attr_acc_dep_guided_ig,
        attr_acc_dep_guided_ig_unaccumulated,
        attr_acc_reverse_dep_guided_ig
    ]))

    running_ranks["STANDARD_IG"].append(variant_rank[0])
    running_ranks["ASCENDING_DEPENDENCY_GUIDED_IG"].append(variant_rank[1])
    running_ranks["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"].append(
        variant_rank[2])
    running_ranks["DESCENDING_DEPENDENCY_GUIDED_IG"].append(variant_rank[3])

print("====== Ranking For Variants ======")
for variant, ranks in running_ranks.items():
    mean_rank = sum(ranks) / len(ranks)
    if variant != "STANDARD_IG":
        std_mean_rank = sum(log_info["STANDARD_IG"]) / \
            len(log_info["STANDARD_IG"])
        margin = (mean_rank - std_mean_rank) / std_mean_rank
        print("Variant: %s | # Samples: %d | Mean rank: %f (%s)" %
              (variant, len(ranks), mean_rank, "{:.2f}".format(margin * 100) + "%"))
    else:
        print("Variant: %s | # Samples: %d | Mean rank: %f" %
              (variant, len(ranks), mean_rank))
