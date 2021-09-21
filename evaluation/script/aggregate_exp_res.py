import os
import argparse
import matplotlib
matplotlib.use('Agg')  # to avoid using Xserver
import matplotlib.pyplot as plt


COLOR_LIST = ["blue", "red", "yellow", "orange"]


def parse_log_file(fpath):
    global log_info

    with open(fpath, 'r') as fin:
        data = fin.readlines()

    header_str = '\t'.join(data[1].strip().split(' | ')[1].split('\t')[1:])
    variants = header_str.split('\t')
    for variant in variants:
        if variant not in log_info:
            if "DELETION" in variant or "RETENTION" in variant:
                log_info[variant] = [[] for _ in range(10)]
            else:
                log_info[variant] = []

    del data[:2]
    for row in data:
        row = row.strip()
        result_str = '\t'.join(row.split(' | ')[1].split('\t')[1:])

        scores = list(map(lambda x: float(x), result_str.split('\t')[:8]))
        for i in range(8):
            log_info[variants[i]].append(scores[i])

        prob_delta = list(map(lambda x: eval(x), result_str.split('\t')[8:]))
        for i in range(8, 16):
            for j in range(len(prob_delta[i - 8])):
                log_info[variants[i]][j].append(prob_delta[i - 8][j])


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

print("====== Mean Attribution Score ======")
color_choice_deletion = 0
color_choice_retention = 0
for variant, scores in log_info.items():
    if "DELETION" in variant:
        if color_choice_deletion == 0:
            save_img_path = args.dir + "/viz/DELETION_comparison.png"
        x_list, y_list = [], []
        for i in range(len(scores)):
            print("[%s] Step #%d (mean) --> %f" %
                  (variant, i, sum(scores[i]) / len(scores[i])))
            x_list.append(i)
            y_list.append(sum(scores[i]) / len(scores[i]))
        plt.plot(x_list, y_list, color=COLOR_LIST[color_choice_deletion], label=variant.replace("DELETION_RES_", ''))
        plt.legend()
        color_choice_deletion += 1
        if color_choice_deletion == 4:
            plt.xlabel("number of steps")
            plt.ylabel("predicted class probability")
            plt.title("Deletion Game Results")
            plt.show()
            plt.savefig(save_img_path, format="PNG")
            plt.clf()
    elif "RETENTION" in variant:
        if color_choice_retention == 0:
            save_img_path = args.dir + "/viz/RETENTION_comparison.png"
        x_list, y_list = [], []
        for i in range(len(scores)):
            print("[%s] Step #%d (mean) --> %f" %
                  (variant, i, sum(scores[i]) / len(scores[i])))
            x_list.append(i)
            y_list.append(sum(scores[i]) / len(scores[i]))
        plt.plot(x_list, y_list, color=COLOR_LIST[color_choice_retention], label=variant.replace("RETENTION_RES_", ''))
        plt.legend()
        color_choice_retention += 1
        if color_choice_retention == 4:
            plt.xlabel("number of steps")
            plt.ylabel("predicted class probability")
            plt.title("Retention Game Results")
            plt.show()
            plt.savefig(save_img_path, format="PNG")
            plt.clf()
    else:
        mean_score = sum(scores) / len(scores)
        if variant in {"ASCENDING_DEPENDENCY_GUIDED_IG", "UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG", "DESCENDING_DEPENDENCY_GUIDED_IG"}:
            std_mean_score = sum(log_info["STANDARD_IG"]) / \
                len(log_info["STANDARD_IG"])
            margin = (mean_score - std_mean_score) / std_mean_score
            print("[ATTR_ACC_SCORE] Variant: %s | # Samples: %d | Mean score: %f (%s)" %
                  (variant, len(scores), mean_score, "{:.2f}".format(margin * 100) + "%"))
        elif variant == "STANDARD_IG":
            print("[ATTR_ACC_SCORE] Variant: %s | # Samples: %d | Mean score: %f" %
                  (variant, len(scores), mean_score))

        if variant in {"FAITH_ASCENDING_DEPENDENCY_GUIDED_IG", "FAITH_UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG", "FAITH_DESCENDING_DEPENDENCY_GUIDED_IG"}:
            std_mean_score = sum(log_info["FAITH_STANDARD_IG"]) / \
                len(log_info["FAITH_STANDARD_IG"])
            margin = (mean_score - std_mean_score) / std_mean_score
            print("[FAITH_SCORE] Variant: %s | # Samples: %d | Mean score: %f (%s)" %
                  (variant, len(scores), mean_score, "{:.2f}".format(margin * 100) + "%"))
        elif variant == "FAITH_STANDARD_IG":
            print("[FAITH_SCORE] Variant: %s | # Samples: %d | Mean score: %f" %
                  (variant, len(scores), mean_score))

running_ranks = {}
for variant, _ in log_info.items():
    running_ranks[variant] = []

for i in range(len(log_info["STANDARD_IG"])):
    attr_acc_std_ig = log_info["STANDARD_IG"][i]
    attr_acc_dep_guided_ig = log_info["ASCENDING_DEPENDENCY_GUIDED_IG"][i]
    attr_acc_dep_guided_ig_unaccumulated = log_info["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"][i]
    attr_acc_reverse_dep_guided_ig = log_info["DESCENDING_DEPENDENCY_GUIDED_IG"][i]
    faith_score_std_ig = log_info["FAITH_STANDARD_IG"][i]
    faith_score_dep_guided_ig = log_info["FAITH_ASCENDING_DEPENDENCY_GUIDED_IG"][i]
    faith_score_dep_guided_ig_unaccumulated = log_info[
        "FAITH_UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"][i]
    faith_score_reverse_dep_guided_ig = log_info["FAITH_DESCENDING_DEPENDENCY_GUIDED_IG"][i]

    sorted_acc_scores = sorted([
        attr_acc_std_ig,
        attr_acc_dep_guided_ig,
        attr_acc_dep_guided_ig_unaccumulated,
        attr_acc_reverse_dep_guided_ig
    ])
    variant_rank = list(map(lambda x: sorted_acc_scores.index(x), [
        attr_acc_std_ig,
        attr_acc_dep_guided_ig,
        attr_acc_dep_guided_ig_unaccumulated,
        attr_acc_reverse_dep_guided_ig,
    ]))

    sorted_faith_scores = sorted([faith_score_std_ig,
                                  faith_score_dep_guided_ig,
                                  faith_score_dep_guided_ig_unaccumulated,
                                  faith_score_reverse_dep_guided_ig
                                  ])
    variant_rank_faith = list(map(lambda x: sorted_faith_scores.index(x), [
        faith_score_std_ig,
        faith_score_dep_guided_ig,
        faith_score_dep_guided_ig_unaccumulated,
        faith_score_reverse_dep_guided_ig,
    ]))

    running_ranks["STANDARD_IG"].append(variant_rank[0])
    running_ranks["ASCENDING_DEPENDENCY_GUIDED_IG"].append(variant_rank[1])
    running_ranks["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"].append(
        variant_rank[2])
    running_ranks["DESCENDING_DEPENDENCY_GUIDED_IG"].append(variant_rank[3])
    running_ranks["FAITH_STANDARD_IG"].append(variant_rank_faith[0])
    running_ranks["FAITH_ASCENDING_DEPENDENCY_GUIDED_IG"].append(
        variant_rank_faith[1])
    running_ranks["FAITH_UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"].append(
        variant_rank_faith[2])
    running_ranks["FAITH_DESCENDING_DEPENDENCY_GUIDED_IG"].append(
        variant_rank_faith[3])

print("====== Ranking For Variants ======")
for variant, ranks in running_ranks.items():
    if "DELETION" in variant or "RETENTION" in variant:
        continue
    mean_rank = sum(ranks) / len(ranks)
    if variant in {"ASCENDING_DEPENDENCY_GUIDED_IG", "UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG", "DESCENDING_DEPENDENCY_GUIDED_IG"}:
        std_mean_rank = sum(running_ranks["STANDARD_IG"]) / \
            len(running_ranks["STANDARD_IG"])
        margin = (mean_rank - std_mean_rank) / std_mean_rank
        print("[ATTR_ACC_SCORE] Variant: %s | # Samples: %d | Mean rank: %f (%s)" %
              (variant, len(ranks), mean_rank, "{:.2f}".format(margin * 100) + "%"))
    elif variant == "STANDARD_IG":
        print("[ATTR_ACC_SCORE] Variant: %s | # Samples: %d | Mean rank: %f" %
              (variant, len(ranks), mean_rank))

    if variant in {"FAITH_ASCENDING_DEPENDENCY_GUIDED_IG", "FAITH_UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG", "FAITH_DESCENDING_DEPENDENCY_GUIDED_IG"}:
        std_mean_rank = sum(running_ranks["STANDARD_IG"]) / \
            len(running_ranks["STANDARD_IG"])
        margin = (mean_rank - std_mean_rank) / std_mean_rank
        print("[FAITH_SCORE] Variant: %s | # Samples: %d | Mean rank: %f (%s)" %
              (variant, len(ranks), mean_rank, "{:.2f}".format(margin * 100) + "%"))
    elif variant == "FAITH_STANDARD_IG":
        print("[FAITH_SCORE] Variant: %s | # Samples: %d | Mean rank: %f" %
              (variant, len(ranks), mean_rank))
