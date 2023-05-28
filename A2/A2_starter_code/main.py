from wsd import wsd
from loader import load_data
from bootstrap import Yarowsky

# load data
dev_instances, test_instances, dev_key, test_key = load_data()

most_freq_baseline_count_test = 0
lesk_count_test = 0
bootstrap_test, mod = 0, 0
simplified_lest_count_test = 0

lemma_to_context = {}

# run most frequent baseline and lesks algorithm
for idx, wsd_instance in test_instances.items():
    sense = set(test_key[idx])
    w = wsd(wsd_instance)

    lemma, context = w.get_lemma_and_context()

    most_frequent_baseline_sense_pred = w.most_frequent_sense_baseline()
    if w.computeOverlap(sense, most_frequent_baseline_sense_pred) > 0:
        most_freq_baseline_count_test += 1

    lesk_sense_pred = w.lesk()
    if w.computeOverlap(sense, lesk_sense_pred) > 0:
        lesk_count_test += 1

    simplified_lesk_sense_results, max_overlap = w.simplified_lesk()
    if (max_overlap > 0):
        simplified_lest_count_test += 1

    if lemma not in lemma_to_context:
        lemma_to_context[lemma] = []

    lemma_to_context[lemma].append(context)

lemma_to_bootstrap = {}

for lemma in lemma_to_context:
    lemma_to_bootstrap[lemma] = Yarowsky(lemma, lemma_to_context[lemma])

for idx, wsd_instance in test_instances.items():
    sense = set(test_key[idx])
    w = wsd(wsd_instance)

    lemma, context = w.get_lemma_and_context()

    pred_bootstrap = lemma_to_bootstrap[lemma].predict(context)

    if w.computeOverlap(sense, pred_bootstrap):
        bootstrap_test += 1


print("Test Most frequent baseline count = ",
      most_freq_baseline_count_test / len(test_instances))
print("Test Lesk count = ", lesk_count_test / len(test_instances))
print("Test Simplified Lesk count = ",
      simplified_lest_count_test / len(test_instances))
print("Accuracy for bootstrap is ", (bootstrap_test/len(test_instances)))


most_freq_baseline_count_dev = 0
lesk_count_dev = 0
bootstrap_dev, mod = 0, 0
simplified_lest_count_dev = 0

lemma_to_context = {}

# run most frequent baseline and lesks algorithm
for idx, wsd_instance in dev_instances.items():
    sense = set(dev_key[idx])
    w = wsd(wsd_instance)

    lemma, context = w.get_lemma_and_context()

    most_frequent_baseline_sense_pred = w.most_frequent_sense_baseline()
    if w.computeOverlap(sense, most_frequent_baseline_sense_pred) > 0:
        most_freq_baseline_count_dev += 1

    lesk_sense_pred = w.lesk()
    if w.computeOverlap(sense, lesk_sense_pred) > 0:
        lesk_count_dev += 1

    simplified_lesk_sense_results, max_overlap = w.simplified_lesk()
    if (max_overlap > 0):
        simplified_lest_count_dev += 1

    if lemma not in lemma_to_context:
        lemma_to_context[lemma] = []

    lemma_to_context[lemma].append(context)

lemma_to_bootstrap = {}

for lemma in lemma_to_context:
    lemma_to_bootstrap[lemma] = Yarowsky(lemma, lemma_to_context[lemma])

for idx, wsd_instance in dev_instances.items():
    sense = set(dev_key[idx])
    w = wsd(wsd_instance)

    lemma, context = w.get_lemma_and_context()

    pred_bootstrap = lemma_to_bootstrap[lemma].predict(context)

    if w.computeOverlap(sense, pred_bootstrap):
        bootstrap_dev += 1


print("dev Most frequent baseline count = ",
      most_freq_baseline_count_dev / len(dev_instances))
print("dev Lesk count = ", lesk_count_dev / len(dev_instances))
print("dev Simplified Lesk count = ",
      simplified_lest_count_dev / len(dev_instances))
print("Accuracy for bootstrap is ", (bootstrap_dev/len(dev_instances)))
