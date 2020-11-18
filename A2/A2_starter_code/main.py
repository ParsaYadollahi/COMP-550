from wsd import wsd
from loader import load_data

# load data
dev_instances, test_instances, dev_key, test_key = load_data()

most_freq_baseline_count = 0
lesk_count = 0

# run most frequent baseline and lesks algorithm
for idx, wsd_instance in test_instances.items():
    sense = set(test_key[idx])

    w = wsd(wsd_instance)

    most_frequent_baseline_sense_pred = w.most_frequent_sense_baseline()
    if (w.computeOverlap(sense, most_frequent_baseline_sense_pred) > 0):
        most_freq_baseline_count = most_freq_baseline_count + 1

    lesk_sense_pred = w.lesk()
    if (w.computeOverlap(sense, lesk_sense_pred) > 0):
        lesk_count = lesk_count + 1

print(most_freq_baseline_count/len(test_instances))
print(lesk_count/len(test_instances))


#     most_freq_baseline_sense_pred = wsd.most_frequent_baseline()
#     most_freq_baseline_count += 1 if len(
#         sense.intersection(most_freq_baseline_sense_pred)) > 0 else 0

#     lesk_sense_pred = wsd.run_lesks()
#     lesk_count += 1 if len(sense.intersection(lesk_sense_pred)) > 0 else 0

# print(most_freq_baseline_count/len(test_instances))
# print(lesk_count/len(test_instances))
