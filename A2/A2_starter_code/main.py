from wsd import wsd
from loader import load_instances, load_key

# load data
data_f = 'multilingual-all-words.en.xml'
key_f = 'wordnet.en.key'
dev_instances, test_instances = load_instances(data_f)
dev_key, test_key = load_key(key_f)

most_freq_baseline_count = 0
lesk_count = 0

# run most frequent baseline and lesks algorithm
for idx, wsd_instance in test_instances.items():
    sense = set(test_key[idx])

    wordSenseDisambiguator = wsd(wsd_instance)

#     most_freq_baseline_sense_pred = wsd.most_frequent_baseline()
#     most_freq_baseline_count += 1 if len(
#         sense.intersection(most_freq_baseline_sense_pred)) > 0 else 0

#     lesk_sense_pred = wsd.run_lesks()
#     lesk_count += 1 if len(sense.intersection(lesk_sense_pred)) > 0 else 0

# print(most_freq_baseline_count/len(test_instances))
# print(lesk_count/len(test_instances))
