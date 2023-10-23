import random

KEYS = {"10": 0.1, "11": 0.4, "0": 0.5}

TARGET_SENTENCE_LENGTH = 50
TARGET_SENTENCE_COUNT = 5000

def build_sentences(subword_probs):
    sentences = set()
    while len(sentences) < TARGET_SENTENCE_COUNT:
        new_sentence = []
        sentence_length = 0
        while sentence_length < TARGET_SENTENCE_LENGTH:
            toAdd = random.choices(list(subword_probs.keys()), weights=subword_probs.values(), k=1)[0]
            sentence_length += len(toAdd)
            new_sentence.append(toAdd)
        sentences.add("".join(new_sentence))
    sentences = list(sentences)
    return sentences

def get_numerator_theta(subwords, sentences, theta):
    ret = 0
    for sentence in sentences: 
        #print((theta, sentence))
        scores = [dict() for i in range(len(sentence) + 1)]
        for subword in subwords.keys():
            if len(subword) <= len(sentence) and subword == sentence[: len(subword)]:
                d = scores[len(subword)]
                if subword == theta:
                    d.update({1: subwords[subword]})
                else:
                    if 0 not in d:
                        d.update({0: subwords[subword]})
                    else: 
                        d[subword] += subwords[subword]
        
        for index in range(1, len(scores)):
            slot_dict = scores[index]
            if len(slot_dict) == 0: continue
            for count, probability_mass in slot_dict.items():
                for subword in subwords.keys():
                    if index + len(subword) <= len(sentence) and subword == sentence[index: index + len(subword)]:
                        d = scores[index + len(subword)]
                        if subword == theta:
                            if count + 1 in d:
                                d[count + 1] += (probability_mass * subwords[subword])
                            else:
                                d.update({count + 1: probability_mass * subwords[subword]})
                        else:
                            if count not in d:
                                d.update({count: probability_mass * subwords[subword]})
                            else: 
                                d[count] += (probability_mass * subwords[subword])
        for count, prob in scores[-1].items():
            ret += (count * prob) 
    return ret

def get_theta_hat(subwords, sentences):
    numerators = {subword : get_numerator_theta(subwords, sentences, subword) for subword in subwords.keys()}
    denominator = sum(numerators.values())
    for subword, value in numerators.items():
        numerators[subword] = value / denominator
    return numerators

def estimate_theta_hat(subwords, sentences, max_iterations = 1000, target_total_change = 0.01):
    total_change = float('inf')
    old_theta_estimates = subwords
    while max_iterations > 0 and total_change > target_total_change:
        total_change = 0
        max_iterations -= 1        
        new_theta_estimates = get_theta_hat(old_theta_estimates, sentences)
        for subword, value in new_theta_estimates.items():
            total_change += abs(value - old_theta_estimates[subword])
        old_theta_estimates = new_theta_estimates
        print(old_theta_estimates)
        print()
    return old_theta_estimates

sentences = build_sentences(KEYS)
print(sentences)
uniform_keys = {"10": 1/3, "11": 1/3, "0": 1/3}
estimate_theta_hat(uniform_keys, sentences)



