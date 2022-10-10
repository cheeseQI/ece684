import random
import re


# backoff for recording state, whether in backoff or not
def finish_sentence(sentence, n, corpus, deterministic=False):
    # there is no 1 gram case, so no answer
    if n == 1:
        return ["none"]
    return recursion_helper(sentence, n, corpus, deterministic, False, n)


def recursion_helper(sentence, n, corpus, deterministic, backoff, original_n):
    # make dics for n = 2 to given n 3, ... 10, according to input;
    # note that n gram means use former n - 1 worlds !
    # [she, was, not] - > "she was not"
    # pattern = r"(?<=she was not )[\w\.\?\!]+"
    if n == 1:
        return ["none"]
    target = ' '.join(sentence[len(sentence) - n + 1: len(sentence)])
    pattern = r"(?<= " + target + " )[\w\.\,\?\!]+"
    next_matches = re.findall(pattern, ' '.join(corpus))
    frq_dic = {}
    for word in next_matches:
        if word in frq_dic:
            frq_dic[word] += 1
        else:
            frq_dic[word] = 1
    # do back-off if no word
    if len(next_matches) == 0:
        backoff = True
        return recursion_helper(sentence, n - 1, corpus, deterministic, backoff, original_n)

    if not deterministic:
        word = random.choice(list(frq_dic))
    else:
        word = max(frq_dic, key=frq_dic.get)
    if word == '.' or word == '?' or word == '!' or len(sentence) == 9:
        sentence.append(word)
        return sentence

    sentence.append(word)
    if backoff:
        recursion_helper(sentence, original_n, corpus, deterministic, False, original_n)
    else:
        recursion_helper(sentence, n, corpus, deterministic, False, original_n)
    return sentence
