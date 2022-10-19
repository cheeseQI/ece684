import nltk


def build_tables():
    # get training set
    lines = nltk.corpus.brown.tagged_sents(tagset='universal')[0:10000]
    # a transition (tag, tag) ; b obs (word, tag); firstly only line_count frequency
    word_indices = {}
    tag_indices = {}
    for line in lines:
        for word_tuple in line:
            word = word_tuple[0].lower()
            tag = word_tuple[1]
            if word not in word_indices:
                word_indices[word] = len(word_indices)
            if tag not in tag_indices:
                tag_indices[tag] = len(tag_indices)

    a = [[0] * len(tag_indices) for i in range(len(tag_indices))]
    b = [[0] * len(word_indices) for i in range(len(tag_indices))]
    pi = [0] * len(tag_indices)
    line_count = 0
    for line in lines:
        line_count += 1
        first_tag = line[0][1]
        pi[tag_indices[first_tag]] += 1
        for i in range(len(line)):
            word = line[i][0].lower()
            tag = line[i][1]
            if i != 0:
                pre_tag = line[i - 1][1]
                a[tag_indices[pre_tag]][tag_indices[tag]] += 1
            b[tag_indices[tag]][word_indices[word]] += 1
    # get pi
    for i in range(len(pi)):
        pi[i] = pi[i] / line_count
    # get a
    for i in range(len(a)):
        count = 0
        # using add one smoothing
        for j in range(len(a[0])):
            count += a[i][j] + 1
        for j in range(len(a[0])):
            a[i][j] = (a[i][j] + 1) / count
    # get b
    for i in range(len(b)):
        count = 0
        for j in range(len(b[0])):
            count += b[i][j] + 1
        for j in range(len(b[0])):
            b[i][j] = (b[i][j] + 1) / count
    # b_{ik} = probability of observing k at state i, consider no-existing corner case
    special = 1 / len(b[0])
    for i in range(len(b)):
        b[i].append(special)
    return a, b, pi, word_indices, tag_indices
