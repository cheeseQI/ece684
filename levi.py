import numpy as np


# This function use dynamic program to find the levenshtein distance between two words.
def levenshtein_dis(false_word: str, word: str):
    n = len(false_word)
    m = len(word)

    # base cases
    if n == 0:
        return m
    if m == 0:
        return n

    # dynamic program
    dp = np.zeros([n + 1, m + 1])
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            curr_state = dp[i - 1][j - 1]
            if false_word[i - 1] != word[j - 1]:
                curr_state += 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, curr_state)
    return dp[n][m]


# The function use a dictionary to find the correct word for the input.
def spelling_corrector(in_word: str):
    fname = './count_1w.txt'
    min_val = 100  # the initial value of min distance
    min_idx = 0  # the index of the correct word
    with open(fname, 'r+', encoding='utf-8') as f:
        idx = 0
        lines = f.readlines()
        # check each word in dictionary to find the min distance
        for line in lines:
            curr = levenshtein_dis(in_word, line.split()[0])
            if curr == 0:
                return line.split()[0]
            if curr < min_val:
                min_val = curr
                min_idx = idx
            idx += 1
    # corner case: the in put word is too long
    if min_val == 100:
        print("cannot find a correct word")
        exit(1)
    res = lines[min_idx].split()[0]
    print('the lv value is ' + str(min_val))
    return res


if __name__ == '__main__':
    word = input("Enter a word: ")
    print(spelling_corrector(word))
