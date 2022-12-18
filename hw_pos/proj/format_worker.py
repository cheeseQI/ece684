# Python program to demonstrate
# writing to CSV
import csv
import os
path_of_the_directory = '.data/imdb/aclImdb'
fields = ['text', 'label']
# name of csv file
csv_train_filename = 'mydata/review_train.csv'
csv_train_neg_filename = 'mydata/review_train_neg.csv'
csv_train_pos_filename = 'mydata/review_train_pos.csv'
csv_test_filename = 'mydata/review_test.csv'
csv_pos_word_filename = 'mydata/pos_vocab.csv'
csv_neg_word_filename = 'mydata/neg_vocab.csv'
csv_synthetic_filename = 'mydata/review_synthetic.csv'
csv_synthetic_2_filename = 'mydata/review_synthetic_from_designed.csv'

train_rows = [[]]
test_rows = [[]]
neg_rows = [[]]
pos_rows = [[]]


def parse_row(dir_path, label, rows):
    for filename in os.listdir(dir_path):
        f = os.path.join(dir_path, filename)
        with open(f, 'r', encoding='utf-8') as file:
            data = file.read()
            rows.append([data, label])


def rewrite_file(filename, rows):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows[1:])


def main():
    # write into neg csv and pos csv
    parse_row(path_of_the_directory + '/train' + '/neg', 'neg', neg_rows)
    parse_row(path_of_the_directory + '/train' + '/pos', 'pos', pos_rows)
    rewrite_file(csv_train_neg_filename, neg_rows)
    rewrite_file(csv_train_pos_filename, pos_rows)
    # write into total csv
    parse_row(path_of_the_directory + '/train' + '/neg', 'neg', train_rows)
    parse_row(path_of_the_directory + '/train' + '/pos', 'pos', train_rows)
    parse_row(path_of_the_directory + '/test' + '/neg', 'neg', test_rows)
    parse_row(path_of_the_directory + '/test' + '/pos', 'pos', test_rows)
    rewrite_file(csv_train_filename, train_rows)
    rewrite_file(csv_test_filename, test_rows)


if __name__ == "__main__":
    main()