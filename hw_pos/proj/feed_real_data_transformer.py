import TransformerLab
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer
import torch

# use tokenizer that based on BERT model -> ignore case #
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(len(tokenizer.vocab))

# e.g. Hello WORLD how ARE yoU -> ['hello', 'world', 'how', 'are', 'you', '?']
# special tokens to mark first of sequence, seperator of seq, padding for seq to get same length, unknown token
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
# convert token into their index in dictionary
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
print(max_input_length)


# preprocess
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    # need to append two tokens to each sequence, one to the start and one to the end.
    tokens = tokens[:max_input_length - 2]
    return tokens


def main():
    # define our fields #
    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)
    LABEL = data.LabelField(dtype=torch.float)

    # load the data and create the validation splits #
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    TransformerLab.transformer(train_data, test_data, LABEL, 'real')


if __name__ == "__main__":
    main()
