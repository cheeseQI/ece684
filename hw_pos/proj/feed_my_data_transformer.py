import torch
from transformers import BertTokenizer
import TransformerLab
from torchtext.legacy import data
from torchtext.legacy import datasets
import format_worker

# use tokenizer that based on BERT model -> ignore case #
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

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


# preprocess
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    # need to append two tokens to each sequence, one to the start and one to the end.
    tokens = tokens[:max_input_length - 2]
    return tokens


def main():
    LABEL = data.LabelField(dtype=torch.float)
    real_TEXT = data.Field(batch_first=True,
                           use_vocab=False,
                           tokenize=tokenize_and_cut,
                           preprocessing=tokenizer.convert_tokens_to_ids,
                           init_token=init_token_idx,
                           eos_token=eos_token_idx,
                           pad_token=pad_token_idx,
                           unk_token=unk_token_idx)
    # load the data and create the validation splits #
    real_train_data, real_test_data = datasets.IMDB.splits(real_TEXT, LABEL)

    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)
    fields = [('text', TEXT), ('label', LABEL)]
    # abandon tmp, no need for
    synthetic_train_data, tmp = data.TabularDataset.splits(
        path='.',
        train=format_worker.csv_synthetic_2_filename,
        test='test2.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )

    TransformerLab.transformer(synthetic_train_data, real_test_data, LABEL, 'synthetic')

if __name__ == "__main__":
    main()
