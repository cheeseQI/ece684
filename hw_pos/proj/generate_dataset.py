from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def gen(tokens):
    print(tokens)
    indexes = tokenizer.convert_tokens_to_ids(tokens)
    print(indexes)

def main():
    gen(['i', 'love', 'the', 'united kingdom'])

if __name__ == "__main__":
    main()