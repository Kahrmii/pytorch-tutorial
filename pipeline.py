from transformers import AutoModel, AutoConfig, BertConfig

Bert = AutoModel.from_pretrained("bert-base-cased")
print(type(Bert))

BertConfig = AutoConfig.from_pretrained("bert-base-cased")
print(type(BertConfig))

