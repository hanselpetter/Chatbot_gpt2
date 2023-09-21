"""In PyTorch"""
from transformers import GPT2Tokenizer, GPT2Model

# Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Input text
text = "Replace me by any text you'd like."

# Tokenize the input text
encoded_input = tokenizer(text, return_tensors='pt')

# Pass the input through the model
output = model(**encoded_input)






"""In Tensorflow"""
# from transformers import GPT2Tokenizer, TFGPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = TFGPT2Model.from_pretrained('gpt2')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)