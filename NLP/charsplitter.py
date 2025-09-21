#%%
from nltk.tokenize import RegexpTokenizer

text = "esta es una oracion simple."

tokenizer = RegexpTokenizer(r".", gaps=False)
tokens = tokenizer.tokenize(text)

print(tokens)
# %%
