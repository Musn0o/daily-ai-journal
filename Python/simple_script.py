import json
import string

# 1. Read JSON file
with open("data.json", "r") as f:
    data = json.load(f)

# 2. Extract texts
texts = data["texts"]

print(texts)

# 3. Preprocess and tokenize using list comprehensions
tokenized_text = [
    text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    for text in texts
]

print(tokenized_text)
