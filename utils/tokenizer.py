## initial version consists of a basic tokenizer

class Tokenizer():
	def __init__(self, vocab):
		self.vocab = vocab
		self.token_to_id = {token:i for i, token in enumerate(vocab)}
		
	def tokenize(self, text):
		text = text.split()
		tokens = [self.token_to_id[token] if token in self.token_to_id else self.token_to_id['<UNK>'] for token in text]

		return tokens