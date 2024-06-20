import torch
import torchvision
from torch import nn, einsum
from torch.nn.functional import cross_entropy as ce
from utils import Tokenizer

class CrossAttention():
	def __init__(self, dim, heads, context_dim) -> None:
		pass

class CoCa(nn.Module):
	def __init__(self, 
			  image_enc,
			  vocab,
			  image_dim=1024,
			  text_dim=1024,
			  num_patches=256,
			  attn_dim=128,
			  heads=8):
		super(CoCa, self).__init__()

		self.image_enc = None
		self.uni_text_dec = nn.ModuleList([])
		self.mml_text_dec = nn.ModuleList([])

		self.image_dim = image_dim
		self.text_dim = text_dim

		self.tokenizer = Tokenizer(vocab)

		self.cls_token = nn.Parameter(torch.randn(self.text_dim))

		self.temperature = None

		self.img_queries = nn.Parameter(torch.randn(num_patches+1, image_dim))
		self.attn_pooler = CrossAttention(dim=attn_dim, heads=heads)

		# building the unimodal text decoder
		# built with succesive decoder layers of vanilla transformers
				

	def compute_image_embeddings(self, images):
		if self.image_enc is not None:
			image_embeddings = self.image_enc(images)
		else:
			raise ValueError("Image Encoder has not been passed as an argument to the CoCa, pass an encoder of your choice to use the Coca model.")


	def compute_text_embeddings(self, text):
		# convert text into input embeddings
		text_tokens = self.tokenizer(text)

		# add cls token to the text
		cls_token = self.cls_token.repeat(text_tokens.shape[0], 1)
		text_tokens = torch.cat([cls_token, text_tokens], dim=1)

		# pass tokens to the unimodal encoder
		uni_output = self.uni_text_dec(text_tokens)
		text_tokens, cls_token = uni_output[:, :-1], uni_output[:, -1]

		return text_tokens, cls_token


	def contrastive_loss(self, img_emb, txt_emb):
		batch = img_emb.shape[0]
		device = img_emb.device

		sim = einsum('i d, j d -> i j', img_emb, txt_emb) # mentioned the dimensions config to compute the dot product between each pair of vectors from two sets of vectors
		sim = sim * self.temperature.exp()
		contrastive_labels = torch.arange(batch, device=device)
		contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
		contrastive_loss = contrastive_loss * self.contrastive_loss_weight


	def captioning_loss(self, mml_output, target):
		mml_output = mml_output.view(-1, mml_output.size(-1))
		target = target.view(-1)
		loss = ce(mml_output, target)
		return loss

	def forward(self, images, text):
		# compute embeddings for contrastive learning
		image_embeddings = self.compute_image_embeddings(images)
		text_tokens, text_embeddings = self.compute_text_embeddings(text)

		# attentional pooling applied for contrastive learning and captioning
		temp = self.attentional_pooling(image_embeddings)

		mml_output = self.mml_text_dec(text_tokens)		

		# compute loss
		con_loss = self.contrastive_loss(image_embeddings, text_embeddings)
		cap_loss = self.captioning_loss(mml_output, None)

		loss = con_loss + cap_loss
		return con_loss + cap_loss
