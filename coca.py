import torch
import torchvision
from torch import nn, einsum
from torch.nn.functional import cross_entropy as ce
from utils import Tokenizer

class SelfAttention:
	def __init__(self) -> None:
		pass

class CrossAttention():
	"""
	Cross attention implementation for attentional pooling
	"""
	def __init__(self, dim, num_heads, context_dim) -> None:
		self.dim = dim
		self.num_heads = num_heads
		self.context_dim = context_dim

		self.queryLinear = nn.Linear(context_dim, dim)
		self.keyLinear = nn.Linear(context_dim, dim)
		self.valueLinear = nn.Linear(context_dim, dim)

	def forward(self, queries, context):
		# get query key value
		q = self.queryLinear(queries)
		k = self.keyLinear(context)
		v = self.valueLinear(context)

		# rearrange on the basis of heads
		q, k, v = map(lambda t: t.reshape(t.shape[0], -1, self.num_heads, t.shape[-1] // self.num_heads).transpose(1, 2), (q, k, v))

		# comptuing attention scores
		attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
		attn = attn.softmax(dim=-1)

		out = torch.matmul(attn, v)

		# current shape is b, h, n, d need to be merged to give output as b, n, h*d
		out = out.transpose(1, 2).reshape(out.shape[0], -1, self.dim)

		return out
	
class TransformerDecoderLayer:
	def __init__(self, dim):
		"""
		Transformer decoder layer
		"""
		self.dim = dim
		self.mha = CrossAttention()
		self.masked_mha = SelfAttention()
		self.ffn = nn.Sequential(
			nn.Linear(dim, dim),
			nn.GeLU(),
		)
		self.norm = nn.LayerNorm(dim)

	def forward(self, x):
		x = self.norm(self.masked_mha(x) + x)
		x = self.norm(self.mha(x) + x)
		x = self.norm(self.ffn(x) + x)
		return x


class TransformerDecoder:
	def __init__(self) -> None:
		pass


class CoCa(nn.Module):
	def __init__(self, 
			  image_enc,
			  vocab,
			  image_dim=1024,
			  text_dim=1024,
			  num_patches=256,
			  attn_dim=128,
			  num_heads=8):
		super(CoCa, self).__init__()

		self.image_enc = None
		self.uni_text_dec = nn.ModuleList([])
		self.mml_text_dec = nn.ModuleList([])

		self.image_dim = image_dim
		self.text_dim = text_dim

		self.tokenizer = Tokenizer(vocab)

		self.cls_token = nn.Parameter(torch.randn(self.text_dim))

		self.temperature = None
		
		# initialising tensors and module for attentional pooling
		self.img_queries = nn.Parameter(torch.randn(num_patches+1, image_dim))
		self.attn_pooler = CrossAttention(dim=attn_dim, num_heads=num_heads, context_dim=image_dim)

		# building the unimodal text decoder
		# built with succesive decoder layers of vanilla transformers
		for i in range(unimodal_depth):
			self.uni_text_dec.append(TransformerDecoder)
				

	def compute_image_embeddings(self, images):
		if self.image_enc is not None:
			image_embeddings = self.image_enc(images)
		else:
			raise ValueError("Image Encoder has not been passed as an argument to the CoCa, pass an encoder of your choice to use the Coca model.")
		
		img_queries = self.img_queries.repeat(images.shape[0], 1, 1)
		img_queries = self.attn_pooler(img_queries, image_embeddings)
		return img_queries


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
