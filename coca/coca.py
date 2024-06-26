import torch
import sys
import torchvision
from torch import nn, einsum
from torch.nn.functional import cross_entropy as ce, normalize
from utils.tokenizer import Tokenizer
import math

class EmbedsToLatents(nn.Module):
	def __init__(self, embed_dim, latent_dim):
		super(EmbedsToLatents, self).__init__()
		self.linear = nn.Linear(embed_dim, latent_dim)

	def forward(self, embeddings):
		out = self.linear(embeddings)
		out = normalize(out, dim=-1)
		return out

class SelfAttention(nn.Module):
	def __init__(self) -> None:
		super(SelfAttention, self).__init__()
		self.num_heads = 8
		self.head_dim = 64
		self.scale = self.head_dim ** -0.5

		self.query = nn.Linear(self.head_dim, self.head_dim * self.num_heads)
		self.key = nn.Linear(self.head_dim, self.head_dim * self.num_heads)
		self.value = nn.Linear(self.head_dim, self.head_dim * self.num_heads)

	def forward(self, x):
		batch_size, seq_length, _ = x.size()

		# Linear projections
		q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
		k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
		v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

		# Transpose for attention dot product: b x h x l x d
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		# Causal attention mask
		attn_mask = torch.tril(torch.ones(seq_length, seq_length)).type_as(q)
		attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

		# Scaled dot product attention
		attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
		attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
		attn = torch.softmax(attn_scores, dim=-1)

		# Attention output
		attn_output = torch.matmul(attn, v)

		# Concatenate heads
		attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

		return attn_output

class CrossAttention(nn.Module):
	"""
	Cross attention implementation for attentional pooling
	"""
	def __init__(self, 
			dim, 
			num_heads, 
			context_dim) -> None:
		super(CrossAttention, self).__init__()
		
		self.dim = dim
		self.num_heads = num_heads
		self.context_dim = context_dim
		
		self.scale = dim ** -0.5

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

		# computing features for each element in the sequence by mutliplying attention scores witht eh value vectors
		out = torch.matmul(attn, v)

		# current shape is b, h, n, d need to be merged to give output as b, n, h*d
		out = out.transpose(1, 2).reshape(out.shape[0], -1, self.dim)

		return out
	
class TransformerDecoderLayer(nn.Module):
	def __init__(self, dim, cross_attn=False):
		"""
		Transformer decoder layer
		"""
		super(TransformerDecoderLayer, self).__init__()

		self.dim = dim
		self.mha = None
		if cross_attn:
			self.mha = CrossAttention(1, 2, 3)
		
		self.masked_mha = SelfAttention()
		self.ffn = nn.Sequential(
			nn.Linear(dim, dim),
			nn.GELU(),
		)
		self.norm = nn.LayerNorm(dim)

	def pos_enc(self, seq):
		max_len = seq.shape[1]
		d_model = seq.shape[2]

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply the sinusoidal formula
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension
		pe = pe.unsqueeze(0)

		seq = seq + pe

	def forward(self, x):
		# apply positional encodings to the input
		x = self.pos_enc(x)

		x = self.norm(self.masked_mha(x) + x)

		# compute the below if cross attention is included
		if self.mha is not None:
			x = self.norm(self.mha(x) + x)

		x = self.norm(self.ffn(x) + x)
		return x


class CoCa(nn.Module):
	def __init__(self, 
			  image_enc,
			  vocab,
			  image_dim=1024,
			  text_dim=1024,
			  num_patches=256,
			  attn_dim=128,
			  num_heads=8,
			  unimodal_depth=3,
			  multimodal_depth=3):
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
			self.uni_text_dec.append(TransformerDecoderLayer(dim=text_dim))

		for i in range(multimodal_depth):	
			self.mml_text_dec.append(TransformerDecoderLayer(dim=text_dim, cross_attn=True))

		self.image_to_latent = EmbedsToLatents(embed_dim=image_dim, latent_dim=image_dim)
		self.text_to_latent = EmbedsToLatents(embed_dim=text_dim, latent_dim=text_dim)

	def compute_image_embeddings(self, images):
		if self.image_enc is not None:
			image_embeddings = self.image_enc(images)
		else:
			raise ValueError("Image Encoder has not been passed as an argument to the CoCa, pass an encoder of your choice to use the Coca model.")
		
		img_queries = self.img_queries.repeat(images.shape[0], 1, 1)
		img_queries = self.attn_pooler(img_queries, image_embeddings)
		return img_queries[:, :-1], img_queries[:, -1]


	def compute_text_embeddings(self, text):
		# convert text into input embeddings
		text_tokens = self.tokenizer(text)

		# add cls token to the text
		cls_token = self.cls_token.repeat(text_tokens.shape[0], 1)
		text_tokens = torch.cat([cls_token, text_tokens], dim=1)

		# pass tokens to the unimodal encoder
		uni_output = self.uni_text_dec(text_tokens)
		text_tokens, cls_token = uni_output[:, :-1], uni_output[:, -1:]

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
		image_tokens, image_embeddings = self.compute_image_embeddings(images)
		text_tokens, text_embeddings = self.compute_text_embeddings(text)

		text_tokens = self.mml_text_dec(text_tokens, image_tokens)		

		# covnert the image embeddings and text embeddings into latent representations as contrastive loss is computed in the latent space
		image_latents = self.image_to_latent(image_embeddings)
		text_latents = self.text_to_latent(text_embeddings)

		# compute loss
		con_loss = self.contrastive_loss(image_latents, text_latents)
		cap_loss = self.captioning_loss(text_tokens, None)

		loss = con_loss + cap_loss
		return con_loss + cap_loss
