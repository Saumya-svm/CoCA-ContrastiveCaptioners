import torch
from torch import nn, einsum
from torch.nn.functional import cross_entropy as ce

class CoCa(nn.Module):
	def __init__(self):
		self.image_enc = nn.ModuleList([])
		self.uni_text_dec = nn.ModuleList([])
		self.mml_text_dec = nn.ModuleList([])

		self.temperature = None

	def compute_image_embeddings(self, images_tokens):
		pass

	def compute_text_embeddings(self, text_tokens):
		pass

	def compute_image_tokens(self, images):
		pass

	def compute_text_tokens(self, text):
		pass

	def contrastive_loss(self, img_emb, txt_emb):
		# sim = torch.einsum('id,jd->i,j', img_emb, txt_emb) # mentioned the dimensions config to compute the dot product between each pair of vectors from two sets of vectors
		# sim = sim/self.temperature


		batch = img_emb.shape[0]
		device = img_emb.device

		sim = einsum('i d, j d -> i j', img_emb, txt_emb) # mentioned the dimensions config to compute the dot product between each pair of vectors from two sets of vectors
		sim = sim * self.temperature.exp()
		contrastive_labels = torch.arange(batch, device=device)
		contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
		contrastive_loss = contrastive_loss * self.contrastive_loss_weight


	def captioning_loss(self, mml_output, text_tokens):
		pass

	def forward(self, images, text):
		# compute tokens
		images_tokens = self.compute_image_tokens(images)
		text_tokens = self.compute_text_tokens(text)

		# compute embeddings for contrastive learning
		image_embeddings = self.compute_image_embeddings(images_tokens)
		text_embeddings = self.compute_text_embeddings(text_tokens)

		# attentional pooling applied for contrastive learning and captioning
		temp = self.attentional_pooling(image_embeddings)

		uni_output = self.uni_text_dec(text_embeddings)
		mml_output = self.mml_text_dec(uni_output)		

		# compute loss
		con_loss = self.contrastive_loss(image_embeddings, text_embeddings)
		cap_loss = self.captioning_loss(mml_output, text_tokens)

		loss = con_loss + cap_loss
		return con_loss + cap_loss