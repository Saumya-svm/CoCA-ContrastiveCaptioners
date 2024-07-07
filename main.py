from coca import CoCa
from transformers import ViTModel
import torch
from argparse import ArgumentParser
from datasets import flickr, get_data
from torch.utils.data import DataLoader

def train(model, num_epochs, optimizer, train_loader):
	model.train()
	for epoch in range(num_epochs):
		for idx, (images, captions) in enumerate(train_loader):
			optimizer.zero_grad()

			loss = model(text=captions, images=images)
			loss.backward()

			optimizer.step()
			print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

	print("Training complete.")
	return model

def main(args):
	image_enc = ViTModel.from_pretrained("google/vit-base-patch16-224")
	model = CoCa(image_enc=image_enc, vocab=['a', 'the', 'an'], text_dim=args.emb_dim)
	
	# print(model)
	text = torch.randint(0, args.vocab_size, (args.batch_size, args.emb_dim))
	images = torch.randn(args.batch_size, args.img_channels, args.img_size, args.img_size)
	# print(images.shape)
	out = model(text=text, images=images)

	# load data
	train_data = flickr(type='train')
	train_loader = DataLoader(train_data, transforms=None, batch=64)

	test_data = flickr(type='test')
	test_loader = DataLoader(test_data, transforms=None, batch=64)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	num_epochs = 20
	model = train(model, num_epochs, optimizer, train_loader)

	if args.eval:
		eval(model, test_loader)
	

if __name__ == "__main__":
	parser = ArgumentParser()
	
	parser.add_argument("--vocab_size", type=int, default=10000)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--emb_dim", type=int, default=512)
	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--img_channels", type=int, default=3)
	parser.add_argument("--eval", type=int, default=0)

	args = parser.parse_args()
	main(args)