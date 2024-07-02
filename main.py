from coca import CoCa
from transformers import ViTModel
import torch
from argparse import ArgumentParser

def main(args):
	image_enc = ViTModel.from_pretrained("google/vit-base-patch16-224")
	model = CoCa(image_enc=image_enc, vocab=['a', 'the', 'an'], text_dim=args.emb_dim)
	
	# print(model)
	text = torch.randint(0, args.vocab_size, (args.batch_size, args.emb_dim))
	images = torch.randn(args.batch_size, args.img_channels, args.img_size, args.img_size)
	# print(images.shape)
	out = model(text=text, images=images)
	print(out)


if __name__ == "__main__":
	parser = ArgumentParser()
	
	parser.add_argument("--vocab_size", type=int, default=10000)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--emb_dim", type=int, default=512)
	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--img_channels", type=int, default=3)

	args = parser.parse_args()
	main(args)