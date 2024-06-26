from coca import CoCa
from transformers import ViTForImageClassification

image_enc = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model = CoCa(image_enc=image_enc, vocab=['a', 'the', 'an'])

print(model)
