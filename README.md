## Contrastive Captioners

<!-- 
#### Building Blocks
- Image Encoder - ViT or similar architectures which process images in patches.
- Text Decoder
    - Unimodal Text Decoder
        - Does not have the cross attention block whereas the Mulitmodal Text Decoder has it, therefore rather than creating a single object for a transformer decoder, it is better to create a transformer decoder with the self attention and a cross attention block to be added as needed. 
-->

This is an implementation of `[Contrastive Captioner](!https://arxiv.org/pdf/2205.01917v2)`. 


