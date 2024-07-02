# Contrastive Captioners

<!-- 
#### Building Blocks
- Image Encoder - ViT or similar architectures which process images in patches.
- Text Decoder
    - Unimodal Text Decoder
        - Does not have the cross attention block whereas the Mulitmodal Text Decoder has it, therefore rather than creating a single object for a transformer decoder, it is better to create a transformer decoder with the self attention and a cross attention block to be added as needed. 
-->

This is an implementation of [Contrastive Captioner](https://arxiv.org/pdf/2205.01917v2). 

The aim is to train self supervised representation learning model from scratch to compare performances of various ssl methods such as masked modelling, contrastive learning, non contrastive learning, language supervision.


### Losses

#### Contrastive Loss
- The type of contrastive loss used in the code is InfoNCE (Information Noise Contrastive Estimation). 


InfoNCE Loss = -log \frac{\exp(sim(q, k^+)/\tau)}{\sum_{i=0}^K \exp(sim(q, k_i)/\tau)}
