# Select and Summarize: Scene Saliency for Movie Script Summarization
Paper link: [NAACL'24](https://aclanthology.org/2024.findings-naacl.218.pdf)

## MENSA Dataset - Movie Scene Saliency

The dataset can be downloaded from the Hugging Face hub:

[rohitsaxena/MENSA](https://huggingface.co/datasets/rohitsaxena/MENSA)


## SELECT & SUMM: Select and Summarize model
The summarization model used in the paper operates in two stages:

### Stage 1: Scene Salinecy Classification
In the first stage, we train a scene saliency classification model.

1. Pre-compute scene encoding using RoBerta-large

```
python precompute_embeddings.py

```
2. Train the classification model. Finally, the best checkpoint is used to select salient scenes to generate a new train/val/test split for summarization:

```
python scene_saliency_classification.py

```
### Stage 2: Summarization

In the second stage, we use only the salient scenes predicted in stage 1 to fine-tune the summarization model (LED). 

```
python summarize.py --fp16 --grad_ckpt

```

## Citation

```
@inproceedings{saxena-keller-2024-select,
    title = "Select and Summarize: Scene Saliency for Movie Script Summarization",
    author = "Saxena, Rohit  and
      Keller, Frank",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.218",
    pages = "3439--3455",}
```
