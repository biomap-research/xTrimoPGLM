# xTrimoPGLM 

##  Introduction to xTrimoPGLM Family Models

xTrimoPGLM is the open-source version of the latest protein language models towards protein understanding tasks (Masked Protein Language Models) and protein design (Casual Protein Language Models). The xTrimoPGLM family models are developed by BioMap and Tsinghua University. Along with this, we have released the int4 quantization xTrimoPGLM-100B weights and other xTrimo-series small models, which include: 1B, 3B, and 10B models trained with masked language modeling for protein understanding, and 1B, 3B, and 7B causal language models aimed at protein design.

### Out-of-Distribution Perplexity Evaluation

We evaluated the xTrimoPGLM (xTMLM or xTCLM) and xTrimoPGLM(100B) models on two OOD test sets, one with sequence identity lower than 0.9 with the training set (<0.9 ID) and the other with sequence identity lower than 0.5 with the training set (<0.5 ID). Each OOD dataset comprises approximately 10,000 protein sequences. The MLM perplexity results, compared against ESM2-3B and ESM2-15B and the CLM perplexity againest ProGen2-xlarge (6.4B), are as follows (lower is better):

| Model               | ESM2(3B)| ESM2 (15B) | xTMLM (1B) | xTMLM (3B) | xTMLM (10B) | xT (100B) | xT (100B)-INT4 |
|:--------------------|:----------:|:----------:|:----------:|:----------:|:--------------------:|:--------------------:|:--------------------:|
| < 0.9 ID           |   7.7   |   7.3    | 9.3    |    7.8    |      7.6    |            **6.7**         |  **6.8**         | 
| < 0.5 ID            |  11.5 |  11.0  |  13.5 |   11.9  |   11.6   |         **10.7**          | **10.8**          |


| Model               | ProGen2-xlarge (6.4B) | xTCLM (1B) | xTCLM (3B) | xTCLM (7B) | xT (100B) | xT (100B)-INT4 |
|:--------------------|:----------:|:----------:|:----------:|:--------------------:|:--------------------:|:--------------------:|
| < 0.9 ID            |   9.7     | 9.8    |    9.3    |      8.9    |            **8.7**         |      **8.9**         | 
| < 0.5 ID            |   14.3    |  14.0  |    13.7  |    13.5   |         **13.3**          |   **13.5**          |

## Downstream Protein Understanding Tasks Evaluation
(TODO)

## Get Started
### Model List
You can choose to manually download the necessary weights.

| Model            |Download                                                                                                                                |                                                                                                                                                                                
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| xTrimoPGLM-1B-MLM        | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-1b-mlm) |
| xTrimoPGLM-3B-MLM     | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-3b-mlm) |
| xTrimoPGLM-10B-MLM   | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-10b-mlm) |     
| xTrimoPGLM-1B-CLM        | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-1b-clm) |
| xTrimoPGLM-3B-CLM     | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-3b-clm) |
| xTrimoPGLM-7B-CLM   | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-7b-clm) |   
| xTrimoPGLM-100B-Int4  (MLM or CLM) | [🤗 Huggingface](https://huggingface.co/biomap-research/xtrimopglm-100b-int4) |                                                                                                                                                                                              |                                                                                                                                                                                  |

## How to use
### xTrimoPGLM-MLM: Masked Langeuage Models for Protein Understanding tasks
(Note that the xTrimoPGLM-100B INT4 quantization can be infered in a single A100/800 GPU with 80G memory.)
```python


# Obtain residue embeddings
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, AutoConfig
import torch

tokenizer  = AutoTokenizer.from_pretrained("biomap-research/xtrimopglm-100b-int4", trust_remote_code=True, use_fast=True)
config = AutoConfig.from_pretrained("biomap-research/xtrimopglm-100b-int4",  trust_remote_code=True, torch_dtype=torch.half)
config.is_causal=False
model = AutoModelForMaskedLM.from_pretrained("biomap-research/xtrimopglm-100b-int4", config = config, torch_dtype=torch.half,trust_remote_code=True)
if torch.cuda.is_available():
    model = model.cuda()

# # if you don't have the single gpu with 80G memory, try the dispatch load.
# from accelerate import load_checkpoint_and_dispatch, init_empty_weights
# with init_empty_weights():
  # model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
# 
# model = load_checkpoint_and_dispatch(
#     model, "<your model cached dir>", device_map="auto", no_split_module_classes=["xTrimoPGLMBlock"], strict=True, dtype=dtype
# )

model.eval()

seq = 'MILMCQHFSGQFSKYFLAVSSDFCHFVFPIILVSHVNFKQMKRKGFALWNDRAVPFTQGIFTTVMILLQYLHGTG'
output = tokenizer(seq, add_special_tokens=True, return_tensors='pt')
with torch.inference_mode():
    inputs = {"input_ids": output["input_ids"].cuda(), "attention_mask": output["attention_mask"].cuda()}
    output_embeddings = model(**inputs, output_hidden_states=True, return_last_hidden_state=True).hidden_states[:-1, 0] # get rid of the <eos> token


# model for the sequence-level tasks
model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)

# model for the token-level tasks
model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
```


Refer the *finetune* folder to check more finetuning examples, such as LoRA and Linear Probing.

### xTrimoPGLM-CLM: Casusal Langeuage Models for Protein Design 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

tokenizer  = AutoTokenizer.from_pretrained("biomap-research/xtrimopglm-100b-int4", trust_remote_code=True, use_fast=True)
config = AutoConfig.from_pretrained("biomap-research/xtrimopglm-100b-int4",  trust_remote_code=True, torch_dtype=torch.half)
config.is_causal=True
model = AutoModelForCausalLM.from_pretrained("biomap-research/xtrimopglm-100b-int4", config = config, torch_dtype=torch.half,trust_remote_code=True)
if torch.cuda.is_available():
    model = model.cuda()

# # if you don't have the single gpu with 80G memory, try the dispatch load.
# from accelerate import load_checkpoint_and_dispatch, init_empty_weights
# with init_empty_weights():
  # model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
# 
# model = load_checkpoint_and_dispatch(
#     model, "<your model cached dir>", device_map="auto", no_split_module_classes=["xTrimoPGLMBlock"], strict=True, dtype=dtype
# )
model.eval()

gen_kwargs = {'max_length': 256, 'top_p': 0.8, 'temperature':0.9, "num_beams": 1}
prompt=['', 'MLFVVL', 'LDL', 'VTQA']

for idx, each in enumerate(prompt):
    print(f"Begin generating idx: {idx} with prompt {each}")
    output = model.chat(tokenizer, each, **gen_kwargs)
    print(f"\nEnd generation with length: {len(output.split())} - seqs: {output}\n")
```
For more inference scrpts of other models, please visit the model card of the huggingface page.


## LICENSE

The code in this repository is open source under the [Creative Commons Attribution-NonCommercial 4.0 International License](./LICENSE).



## Citations

If you find our work useful, please consider citing the following paper:
```
@article{chen2024xtrimopglm,
  title={xTrimoPGLM: unified 100B-scale pre-trained transformer for deciphering the language of protein},
  author={Chen, Bo and Cheng, Xingyi and Li, Pan and Geng, Yangli-ao and Gong, Jing and Li, Shen and Bei, Zhilei and Tan, Xu and Wang, Boyan and Zeng, Xin and others},
  journal={arXiv preprint arXiv:2401.06199},
  year={2024}
}

@article{cheng2024training,
  title={Training Compute-Optimal Protein Language Models},
  author={Cheng, Xingyi and Chen, Bo and Li, Pan and Gong, Jing and Tang, Jie and Song, Le},
  journal={bioRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
