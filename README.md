## xLLMs-100
This repository is used in our paper:

[LLMs Beyond English: Scaling the Multilingual Capability of LLMs with Cross-Lingual Feedback (Findings of ACL)](https://aclanthology.org/2024.findings-acl.488/) 


-------
### Notes
+ Due to licences issues, the dataset and model used in this paper is not publicly avaliable on github or huggingface, if you are interested in the data, please contact Wen Lai (wen.lai@tum.de)
+ This respository was rebuild after the internship and only contains the training and inference code, please prepre the dataset for inference by yourself.


----------
**Requirments**
1. python==3.9
2. torch==2.1.0 (cuda==11.8)
3. transformers==4.35.2
4. trl==0.7.4
5. peft==0.6.0
----------

### Step 1: Data Preparation
+ Download the multilingual instruction / cross-lingual human feedback data from huggingface: ** or somewhere else?

### Step 2: Supervised Finetuning
+ The scripts are written in [LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0) and it can be easily converted to ```bash``` scripts or ```Slurm```, please note the environment variables and path settings.
```
bsub < scrripts/sft.bsub
```

### Step 3: Align LLMs with human feedback using DPO
```
bsub < scrripts/dpo.bsub
```

### Step 4: Inference
+ After training the model or directly download the model, you can generate the results in the benchmarks presented in our paper.
```
bsub < inference/$TASK/*.bsub
```


### Citation
```
@inproceedings{lai-etal-2024-llms,
    title = "{LLM}s Beyond {E}nglish: Scaling the Multilingual Capability of {LLM}s with Cross-Lingual Feedback",
    author = "Lai, Wen  and
      Mesgar, Mohsen  and
      Fraser, Alexander",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.488",
    pages = "8186--8213",
}
```
