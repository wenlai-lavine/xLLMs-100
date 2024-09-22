from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, Dict, List
from datasets import load_dataset
import torch
import json
import transformers
from transformers import GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os


# import ptvsd
# ptvsd.enable_attach(address=('0.0.0.0', 5678))
# ptvsd.wait_for_attach()


DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "translate_instruct": "Please translate the following {source_lang} sentence to {target_lang}: {source_sentence}"
}



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    peft_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    cache_dir: str = field(
        default='/fs/scratch/rng_cr_bcai_dl/law1rng/.cache',
        metadata={"help": "set the cache dir."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    torch_dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata={"help": "The dtype to use for inference."},
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    nllb_config: str = field(default='/fs/scratch/rng_cr_bcai_dl/law1rng/lavine_bosch/lavine_code/RLCLF/data_preprocess/config_file/country_nllb.csv', metadata={"help": "Path to the training data."})


@dataclass
class GeneratingArguments:
    batch_size: int = field(default=8)
    output_path: str = field(default=None, metadata={"help": "Path to the output."})
    temperature: float = field(default=0.7)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.75)
    top_k: float = field(default=40)
    num_beams: int = field(default=1)
    max_new_tokens: int = field(default=512)
    template: str = field(default="alpaca")
    labels: Optional[List[str]] = field(default=None)
    transcot: bool = field(default=False)
    transcot_skip_example: bool = field(default=False)
    evaluate: str = field(default="generate")
    src_lang: str = field(default='eng_Latn')
    src_lang_name: str = field(default='English')
    tgt_lang: str = field(default='zho_Hans')
    tgt_lang_name: str = field(default='Chinese')
    sample_size: int = field(default=200)
    from_english: bool = field(default=True)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GeneratingArguments))
    model_args, data_args, generating_args = parser.parse_args_into_dataclasses()

    peft_model_id = model_args.peft_model_name_or_path
    peft_config = PeftConfig.from_pretrained(peft_model_id)

    model = transformers.AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, cache_dir=model_args.cache_dir)
    model = PeftModel.from_pretrained(model, peft_model_id)
    model = model.cuda()

    model.eval()

    if torch.cuda.device_count() > 1:
        from accelerate import load_checkpoint_and_dispatch
        load_checkpoint_and_dispatch(
            model,
            model_args.model_name_or_path,
            device_map="auto",
            offload_state_dict=True,
            no_split_module_classes=["LlamaDecoderLayer"],
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.padding_side = "left"

    def generate_prompt(instruction, input=None, template="alpaca"):
        if template == "alpaca":
            if input:
                return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        elif template == "raw":
            if input:
                return f"{instruction}\n\n{input}"
            else:
                return f"{instruction}"
        else:
            raise NotImplementedError

    def evaluate_by_generate(
        dataset,
        template,
        generation_config
    ):
        # _INSTRUCTION = "Given a premise and two options (choice1 and choice2), please help me to pick the more plausible option as choice1 or choice2 by answer the quesiton: what is the {question}?\nPremise: {premise}\nchoice1: {choice1} choice2:{choice2}"
        _INSTRUCTION = 'Here is a premise: {premise}. What is the {question}? Help me pick the more plausible option: -A: {choice1}, -B: {choice2}.'
        prompt = [generate_prompt(instruction=_INSTRUCTION.format(premise=premise, question=question, choice1=choice1, choice2=choice2), template=template) for premise, question, choice1, choice2 in zip(dataset["premise"], dataset["question"], dataset['choice1'], dataset['choice2'])]
        # prompt = [generate_prompt(ins, template) for ins in dataset["instruction"]]
        inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        return dataset | {"prediction": [o[len(p):].strip() for p, o in zip(prompt, output)]}

    generation_config = GenerationConfig(
        temperature=generating_args.temperature,
        do_sample=generating_args.do_sample,
        top_p=generating_args.top_p,
        top_k=generating_args.top_k,
        num_beams=max(2, generating_args.num_beams) if generating_args.labels else generating_args.num_beams,
        max_new_tokens=generating_args.max_new_tokens,
        force_word_ids=[tokenizer(generating_args.labels, add_special_tokens=False)["input_ids"]] if generating_args.labels else None
    )

    ## 新建输出路径
    os.makedirs(os.path.join(generating_args.output_path, model_args.model_name_or_path, 'XCOPA'), exist_ok=True)

    lang_list = ['et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh']

    ### 数据处理
    ## 循环
    for lang in lang_list:
        print('generate the results for ' + lang + '.')
        ## load dataset from list
        dataset = load_dataset('json', data_files=os.path.join(data_args.data_path, lang, 'test.' + lang + '.jsonl'))

        output_file = open(os.path.join(generating_args.output_path, model_args.model_name_or_path, 'XCOPA',  lang + '.jsonl'), 'w')

        # with open(os.path.join(generating_args.output_path, src_lang + '-' + tgt_lang + '.jsonl'), "w") as output_file:
        for i in tqdm(range(0, len(dataset['train']), generating_args.batch_size)):
            d = dataset['train'][i:i + generating_args.batch_size]
            ## ? translate input
            if generating_args.evaluate == "generate":
                output = evaluate_by_generate(d, template=generating_args.template, generation_config=generation_config)
            output_file.writelines(
                json.dumps(sample, ensure_ascii=False) + "\n" for sample in [dict(zip(output.keys(),t)) for t in zip(*output.values())]
            )
            output_file.flush()

if __name__ == "__main__":
    inference()
