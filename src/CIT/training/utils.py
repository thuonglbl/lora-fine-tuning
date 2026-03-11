import os
import subprocess

import numpy as np
import wandb
from CIT.config import settings
from sklearn.model_selection import KFold, train_test_split
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
MAX_SEQ_LENGTH = 8000  # Choose any! We auto support RoPE Scaling internally!
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.

recall_guidelines = """Remember to cite your sources, especially the Confluence URLs according to the format I gave you and to simply say that you don't have the information when wou cannot answer.
    You must absolutely not invent or give infomation that is not in the context.
"""
system_message_content = f"""You are an assistant working for an internal IT service of a company called {settings.COMPANY_NAME}. Always answer within the context of the company.
You work as a RAG. I'll give you tutorials about IT services.
Answer only with the context I give you, detail the steps the user has to follow to solve his issue.
Do not invent information that is not in the context so if you cannot answer the question,
say `I don't have this information`.
When you give an information, always cite from which source (title + document url) your answer comes from. The answer should remain concise.
So the goal is to provide the detailed steps to solve the issue and to cite the source of the information you provide.
Your answer must respect the following format:\n
Sources: Title(s) of the document(s)\n
URL: url(s) of the relevant document(s)
To solve this issue apply the following steps:\n
1 Step 1\n
2 Step 2\n
3 Step 3\n etc.\n

"""

llama31_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Context:{context}\n

{recall_guidelines}\n
Question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer:
{answer}<|eot_id|>"""


class Kfolds():
    """
    Class to perform K-fold cross-validation on a dataset, with balance on the =-urls questions.
    Args:
        dataset: A datasets.Dataset object containing the dataset to be split.
        n_splits: The number of splits to perform.
        test_size: The proportion of the dataset to include in the test split."""
    def __init__(self,dataset, n_splits=5):
        self.dataset = dataset
        self.n_splits = n_splits
        #first build folds on questions with at least one url to retrieve
        ids_with_urls = set([q["id"] for q in dataset if len(q["urls"]) > 0])
        unique_ids = np.array(sorted(ids_with_urls)) # sort to have a deterministic order

        # Create KFold object
        kf = KFold(n_splits=self.n_splits,shuffle=True, random_state=42)
        folds=kf.split(unique_ids)
        folds_list_ids=[]
        for train_idxs, test_idxs in folds:
            folds_list_ids.append({"train":list(unique_ids[train_idxs]), "test":list(unique_ids[test_idxs])})

        #now split with relevant balance of questions without urls
        ids_zero_urls = np.array(list(set([q["id"] for q in dataset if len(q["urls"]) == 0])))
        if len(ids_zero_urls) < self.n_splits:#in case there are not enough questions without urls, we just use the folds on questions with urls
            self.splits_ids = folds_list_ids
        else:
            folds_0_urls=kf.split(ids_zero_urls)
            folds_0_urls_list_ids=[]
            for train_idxs, test_idxs in folds_0_urls:
                folds_0_urls_list_ids.append({"train":list(ids_zero_urls[train_idxs]), "test":list(ids_zero_urls[test_idxs])})
            # now merge the two folds
            self.splits_ids = []
            for i in range(len(folds_list_ids)):
                self.splits_ids.append({"train": folds_list_ids[i]["train"]+folds_0_urls_list_ids[i]["train"],
                                        "test": folds_list_ids[i]["test"]+folds_0_urls_list_ids[i]["test"]})
                

        self.splits = []
        for i in range(len(self.splits_ids)):
            train_ids = self.splits_ids[i]["train"]
            test_ids = self.splits_ids[i]["test"]
            train = [q for q in dataset if q["id"] in train_ids]
            test = [q for q in dataset if q["id"] in test_ids]
            self.splits.append((train, test))

    def get_split(self, index):
        return self.splits[index]


def format_func(dataset):
    """Function to format the dataset for training.
    Args:
        dataset: A datasets.Dataset object containing the dataset to be formatted.
    Returns:
        dict: A dictionary containing the formatted dataset.
    """
    texts = []
    questions = dataset["question"]
    answers = dataset["answer"]
    retrieved_contexts = dataset["retrieved_context"]
    for q, a, c in zip(questions, answers, retrieved_contexts):
        prompt = llama31_prompt.format(
            system_prompt=system_message_content,
            context=c,
            recall_guidelines=recall_guidelines,
            question=q,
            answer=a,
        )
        texts.append(prompt)
    return {"text": texts}




def format_dataset(dataset_questions):
    """
    Function to format the dataset for training.
    Args:
        dataset_questions: A datasets.Dataset object containing the dataset to be formatted.
        Returns:
        """
    formatted_dataset = dataset_questions.map(format_func, batched=True)
    # split on the ids to avoid having the same question in train and validation sets
    ids = [q["id"] for q in dataset_questions]
    unique_ids = list(set(ids))
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
    train_dataset = formatted_dataset.filter(lambda x: x["id"] in train_ids)
    val_dataset = formatted_dataset.filter(lambda x: x["id"] in val_ids)
    return train_dataset, val_dataset

def load_model_with_lora(base_model_name, r, alpha, lora_dropout):
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    device_map={"": 0},
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    dtype=DTYPE,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)  
    model = FastLanguageModel.get_peft_model(
    model,
    r=r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=alpha,  # Choose any number > 0 ! Suggested *2 rank
    lora_dropout=lora_dropout,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

with open("wandb_key.txt", "r") as f:
    WANDB_KEY = f.read()
wandb.login(key=WANDB_KEY)

def build_trainer(model, tokenizer, train_dataset, val_dataset, batch_size, num_epochs, run_name, output_dir):
    """
    Function to build the trainer for training the model.
    Args:
        model: The model to be trained.
        tokenizer: The tokenizer to be used.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        batch_size: The batch size to be used for training.
        num_epochs: The number of epochs to train for.
        run_name: The name of the run for wandb.
        output_dir: The directory to save the model and logs.
    Returns:
        trainer: The trainer object.
    """
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=8,
            packing=False,  # Can make training 5x faster for short sequences.(not the case for us)  
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(int(batch_size / 2),1),
            gradient_accumulation_steps=4,  # 4x batch_size
            warmup_steps=5,  # 5 steps with linearly increasing learning rate from 0 to the set learning_rate
            num_train_epochs=num_epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=69,
            eval_strategy="steps",
            eval_steps=0.1,
            eval_on_start=True,
            report_to="wandb",
            run_name=run_name,
            output_dir=output_dir,
        ),
    )
    return trainer



### Functions used to create modelfiles and create ollama models

def create_modelfile_from_base_model_file(base_modelfile_path, output_model_path, output_modelfile_path):
    """
    Function to create a model file from a base model file.
    Args:
        base_model_file: The base model file to be used.
        output_model_file: The output model folder where the adapter is.
        output_modelfile_path: The output model file to be created.

    Returns:
        None
    """
    with open(base_modelfile_path, 'r') as file:
        modelfile = file.readlines()
    line_adapter=f"ADAPTER {output_model_path}\n"
    modelfile=modelfile[:5]+[line_adapter] + modelfile[5:]
    with open(output_modelfile_path, 'w') as file:
        file.writelines(modelfile)

    #create ollama model
    modelfile_abs_path = os.path.abspath(output_modelfile_path)
    model_name = os.path.basename(modelfile_abs_path)
    command_subprocess = ["ollama", "create", model_name, "-f", modelfile_abs_path]
    result = subprocess.run(command_subprocess, capture_output=True, text=True)
    print(result.stdout)
    print(f"Ollama model {model_name} created from {modelfile_abs_path}")
    return modelfile

def check_has_subfolers(path):
    has_subfolders = False

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            has_subfolders = True
            break
    return has_subfolders