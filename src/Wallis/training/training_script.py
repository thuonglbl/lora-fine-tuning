# script to train a base model using qLoRA (quantized in 4 bits) using unsloth librairy

import json
from argparse import ArgumentParser

import datasets
import wandb
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

MAX_SEQ_LENGTH = 6000  # Choose any! We auto support RoPE Scaling internally!
DTYPE = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.


parser = ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="unsloth/Meta-Llama-3.1-8B")
parser.add_argument(
    "--path_train_questions",
    type=str,
    default="../data/intermediate_results/train/retrieved_qu_1703_train.jsonl",
)
parser.add_argument(
    "--r", type=int, default=16, help="rank of the low-rank approximation"
)
parser.add_argument(
    "--alpha",
    type=int,
    default=16,
    help="alpha parameter for the low-rank approximation",
)
parser.add_argument("--lora_dropout", type=float, default=0)
parser.add_argument(
    "--run_name", type=str, default="llama31_18-03", help="name of the run in wandb"
)
parser.add_argument(
    "--batch_size", type=int, default=24, help="batch size for training"
)
parser.add_argument(
    "--num_epochs", type=int, default=1, help="number of epochs for training"
)
parser.add_argument(
    "--training_output_dir",
    type=str,
    default="outputs_18-03",
    help="output directory for the model",
)
parser.add_argument(
    "--output_model_path",
    type=str,
    default="outputs/model18_03",
    help="output directory for the model",
)

system_prompt = "You are an assistant working for the canton du Valais in Switzerland.\
        You work as a RAG. I'll give you text of laws of the canton,and a question, written in french .\
        Answer in french only within the context of the canton du Valais and with the context I give you.\
        Always cite from which article your answer comes from. And say that you don't know if you don't know.\
        You can also ask me to give you more context if you need it."

llama31_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Context:{context}
Question:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer:
{answer}<|eot_id|>"""


def format_dataset(dataset, use_retrieved_context=True):
    """
    takes a dataset and format it into train set compatible with SFT Trainer (with text key)
    """
    if use_retrieved_context:
        contexts = dataset["retrieved_context"]
    else:
        contexts = dataset["context"]
    questions = dataset["question"]
    answers = dataset["answer"]
    texts = []
    for context, question, answer in zip(contexts, questions, answers):
        ext = llama31_prompt.format(
            system_prompt=system_prompt,
            context=context,
            question=question,
            answer=answer,
        )
        texts.append(ext)
    return {"text": texts}


# load wandb key to view training stats live
with open("wandb_key.txt", "r") as f:
    WANDB_KEY = f.read()
wandb.login(key=WANDB_KEY)


if __name__ == "__main__":
    args = parser.parse_args()
    base_model_name = args.base_model_name
    r = args.r
    alpha = args.alpha
    lora_dropout = args.lora_dropout
    run_name = args.run_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    path_train_questions = args.path_train_questions
    output_dir = args.training_output_dir
    output_model_path = args.output_model_path

    # load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # apply lora adaptation with
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    with open(path_train_questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    debug = False
    if debug:
        questions = questions[:10]

    # preprocess train set
    dataset_questions = datasets.Dataset.from_list(questions)
    formatted_dataset = dataset_questions.map(format_dataset, batched=True)
    train_test_split = formatted_dataset.train_test_split(test_size=0.15, shuffle=False)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]

    # define training arguments
    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(batch_size / 2),
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=num_epochs,  # Set this to 1 for 1 full training run.
        # max_steps = 60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=69,
        eval_strategy="steps",
        eval_steps=20,
        eval_on_start=True,
        report_to="wandb",
        run_name=run_name,
        output_dir=output_dir,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=args,
    )
    trainer_stats = trainer.train()
    print("Training stats:")
    print(trainer_stats)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print("Model saved to:", output_model_path)
    print("Training finished")
