import os
import csv
import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import wandb
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, AutoModel, AutoTokenizer, Trainer
from transformers.models.bert.configuration_bert import BertConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import time
import dataclasses

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='YYLY66/mRNABERT')
    num_labels: int = field(default=1, metadata={"help": "Number of regression targets (output dimensions)."})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=32, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=64, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="q,v,wo", metadata={"help": "where to perform LoRA"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    num_train_epochs: int = field(default=20, metadata={"help": "Total number of training epochs to perform."})
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=2)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=32)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=5)
    save_steps: int = field(default=35)
    eval_steps: int = field(default=35)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=35)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=0.0001)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="r2_score_mean")
    greater_is_better: bool = field(default=True)
    output_dir: str = field(default="output_gena")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: Optional[str] = field(default='none')
    overwrite_output_dir: bool = field(default=True)
    log_level: str = field(default="info")
    eval_and_save_results: bool = field(default=True)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        with open(data_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        data = [row for row in data if all(v != '' for v in row[1:])]
        texts = [row[0] for row in data]
        labels = [[float(v) for v in row[1:]] for row in data]

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(labels[0]) if labels else 0
        self.label_names = header[1:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.tensor(labels).float()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def calculate_metric_for_regression(logits: np.ndarray, labels: np.ndarray, label_names=None):
    """Calculate per-label and mean metrics for single- or multi-label regression."""
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    predictions = logits.squeeze()
    labels = labels.squeeze()

    # ensure 2D: (n_samples, n_labels)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
        labels = labels[:, np.newaxis]

    n_labels = predictions.shape[1]
    metrics = {}

    per_label = {"mse": [], "pearson": [], "spearman": [], "r2": []}
    for i in range(n_labels):
        preds_i = predictions[:, i]
        labels_i = labels[:, i]
        pearson_i, _ = pearsonr(labels_i, preds_i)
        spearman_i, _ = spearmanr(labels_i, preds_i)
        per_label["mse"].append(mean_squared_error(labels_i, preds_i))
        per_label["pearson"].append(pearson_i)
        per_label["spearman"].append(spearman_i)
        per_label["r2"].append(r2_score(labels_i, preds_i))

    metrics["mse_loss_mean"] = np.mean(per_label["mse"])
    metrics["pearson_corr_mean"] = np.mean(per_label["pearson"])
    metrics["spearman_corr_mean"] = np.mean(per_label["spearman"])
    metrics["r2_score_mean"] = np.mean(per_label["r2"])

    return metrics

def train():
    """Train the model."""

    # parse arguments: TrainingArguments inherits from the HF class and adds some custom arguments for our use case
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb sweep passes all parameters as CLI args via ${args}, so no manual override needed.
    # Use WANDB_RUN_ID (set by the sweep agent before launch) to isolate each run's checkpoints.
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        run_name = f"run_{run_id}"
        training_args = dataclasses.replace(training_args, output_dir=os.path.join(training_args.output_dir, run_name), run_name=run_name)
    print(f"Output dir: {training_args.output_dir}")

    # model class definition is pulled from repo through trust_remote_code=True, AutoModelForSequenceClassification adds a regression head on top of the base model
    # MSE loss is used by default for regression tasks in Hugging Face Transformers when num_labels=1
    config = BertConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=model_args.num_labels,
        problem_type="regression",
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        config=config
    )

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # tokenizer loaded from base model, model_max_length is explicitely specified, default is 1024 here (truncation)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # datasets are loaded and tokenized according to naming convention (train.csv, dev.csv, test.csv)
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "train.csv"))
    print(f"num_labels in data: {train_dataset.num_labels}")
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "dev.csv"))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "test.csv"))

    # this takes care of padding the input sequences to the same length in a batch and creating attention masks
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    label_names = train_dataset.label_names

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return calculate_metric_for_regression(logits, labels, label_names=label_names)

    # every certain number of steps, compute_metrics called on eval dataset
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # this does a final test on the test set and saves the results in a json file in the output directory
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    start = time.perf_counter()
    train()
    end = time.perf_counter()
    print(f"Training completed in {end - start:.2f} seconds")
