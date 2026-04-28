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
from transformers import BertConfig, BertForSequenceClassification, AutoModel, AutoTokenizer, Trainer, EarlyStoppingCallback
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
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=64)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=250)
    eval_steps: int = field(default=250)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=300)
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
    report_to: Optional[str] = field(default='wandb')
    overwrite_output_dir: bool = field(default=True)
    log_level: str = field(default="info")
    eval_and_save_results: bool = field(default=True)
    early_stopping_patience: int = field(default=5, metadata={"help": "Stop after this many evals with no improvement."})
    early_stopping_threshold: float = field(default=0.0, metadata={"help": "Minimum improvement to count as an improvement for early stopping."})

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
        texts = [row[0] for row in data]
        labels = [[float(v) if v != '' else float('nan') for v in row[1:]] for row in data]

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

        #labels_arr = np.array(labels, dtype=float)
        #print(f"\n[Dataset] Loaded {data_path}")
        #print(f"[Dataset]   input_ids shape:      {self.input_ids.shape}  (n_samples, seq_len)")
        #print(f"[Dataset]   attention_mask shape: {self.attention_mask.shape}")
        #print(f"[Dataset]   num_labels: {self.num_labels}  |  label_names: {self.label_names}")
        #print(f"[Dataset]   NaN counts per label (these positions are masked out of loss & metrics):")
        #for j, name in enumerate(self.label_names):
        #    nan_count = int(np.isnan(labels_arr[:, j]).sum())
        #    valid_count = len(labels) - nan_count
        #    print(f"[Dataset]     '{name}': {nan_count} NaN  ({valid_count} valid)")

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
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        #if not getattr(self, '_debug_printed', False):
        #    self._debug_printed = True
        #    print(f"\n[Collator] First batch shapes:")
        #    print(f"[Collator]   input_ids:      {input_ids.shape}  (batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]})")
        #    print(f"[Collator]   attention_mask: {batch['attention_mask'].shape}  (1=real token, 0=padding)")
        #    print(f"[Collator]   labels:         {labels.shape}  (batch_size={labels.shape[0]}, num_labels={labels.shape[1]})")
        #    nan_per_label = torch.isnan(labels).sum(dim=0)
        #    valid_per_label = (~torch.isnan(labels)).sum(dim=0)
        #    print(f"[Collator]   NaN per label in this batch: {nan_per_label.tolist()}  (masked out of loss)")
        #    print(f"[Collator]   Valid per label in this batch: {valid_per_label.tolist()}  (used in loss)")

        return batch

class MaskedRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        mask = ~torch.isnan(labels)
        loss = torch.nn.functional.mse_loss(logits[mask], labels[mask], reduction='mean')

        #if not getattr(self, '_loss_debug_printed', False):
        #    self._loss_debug_printed = True
        #    print(f"\n[Loss] First compute_loss call:")
        #    print(f"[Loss]   labels shape: {labels.shape}  (batch_size={labels.shape[0]}, num_labels={labels.shape[1]})")
        #    print(f"[Loss]   logits shape: {logits.shape}  (batch_size={logits.shape[0]}, num_labels={logits.shape[1]})")
        #    print(f"[Loss]   mask shape:   {mask.shape}  (True=valid, False=NaN — excluded from loss)")
        #    total = mask.numel()
        #    valid = mask.sum().item()
        #    print(f"[Loss]   valid positions: {valid}/{total}  ({100*valid/total:.1f}% of batch×labels used in MSE)")
        #    print(f"[Loss]   logits[mask] shape: {logits[mask].shape}  — flattened valid predictions fed to MSE")
        #    print(f"[Loss]   labels[mask] shape: {labels[mask].shape}  — flattened valid targets fed to MSE")
        #    print(f"[Loss]   MSE loss (mean over valid positions): {loss.item():.6f}")

        return (loss, outputs) if return_outputs else loss


def calculate_metric_for_regression(logits: np.ndarray, labels: np.ndarray, label_names=None):
    """Calculate per-label and mean metrics for single- or multi-label regression."""
    #print(f"\n[Metrics] Raw logits shape: {logits.shape}, labels shape: {labels.shape}")

    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        #print(f"[Metrics] Reshaped 3D logits to: {logits.shape}")

    predictions = logits.squeeze()
    labels = labels.squeeze()

    # ensure 2D: (n_samples, n_labels)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
        labels = labels[:, np.newaxis]

    #print(f"[Metrics] After squeeze — predictions: {predictions.shape}, labels: {labels.shape}  (n_samples, n_labels)")

    n_labels = predictions.shape[1]
    metrics = {}

    all_valid_preds = []
    all_valid_labels = []
    per_label = {"pearson": [], "spearman": [], "r2": []}
    #print(f"[Metrics] Per-label masking and scores:")
    for i in range(n_labels):
        preds_i = predictions[:, i]
        labels_i = labels[:, i]
        valid = ~np.isnan(labels_i)
        name = label_names[i] if label_names else str(i)
        #print(f"[Metrics]   label {i} ('{name}'): {valid.sum()}/{len(labels_i)} valid samples after NaN mask")
        if valid.sum() < 2:
            print(f"[Metrics]     -> skipped (fewer than 2 valid samples)")
            continue
        preds_i = preds_i[valid]
        labels_i = labels_i[valid]

        # here we collect all valid predictions and labels because we want to
        # compute mse loss in a macro way not as a micro average
        # this way eval/mse-loss and train/mse-loss are calculated in the same way and comparable.
        all_valid_preds.append(preds_i)
        all_valid_labels.append(labels_i)

        # these metrics are calculated label-wise then averaged across labels
        # to match RiboNN's mean-R2 approach.
        pearson_i, _ = pearsonr(labels_i, preds_i)
        spearman_i, _ = spearmanr(labels_i, preds_i)
        r2_i = r2_score(labels_i, preds_i)
        per_label["pearson"].append(pearson_i)
        per_label["spearman"].append(spearman_i)
        per_label["r2"].append(r2_i)
        #print(f"[Metrics]     pearson={pearson_i:.4f}  spearman={spearman_i:.4f}  r2={r2_i:.4f}")

    metrics["mse_loss_mean"] = mean_squared_error(
        np.concatenate(all_valid_labels), np.concatenate(all_valid_preds)
    )
    metrics["pearson_corr_mean"] = np.mean(per_label["pearson"])
    metrics["spearman_corr_mean"] = np.mean(per_label["spearman"])
    metrics["r2_score_mean"] = np.mean(per_label["r2"])

    #print(f"[Metrics] Aggregated ({len(per_label['r2'])} labels contributed):")
    #print(f"[Metrics]   mse_loss_mean (macro over all valid positions): {metrics['mse_loss_mean']:.6f}")
    #print(f"[Metrics]   pearson_corr_mean: {metrics['pearson_corr_mean']:.4f}")
    #print(f"[Metrics]   spearman_corr_mean: {metrics['spearman_corr_mean']:.4f}")
    #print(f"[Metrics]   r2_score_mean:      {metrics['r2_score_mean']:.4f}")

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
    print("name of modules:")
    print([n for n, _ in model.named_modules() if 'attention' in n][:20])
    
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
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "dev.csv"))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "test.csv"))

    print(f"\n[Train] Dataset sizes: train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}")
    print(f"[Train] num_labels={train_dataset.num_labels}  label_names={train_dataset.label_names}")
    print(f"[Train] Model output head: num_labels={model_args.num_labels}  problem_type=regression")

    # this takes care of padding the input sequences to the same length in a batch and creating attention masks
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    label_names = train_dataset.label_names

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return calculate_metric_for_regression(logits, labels, label_names=label_names)

    # every certain number of steps, compute_metrics called on eval dataset
    trainer = MaskedRegressionTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience, early_stopping_threshold=training_args.early_stopping_threshold)],
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
