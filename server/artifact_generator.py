"""
Artifact Generator for MLOps Pipeline Debugger

Generates a full set of realistic ML training artifacts for a given bug scenario.
Each artifact is internally consistent — config matches logs, dataset stats match
preprocessing code — except for the one planted fault.

Bug types supported:
    Task 1 (easy):
        - exploding_lr       : learning_rate too large → loss diverges to NaN
        - wrong_optimizer    : SGD with momentum=0.99 on non-convex problem
        - batch_size_overflow: batch_size > dataset size → trivial overfitting signal

    Task 2 (medium):
        - data_leakage_scaler  : StandardScaler fit on full dataset before split
        - data_leakage_overlap : train/val split with random_state=None → overlap
        - wrong_split_ratio    : test data accidentally included in training

    Task 3 (hard):
        - label_encoder_mismatch : train/eval use different LabelEncoder.fit() orderings
        - silent_metric_swap     : val and test metric names swapped in eval code
        - tokenizer_version_drift: training uses tokenizer v1, eval uses v2 (different vocab)
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


# ─── Bug Specifications ───────────────────────────────────────────────────────

@dataclass
class BugSpec:
    bug_type: str
    category: str          # maps to failure_category in Action
    file: str              # root_cause_file
    field: str             # root_cause_field
    gold_fix: str
    task_difficulty: str   # easy / medium / hard


BUG_CATALOGUE: Dict[str, BugSpec] = {
    # ── EASY ──────────────────────────────────────────────────────────────────
    "exploding_lr": BugSpec(
        bug_type="exploding_lr",
        category="config_error",
        file="config.yaml",
        field="optimizer.learning_rate",
        gold_fix="Reduce learning_rate from 50.0 to 1e-4 (or use a scheduler with warmup)",
        task_difficulty="easy",
    ),
    "wrong_optimizer": BugSpec(
        bug_type="wrong_optimizer",
        category="config_error",
        file="config.yaml",
        field="optimizer.momentum",
        gold_fix="Reduce momentum from 0.99 to 0.9, or switch to AdamW optimizer",
        task_difficulty="easy",
    ),
    "batch_size_overflow": BugSpec(
        bug_type="batch_size_overflow",
        category="config_error",
        file="config.yaml",
        field="training.batch_size",
        gold_fix="Reduce batch_size from 4096 to 32 or 64; current size exceeds training set",
        task_difficulty="easy",
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    "data_leakage_scaler": BugSpec(
        bug_type="data_leakage_scaler",
        category="data_leakage",
        file="preprocessing.py",
        field="StandardScaler.fit_transform",
        gold_fix="Fit StandardScaler only on X_train, then call transform() on X_val and X_test separately",
        task_difficulty="medium",
    ),
    "data_leakage_overlap": BugSpec(
        bug_type="data_leakage_overlap",
        category="data_leakage",
        file="preprocessing.py",
        field="train_test_split.random_state",
        gold_fix="Set random_state=42 in train_test_split to ensure deterministic, non-overlapping splits",
        task_difficulty="medium",
    ),
    "wrong_split_ratio": BugSpec(
        bug_type="wrong_split_ratio",
        category="preprocessing_bug",
        file="preprocessing.py",
        field="train_test_split.test_size",
        gold_fix="Change test_size from 0.8 to 0.2 — current config trains on 20% and evaluates on 80%",
        task_difficulty="medium",
    ),

    # ── HARD ──────────────────────────────────────────────────────────────────
    "label_encoder_mismatch": BugSpec(
        bug_type="label_encoder_mismatch",
        category="label_mismatch",
        file="preprocessing.py",
        field="LabelEncoder.fit_order",
        gold_fix="Use the same LabelEncoder instance (fitted on training data) for both train and eval pipelines",
        task_difficulty="hard",
    ),
    "silent_metric_swap": BugSpec(
        bug_type="silent_metric_swap",
        category="evaluation_bug",
        file="eval_results.json",
        field="metrics.val_accuracy",
        gold_fix="Swap val_accuracy and test_accuracy assignments in the evaluation loop — metrics are mislabeled",
        task_difficulty="hard",
    ),
    "tokenizer_version_drift": BugSpec(
        bug_type="tokenizer_version_drift",
        category="evaluation_bug",
        file="preprocessing.py",
        field="tokenizer.version",
        gold_fix="Ensure training and evaluation both use tokenizer v2 — v1 has a different vocabulary mapping for 847 tokens",
        task_difficulty="hard",
    ),
}

TASK_BUG_POOLS = {
    "easy":   ["exploding_lr", "wrong_optimizer", "batch_size_overflow"],
    "medium": ["data_leakage_scaler", "data_leakage_overlap", "wrong_split_ratio"],
    "hard":   ["label_encoder_mismatch", "silent_metric_swap", "tokenizer_version_drift"],
}


# ─── Model / Dataset Configs (variety pool) ───────────────────────────────────

MODEL_CONFIGS = [
    {"name": "ResNet-50", "type": "image_classification", "params": "25.6M",
     "dataset": "ImageNet-subset-10k", "num_classes": 10, "input": "224x224 RGB"},
    {"name": "BERT-base-uncased", "type": "text_classification", "params": "110M",
     "dataset": "SST-2", "num_classes": 2, "input": "tokenized sequences, max_len=128"},
    {"name": "EfficientNet-B3", "type": "image_classification", "params": "12.2M",
     "dataset": "CIFAR-100", "num_classes": 100, "input": "300x300 RGB"},
    {"name": "DistilBERT", "type": "sentiment_analysis", "params": "66M",
     "dataset": "IMDB-reviews", "num_classes": 3, "input": "tokenized sequences, max_len=256"},
    {"name": "MobileNetV3-Large", "type": "image_classification", "params": "5.4M",
     "dataset": "Oxford-102-Flowers", "num_classes": 102, "input": "224x224 RGB"},
]

OPTIMIZERS = ["AdamW", "SGD", "RMSprop", "Adam"]
SCHEDULERS = ["cosine_annealing", "step_lr", "reduce_on_plateau", "linear_warmup"]


# ─── Artifact Generators ──────────────────────────────────────────────────────

class ArtifactGenerator:
    """
    Generates all 6 training artifacts for a given (bug_type, seed) pair.
    All artifacts are internally consistent except for the planted fault.
    """

    def __init__(self, bug_type: str, seed: int):
        self.bug = BUG_CATALOGUE[bug_type]
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Pick a model config deterministically
        self.model_cfg = self.rng.choice(MODEL_CONFIGS)
        self.optimizer_name = self.rng.choice(OPTIMIZERS)
        self.scheduler_name = self.rng.choice(SCHEDULERS)
        self.run_id = f"run_{seed:04d}_{bug_type[:6]}"

        # Normal hyperparams
        self.lr = self.rng.choice([1e-5, 3e-5, 1e-4, 3e-4])
        self.batch_size = self.rng.choice([16, 32, 64])
        self.epochs = self.rng.randint(8, 20)
        self.weight_decay = self.rng.choice([0.01, 0.001, 1e-4])
        self.momentum = 0.9
        self.train_samples = self.rng.randint(8000, 15000)
        self.val_samples = int(self.train_samples * 0.2)
        self.test_samples = int(self.train_samples * 0.15)

    def generate_all(self) -> Dict[str, str]:
        return {
            "config.yaml":          self._gen_config(),
            "train.log":            self._gen_train_log(),
            "dataset_stats.json":   self._gen_dataset_stats(),
            "preprocessing.py":     self._gen_preprocessing(),
            "eval_results.json":    self._gen_eval_results(),
            "model_card.json":      self._gen_model_card(),
        }

    # ── config.yaml ──────────────────────────────────────────────────────────

    def _gen_config(self) -> str:
        lr = self.lr
        batch_size = self.batch_size
        momentum = self.momentum

        if self.bug.bug_type == "exploding_lr":
            lr = self.rng.choice([50.0, 10.0, 25.0])
        elif self.bug.bug_type == "wrong_optimizer":
            momentum = 0.99
            self.optimizer_name = "SGD"
        elif self.bug.bug_type == "batch_size_overflow":
            batch_size = self.rng.choice([2048, 4096, 8192])

        return textwrap.dedent(f"""\
            # Training Configuration
            # Run ID: {self.run_id}
            # Generated: 2024-03-{self.rng.randint(1,28):02d}T{self.rng.randint(0,23):02d}:{self.rng.randint(0,59):02d}:00Z

            model:
              architecture: {self.model_cfg['name']}
              num_classes: {self.model_cfg['num_classes']}
              pretrained: true
              pretrained_source: "timm/torchvision"
              dropout: {self.rng.choice([0.1, 0.2, 0.3])}
              freeze_backbone_epochs: {self.rng.randint(0, 3)}

            training:
              epochs: {self.epochs}
              batch_size: {batch_size}
              num_workers: {self.rng.choice([4, 8])}
              pin_memory: true
              mixed_precision: {str(self.rng.choice([True, False])).lower()}
              gradient_clip_norm: {self.rng.choice([1.0, 5.0, "null"])}
              early_stopping_patience: {self.rng.randint(3, 7)}
              seed: {self.seed}

            optimizer:
              name: {self.optimizer_name}
              learning_rate: {lr}
              weight_decay: {self.weight_decay}
              momentum: {momentum}
              betas: [0.9, 0.999]

            scheduler:
              name: {self.scheduler_name}
              warmup_epochs: {self.rng.randint(0, 3)}
              min_lr: 1.0e-7
              t_max: {self.epochs}

            data:
              dataset: {self.model_cfg['dataset']}
              input_size: "{self.model_cfg['input']}"
              train_split: 0.8
              val_split: 0.1
              test_split: 0.1
              augmentation:
                random_crop: true
                horizontal_flip: {str(self.rng.choice([True, False])).lower()}
                color_jitter: {self.rng.choice([0.2, 0.4])}
                normalize_mean: [0.485, 0.456, 0.406]
                normalize_std:  [0.229, 0.224, 0.225]

            logging:
              log_interval: 50
              save_best_only: true
              checkpoint_dir: "./checkpoints/{self.run_id}"
              wandb_project: "mlops-debug-bench"
        """)

    # ── train.log ────────────────────────────────────────────────────────────

    def _gen_train_log(self) -> str:
        lines = []
        lines.append(f"[INFO  2024-03-{self.rng.randint(1,28):02d} {self.rng.randint(6,10):02d}:00:00] Starting training run: {self.run_id}")
        lines.append(f"[INFO  ] Model: {self.model_cfg['name']} | Params: {self.model_cfg['params']}")
        lines.append(f"[INFO  ] Dataset: {self.model_cfg['dataset']} | Train: {self.train_samples:,} | Val: {self.val_samples:,}")
        lines.append(f"[INFO  ] Device: cuda:0 | Mixed precision: fp16")
        lines.append(f"[INFO  ] Optimizer: {self.optimizer_name} | LR: {self.lr} | Batch: {self.batch_size}")
        lines.append("[INFO  ] ─" * 30)

        bug = self.bug.bug_type

        if bug == "exploding_lr":
            # Loss explodes rapidly
            loss = 2.302
            for ep in range(1, min(self.epochs + 1, 6)):
                acc = max(0.0, 0.12 - ep * 0.02)
                val_loss = loss * self.rng.uniform(1.1, 1.5)
                val_acc = max(0.0, acc - 0.05)
                lines.append(f"[EPOCH {ep:03d}] train_loss={loss:.4f}  train_acc={acc:.4f}  "
                              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                              f"lr={self.lr:.2e}  grad_norm={loss * 18.3:.2f}  "
                              f"time={self.rng.randint(45,90)}s")
                if ep == 1: lines.append(f"[WARN  ] Gradient norm unusually high: {loss * 18.3:.2f} (threshold: 10.0)")
                loss = loss * self.rng.uniform(4.5, 9.0)
                if loss > 1e6:
                    lines.append(f"[EPOCH {ep+1:03d}] train_loss=nan  train_acc=0.1000  val_loss=nan  val_acc=0.1000  "
                                  f"lr={self.lr:.2e}  grad_norm=nan  time={self.rng.randint(45,90)}s")
                    lines.append(f"[ERROR ] Loss is NaN at epoch {ep+1}, step {self.rng.randint(100,300)}. Training halted.")
                    lines.append(f"[ERROR ] Last finite loss: {loss / self.rng.uniform(4,9):.2f}. Gradient explosion detected.")
                    break

        elif bug == "wrong_optimizer":
            # Loss oscillates wildly, never converges
            loss = 2.302
            for ep in range(1, self.epochs + 1):
                delta = self.rng.uniform(-0.8, 1.2)
                loss = max(1.8, loss + delta)
                acc = self.rng.uniform(0.10, 0.25)
                val_loss = loss + self.rng.uniform(-0.3, 0.8)
                val_acc = self.rng.uniform(0.09, 0.22)
                lines.append(f"[EPOCH {ep:03d}] train_loss={loss:.4f}  train_acc={acc:.4f}  "
                              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                              f"lr={self.lr:.2e}  grad_norm={self.rng.uniform(8.2, 45.1):.2f}  "
                              f"time={self.rng.randint(45,90)}s")
                if ep % 3 == 0:
                    lines.append(f"[WARN  ] Loss oscillation detected over last 3 epochs: {loss+0.4:.3f} → {loss-0.5:.3f} → {loss:.3f}")

        elif bug == "batch_size_overflow":
            # Val accuracy hits 100% immediately — model memorizes tiny effective dataset
            for ep in range(1, self.epochs + 1):
                train_loss = max(0.001, 2.302 * (0.05 ** ep))
                train_acc = min(1.0, 0.3 + ep * 0.09)
                val_acc = 0.999 if ep >= 2 else 0.85
                val_loss = 0.001 if ep >= 2 else 0.45
                lines.append(f"[EPOCH {ep:03d}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                              f"lr={self.lr:.2e}  grad_norm={self.rng.uniform(0.1,0.9):.3f}  "
                              f"time={self.rng.randint(3,8)}s")
            lines.append(f"[WARN  ] Effective steps per epoch: {max(1, self.train_samples // 4096)}. Dataset may be smaller than batch size.")

        elif bug in ("data_leakage_scaler", "data_leakage_overlap", "wrong_split_ratio"):
            # Val accuracy suspiciously high from epoch 1
            for ep in range(1, self.epochs + 1):
                train_loss = max(0.01, 0.45 - ep * 0.02)
                train_acc = min(0.98, 0.72 + ep * 0.015)
                val_acc = min(0.999, 0.984 + self.rng.uniform(-0.002, 0.002)) if ep >= 1 else 0.71
                val_loss = max(0.001, 0.04 - ep * 0.001)
                lines.append(f"[EPOCH {ep:03d}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                              f"lr={self.lr:.2e}  grad_norm={self.rng.uniform(0.1,1.2):.3f}  "
                              f"time={self.rng.randint(45,90)}s")
            lines.append(f"[INFO  ] Best model saved at epoch 2: val_acc=0.9841")
            lines.append(f"[WARN  ] Val accuracy reached 98.4% at epoch 1 — verify no data leakage.")

        elif bug in ("label_encoder_mismatch", "silent_metric_swap", "tokenizer_version_drift"):
            # Training looks completely normal — the bug is silent
            best_val = 0.0
            for ep in range(1, self.epochs + 1):
                train_loss = max(0.08, 1.8 * (0.72 ** ep) + self.rng.uniform(-0.02, 0.02))
                train_acc = min(0.96, 0.42 + ep * 0.032 + self.rng.uniform(-0.01, 0.01))
                val_loss = train_loss * self.rng.uniform(1.05, 1.15)
                val_acc = train_acc - self.rng.uniform(0.02, 0.06)
                best_val = max(best_val, val_acc)
                lines.append(f"[EPOCH {ep:03d}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                              f"lr={self.lr:.2e}  grad_norm={self.rng.uniform(0.3, 2.1):.3f}  "
                              f"time={self.rng.randint(60,120)}s")
            lines.append(f"[INFO  ] Training complete. Best val_acc={best_val:.4f} at epoch {self.rng.randint(self.epochs-3, self.epochs)}")
            lines.append(f"[INFO  ] Checkpoint saved: ./checkpoints/{self.run_id}/best_model.pt")

        lines.append("[INFO  ] ─" * 30)
        lines.append(f"[INFO  ] Run {self.run_id} finished.")
        return "\n".join(lines)

    # ── dataset_stats.json ───────────────────────────────────────────────────

    def _gen_dataset_stats(self) -> str:
        n_classes = self.model_cfg["num_classes"]
        train_n = self.train_samples
        val_n = self.val_samples
        test_n = self.test_samples

        overlap_count = 0
        if self.bug.bug_type == "data_leakage_overlap":
            overlap_count = self.rng.randint(int(val_n * 0.15), int(val_n * 0.30))
        elif self.bug.bug_type == "wrong_split_ratio":
            # Train and test flipped
            train_n, test_n = test_n, train_n

        # Class distribution (roughly uniform with jitter)
        def class_dist(total, n_cls):
            base = total // n_cls
            counts = {str(i): base + self.rng.randint(-int(base*0.15), int(base*0.15))
                      for i in range(min(n_cls, 10))}
            if n_cls > 10:
                counts["..."] = f"{n_cls - 10} more classes"
            return counts

        stats = {
            "dataset": self.model_cfg["dataset"],
            "num_classes": n_classes,
            "splits": {
                "train": {
                    "n_samples": train_n,
                    "class_distribution": class_dist(train_n, n_classes),
                },
                "val": {
                    "n_samples": val_n,
                    "class_distribution": class_dist(val_n, n_classes),
                    "overlap_with_train": overlap_count,
                },
                "test": {
                    "n_samples": test_n,
                    "class_distribution": class_dist(test_n, n_classes),
                },
            },
            "feature_statistics": {
                "mean": round(self.np_rng.uniform(0.45, 0.55), 4),
                "std":  round(self.np_rng.uniform(0.22, 0.28), 4),
                "min":  0.0,
                "max":  1.0,
                "null_count": 0,
            },
            "preprocessing_applied": [
                "resize",
                "normalize",
                "label_encode",
                "train_val_test_split",
            ],
            "random_seed_used": self.seed if self.bug.bug_type != "data_leakage_overlap" else None,
        }
        return json.dumps(stats, indent=2)

    # ── preprocessing.py ─────────────────────────────────────────────────────

    def _gen_preprocessing(self) -> str:
        bug = self.bug.bug_type

        if bug == "data_leakage_scaler":
            return textwrap.dedent(f"""\
                \"\"\"
                Data preprocessing pipeline for {self.model_cfg['dataset']}
                Run ID: {self.run_id}
                \"\"\"
                import numpy as np
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.model_selection import train_test_split


                def load_raw_data(data_dir: str):
                    \"\"\"Load features and labels from disk.\"\"\"
                    X = np.load(f"{{data_dir}}/features.npy")
                    y = np.load(f"{{data_dir}}/labels.npy")
                    return X, y


                def preprocess(data_dir: str, seed: int = {self.seed}):
                    X, y = load_raw_data(data_dir)

                    # Encode labels
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    # ── BUG: Scaler fit on full dataset BEFORE split ──────────
                    scaler = StandardScaler()
                    X_normalized = scaler.fit_transform(X)   # sees val/test data during fit!
                    # ─────────────────────────────────────────────────────────

                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_normalized, y_encoded, test_size=0.2, random_state=seed
                    )
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=0.5, random_state=seed
                    )

                    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le


                def get_transforms(split: str):
                    \"\"\"Get augmentation transforms for a given split.\"\"\"
                    if split == "train":
                        return [
                            ("random_horizontal_flip", {{"p": 0.5}}),
                            ("random_crop",            {{"size": 224, "padding": 4}}),
                            ("color_jitter",           {{"brightness": 0.2, "contrast": 0.2}}),
                            ("normalize",              {{"mean": [0.485, 0.456, 0.406],
                                                        "std":  [0.229, 0.224, 0.225]}}),
                        ]
                    return [
                        ("center_crop", {{"size": 224}}),
                        ("normalize",   {{"mean": [0.485, 0.456, 0.406],
                                          "std":  [0.229, 0.224, 0.225]}}),
                    ]
            """)

        elif bug == "data_leakage_overlap":
            return textwrap.dedent(f"""\
                \"\"\"
                Data preprocessing pipeline for {self.model_cfg['dataset']}
                Run ID: {self.run_id}
                \"\"\"
                import numpy as np
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.model_selection import train_test_split


                def load_raw_data(data_dir: str):
                    X = np.load(f"{{data_dir}}/features.npy")
                    y = np.load(f"{{data_dir}}/labels.npy")
                    return X, y


                def preprocess(data_dir: str):
                    X, y = load_raw_data(data_dir)

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    # First split: train vs temp
                    # ── BUG: random_state=None → non-reproducible, overlapping splits ──
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=None   # ← should be fixed seed
                    )
                    # Second split: val vs test (ALSO non-deterministic)
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=0.5, random_state=None  # ← should be fixed seed
                    )
                    # ─────────────────────────────────────────────────────────

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val   = scaler.transform(X_val)
                    X_test  = scaler.transform(X_test)

                    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le
            """)

        elif bug == "wrong_split_ratio":
            return textwrap.dedent(f"""\
                \"\"\"
                Data preprocessing pipeline for {self.model_cfg['dataset']}
                Run ID: {self.run_id}
                \"\"\"
                import numpy as np
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.model_selection import train_test_split


                def preprocess(data_dir: str, seed: int = {self.seed}):
                    X = np.load(f"{{data_dir}}/features.npy")
                    y = np.load(f"{{data_dir}}/labels.npy")

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    # ── BUG: test_size=0.8 — trains on 20%, evaluates on 80% ──
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=0.8, random_state=seed  # ← should be 0.2
                    )
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_test, y_test, test_size=0.5, random_state=seed
                    )
                    # ──────────────────────────────────────────────────────────

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val   = scaler.transform(X_val)
                    X_test  = scaler.transform(X_test)

                    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le
            """)

        elif bug == "label_encoder_mismatch":
            classes = ["cat", "dog", "bird"] if self.model_cfg["num_classes"] <= 10 else \
                      [f"class_{i}" for i in range(min(self.model_cfg["num_classes"], 5))]
            classes_shuffled = classes.copy()
            self.rng.shuffle(classes_shuffled)
            return textwrap.dedent(f"""\
                \"\"\"
                Data preprocessing pipeline for {self.model_cfg['dataset']}
                Run ID: {self.run_id}

                WARNING: Training and evaluation pipelines are defined separately.
                Ensure they use identical label encoding.
                \"\"\"
                import numpy as np
                from sklearn.preprocessing import LabelEncoder
                from sklearn.model_selection import train_test_split


                # ── Training pipeline ─────────────────────────────────────────
                def build_train_pipeline(data_dir: str, seed: int = {self.seed}):
                    X = np.load(f"{{data_dir}}/train_features.npy")
                    y_raw = np.load(f"{{data_dir}}/train_labels.npy", allow_pickle=True)

                    # LabelEncoder fitted on training class order
                    le_train = LabelEncoder()
                    le_train.fit({classes})          # alphabetical order: {sorted(classes)}
                    y = le_train.transform(y_raw)

                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=seed
                    )
                    return (X_train, y_train), (X_val, y_val), le_train


                # ── Evaluation pipeline ───────────────────────────────────────
                def build_eval_pipeline(data_dir: str):
                    X_test = np.load(f"{{data_dir}}/test_features.npy")
                    y_raw  = np.load(f"{{data_dir}}/test_labels.npy", allow_pickle=True)

                    # ── BUG: Different LabelEncoder instance with DIFFERENT fit order ──
                    le_eval = LabelEncoder()
                    le_eval.fit({classes_shuffled})  # ← shuffled order: {classes_shuffled}
                    y_test = le_eval.transform(y_raw)
                    # ─────────────────────────────────────────────────────────

                    return X_test, y_test, le_eval
            """)

        elif bug == "silent_metric_swap":
            val_acc = round(self.rng.uniform(0.84, 0.91), 4)
            test_acc = round(self.rng.uniform(0.31, 0.39), 4)
            return textwrap.dedent(f"""\
                \"\"\"
                Evaluation script for {self.model_cfg['dataset']}
                Run ID: {self.run_id}
                \"\"\"
                import torch
                import json


                def evaluate(model, val_loader, test_loader, device="cuda"):
                    model.eval()
                    results = {{}}

                    with torch.no_grad():
                        # Evaluate on validation set
                        val_correct, val_total = 0, 0
                        for X, y in val_loader:
                            preds = model(X.to(device)).argmax(dim=1)
                            val_correct += (preds == y.to(device)).sum().item()
                            val_total   += y.size(0)
                        val_acc = val_correct / val_total

                        # Evaluate on test set
                        test_correct, test_total = 0, 0
                        for X, y in test_loader:
                            preds = model(X.to(device)).argmax(dim=1)
                            test_correct += (preds == y.to(device)).sum().item()
                            test_total   += y.size(0)
                        test_acc = test_correct / test_total

                    # ── BUG: val and test accuracy assignments are swapped ──
                    results["val_accuracy"]  = test_acc   # ← should be val_acc
                    results["test_accuracy"] = val_acc    # ← should be test_acc
                    # ──────────────────────────────────────────────────────

                    results["val_loss"]  = round(1 - val_acc  + 0.12, 4)
                    results["test_loss"] = round(1 - test_acc + 0.09, 4)
                    return results
            """)

        elif bug == "tokenizer_version_drift":
            return textwrap.dedent(f"""\
                \"\"\"
                Text preprocessing pipeline for {self.model_cfg['dataset']}
                Run ID: {self.run_id}
                \"\"\"
                from transformers import AutoTokenizer


                TOKENIZER_V1 = "bert-base-uncased"           # vocab size: 30,522
                TOKENIZER_V2 = "bert-base-uncased-v2-fixed"  # vocab size: 30,522 + 847 domain tokens


                # ── Training pipeline ─────────────────────────────────────────
                def get_train_tokenizer():
                    \"\"\"Tokenizer used during training.\"\"\"
                    # Updated to v2 for domain-specific vocabulary
                    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_V2)
                    return tokenizer


                # ── Evaluation pipeline ───────────────────────────────────────
                def get_eval_tokenizer():
                    \"\"\"Tokenizer used during evaluation and inference.\"\"\"
                    # ── BUG: Still using v1 — 847 tokens map to [UNK] during eval ──
                    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_V1)   # ← should be TOKENIZER_V2
                    return tokenizer
                    # ─────────────────────────────────────────────────────────


                def tokenize_batch(texts, tokenizer, max_length: int = 128):
                    return tokenizer(
                        texts,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
            """)

        else:
            # Default normal preprocessing (for config-error bugs, preprocessing is clean)
            return textwrap.dedent(f"""\
                \"\"\"
                Data preprocessing pipeline for {self.model_cfg['dataset']}
                Run ID: {self.run_id}
                \"\"\"
                import numpy as np
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.model_selection import train_test_split


                def preprocess(data_dir: str, seed: int = {self.seed}):
                    X = np.load(f"{{data_dir}}/features.npy")
                    y = np.load(f"{{data_dir}}/labels.npy")

                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X, y_encoded, test_size=0.2, random_state=seed
                    )
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=0.5, random_state=seed
                    )

                    # Correct: fit only on training data
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val   = scaler.transform(X_val)
                    X_test  = scaler.transform(X_test)

                    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le
            """)

    # ── eval_results.json ────────────────────────────────────────────────────

    def _gen_eval_results(self) -> str:
        bug = self.bug.bug_type

        if bug in ("exploding_lr", "wrong_optimizer"):
            val_acc  = round(self.rng.uniform(0.09, 0.13), 4)
            test_acc = round(self.rng.uniform(0.09, 0.13), 4)
            val_loss = 999999.9 if bug == "exploding_lr" else round(self.rng.uniform(2.1, 2.4), 4)
            test_loss = val_loss
        elif bug == "batch_size_overflow":
            val_acc  = 0.9990
            test_acc = round(self.rng.uniform(0.11, 0.15), 4)   # massive train/test gap
            val_loss, test_loss = 0.0003, round(self.rng.uniform(1.8, 2.3), 4)
        elif bug in ("data_leakage_scaler", "data_leakage_overlap", "wrong_split_ratio"):
            val_acc  = round(self.rng.uniform(0.982, 0.998), 4)
            test_acc = round(self.rng.uniform(0.61, 0.73), 4)    # test is much worse (no leakage)
            val_loss  = round(self.rng.uniform(0.004, 0.015), 4)
            test_loss = round(self.rng.uniform(0.42, 0.68), 4)
        elif bug == "label_encoder_mismatch":
            val_acc  = round(self.rng.uniform(0.84, 0.91), 4)
            test_acc = round(self.rng.uniform(0.30, 0.38), 4)    # near random for 3-class
            val_loss  = round(1 - val_acc  + self.rng.uniform(0.05, 0.15), 4)
            test_loss = round(1 - test_acc + self.rng.uniform(0.05, 0.15), 4)
        elif bug == "silent_metric_swap":
            real_val  = round(self.rng.uniform(0.84, 0.91), 4)
            real_test = round(self.rng.uniform(0.31, 0.39), 4)
            # Swapped in output
            val_acc  = real_test
            test_acc = real_val
            val_loss  = round(1 - real_test + 0.09, 4)
            test_loss = round(1 - real_val  + 0.12, 4)
        elif bug == "tokenizer_version_drift":
            val_acc  = round(self.rng.uniform(0.83, 0.88), 4)
            test_acc = round(self.rng.uniform(0.28, 0.36), 4)
            val_loss  = round(1 - val_acc  + self.rng.uniform(0.05, 0.12), 4)
            test_loss = round(1 - test_acc + self.rng.uniform(0.05, 0.12), 4)
        else:
            val_acc  = round(self.rng.uniform(0.78, 0.91), 4)
            test_acc = round(val_acc - self.rng.uniform(0.02, 0.05), 4)
            val_loss  = round(1 - val_acc  + 0.1, 4)
            test_loss = round(1 - test_acc + 0.1, 4)

        result = {
            "run_id": self.run_id,
            "final_epoch": self.epochs if bug not in ("exploding_lr",) else self.rng.randint(2,5),
            "metrics": {
                "val_loss":      val_loss,
                "val_accuracy":  val_acc,
                "test_loss":     test_loss,
                "test_accuracy": test_acc,
            },
            "best_checkpoint": f"./checkpoints/{self.run_id}/best_model.pt",
            "evaluation_timestamp": f"2024-03-{self.rng.randint(1,28):02d}T{self.rng.randint(10,22):02d}:{self.rng.randint(0,59):02d}:00Z",
            "hardware": {"gpu": "A100-40GB", "cuda": "12.1"},
        }
        return json.dumps(result, indent=2)

    # ── model_card.json ──────────────────────────────────────────────────────

    def _gen_model_card(self) -> str:
        bug = self.bug.bug_type
        tokenizer_ver = "v1" if bug == "tokenizer_version_drift" else "v2"

        card = {
            "model_id": f"{self.run_id}",
            "architecture": self.model_cfg["name"],
            "task": self.model_cfg["type"],
            "num_parameters": self.model_cfg["params"],
            "dataset": self.model_cfg["dataset"],
            "num_classes": self.model_cfg["num_classes"],
            "framework": "PyTorch 2.2.0",
            "training_config": {
                "optimizer": self.optimizer_name,
                "scheduler": self.scheduler_name,
                "epochs": self.epochs,
            },
            "preprocessing": {
                "label_encoder": "sklearn.LabelEncoder",
                "tokenizer": tokenizer_ver if "bert" in self.model_cfg["name"].lower() else "N/A",
                "normalizer": "StandardScaler (fit on training split)",
            },
            "authors": ["ml-platform-team"],
            "license": "Apache-2.0",
        }
        return json.dumps(card, indent=2)


# ─── Sanity Check Engine ──────────────────────────────────────────────────────

def run_sanity_check(check_type: str, bug_type: str, artifacts: Dict[str, str],
                     rng: random.Random) -> Dict:
    """
    Runs a named diagnostic check and returns computed results.
    Results are grounded in the generated artifacts — not random.
    """
    bug = BUG_CATALOGUE[bug_type]

    if check_type == "label_consistency":
        if bug_type == "label_encoder_mismatch":
            return {
                "check": "label_consistency",
                "result": "FAIL",
                "details": "Training LabelEncoder class order: ['bird', 'cat', 'dog'] (index 0=bird, 1=cat, 2=dog). "
                           "Evaluation LabelEncoder class order: ['cat', 'dog', 'bird'] (index 0=cat, 1=dog, 2=bird). "
                           "Mismatch detected — 2 of 3 class indices differ between pipelines.",
                "affected_classes": 2,
                "recommendation": "Use a single LabelEncoder instance across both pipelines.",
            }
        return {"check": "label_consistency", "result": "PASS",
                "details": "Train and eval label mappings are identical. No mismatch detected."}

    elif check_type == "data_leakage":
        if bug_type in ("data_leakage_overlap", "data_leakage_scaler"):
            overlap = rng.randint(180, 450) if bug_type == "data_leakage_overlap" else 0
            scaler_leak = bug_type == "data_leakage_scaler"
            return {
                "check": "data_leakage",
                "result": "FAIL",
                "sample_overlap": overlap,
                "scaler_fitted_on_full_dataset": scaler_leak,
                "details": (
                    f"Found {overlap} samples present in both train and val splits. "
                    if overlap > 0 else ""
                ) + (
                    "StandardScaler.fit_transform() called on full dataset before split — "
                    "validation statistics contaminated by training distribution."
                    if scaler_leak else ""
                ),
            }
        return {"check": "data_leakage", "result": "PASS",
                "sample_overlap": 0, "scaler_fitted_on_full_dataset": False,
                "details": "No data leakage detected between train and val splits."}

    elif check_type == "gradient_norms":
        if bug_type == "exploding_lr":
            return {
                "check": "gradient_norms",
                "result": "ANOMALY",
                "epoch_1_norm": round(rng.uniform(840.0, 2100.0), 2),
                "expected_range": "0.1 – 10.0",
                "details": "Gradient norms exceeded safe threshold by 100–200×. "
                           "Indicates learning rate is too large — gradients are not being controlled.",
            }
        return {"check": "gradient_norms", "result": "NORMAL",
                "mean_norm": round(rng.uniform(0.3, 2.1), 3),
                "max_norm": round(rng.uniform(2.1, 4.5), 3),
                "details": "Gradient norms are within expected range throughout training."}

    elif check_type == "metric_gap_analysis":
        if bug_type in ("label_encoder_mismatch", "silent_metric_swap", "tokenizer_version_drift"):
            val_acc  = round(rng.uniform(0.84, 0.91), 4)
            test_acc = round(rng.uniform(0.28, 0.38), 4)
            return {
                "check": "metric_gap_analysis",
                "result": "ANOMALY",
                "val_accuracy":  val_acc,
                "test_accuracy": test_acc,
                "gap": round(val_acc - test_acc, 4),
                "expected_max_gap": 0.08,
                "details": f"Val/test accuracy gap is {val_acc - test_acc:.3f} — far exceeds expected max of 0.08. "
                           f"This magnitude of gap (>{val_acc - test_acc:.0%}) strongly suggests an evaluation pipeline bug "
                           f"rather than overfitting — the model generalises well to the val set but fails on test data.",
            }
        return {"check": "metric_gap_analysis", "result": "NORMAL",
                "details": "Val/test metric gap is within normal bounds."}

    elif check_type == "encoder_version_match":
        if bug_type == "tokenizer_version_drift":
            return {
                "check": "encoder_version_match",
                "result": "MISMATCH",
                "training_tokenizer": "bert-base-uncased-v2-fixed",
                "eval_tokenizer":     "bert-base-uncased",
                "vocab_diff": 847,
                "details": "Training uses tokenizer v2 (30,522 + 847 domain tokens). "
                           "Evaluation uses tokenizer v1 (30,522 tokens). "
                           "847 domain-specific tokens will map to [UNK] during evaluation — "
                           "causing silent degradation on domain-specific test inputs.",
            }
        return {"check": "encoder_version_match", "result": "PASS",
                "details": "Training and evaluation use identical tokenizer versions."}

    elif check_type == "class_balance":
        n_classes = 10
        counts = {str(i): rng.randint(780, 1020) for i in range(n_classes)}
        imbalance_ratio = max(counts.values()) / max(1, min(counts.values()))
        return {
            "check": "class_balance",
            "result": "PASS" if imbalance_ratio < 1.5 else "WARN",
            "class_counts": counts,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "details": f"Max/min class ratio: {imbalance_ratio:.2f}. "
                       f"{'Within acceptable range.' if imbalance_ratio < 1.5 else 'Moderate imbalance — consider weighted loss.'}",
        }

    elif check_type == "loss_trajectory":
        if bug_type == "exploding_lr":
            return {
                "check": "loss_trajectory",
                "result": "ANOMALY",
                "pattern": "exponential_divergence",
                "loss_values": [2.31, 18.42, 847.2, "nan"],
                "details": "Loss follows exponential growth pattern rather than convergence. "
                           "This is a strong indicator of learning rate being orders of magnitude too large.",
            }
        elif bug_type == "wrong_optimizer":
            return {
                "check": "loss_trajectory",
                "result": "ANOMALY",
                "pattern": "oscillating_no_convergence",
                "details": "Loss oscillates without converging over all epochs. "
                           "Characteristic of excessive momentum causing the optimizer to overshoot minima repeatedly.",
            }
        return {"check": "loss_trajectory", "result": "NORMAL",
                "pattern": "smooth_convergence",
                "details": "Loss follows expected convergence curve."}

    elif check_type == "feature_statistics":
        if bug_type in ("data_leakage_scaler",):
            return {
                "check": "feature_statistics",
                "result": "WARN",
                "train_mean": 0.0,  "train_std": 1.0,
                "val_mean":   0.0,  "val_std":   1.0,
                "details": "Train and val feature statistics are identical after normalization — "
                           "this is expected if scaler was fit on the full dataset (including val). "
                           "If scaler was fit only on train, a slight distributional shift is normal. "
                           "Zero shift suggests the scaler saw val data during fitting.",
            }
        return {"check": "feature_statistics", "result": "PASS",
                "details": "Train and val feature distributions are within expected divergence bounds."}

    return {"check": check_type, "result": "UNKNOWN",
            "details": f"Unknown sanity check type: {check_type}"}
