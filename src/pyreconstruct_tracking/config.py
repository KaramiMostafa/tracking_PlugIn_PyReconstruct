from dataclasses import dataclass
from typing import Optional, Tuple
import yaml
from pathlib import Path

@dataclass
class PathsConfig:
    output_dir: str = "runs/out"

@dataclass
class ModelConfig:
    input_dim: Optional[int] = None
    embed_dim: int = 64
    num_heads: int = 2
    ff_hidden_dim: int = 256
    num_layers: int = 2
    output_dim: int = 2
    prior_mu: float = 0.0
    prior_sigma: float = 0.1
    dropout: float = 0.1
    use_layernorm: bool = True
    kl_beta: float = 0.0

@dataclass
class TrainConfig:
    num_epochs: int = 100
    lr: float = 1e-3
    margin: float = 0.2
    weight_decay: float = 1e-5
    batch_size: int = 128
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    device: str = "auto"  # "auto"|"cpu"|"cuda"

@dataclass
class MatchConfig:
    max_single_neighbors: int = 8
    max_triplet_neighbors: int = 10
    triplet_dist_thresh: float = 50.0
    belief_max_iter: int = 10
    belief_damping: float = 0.9
    base_percentile: int = 20
    motion_threshold: float = 200.0
    area_variation: float = 1.5

@dataclass
class VizConfig:
    static_every: int = 1           # 0 disables
    interactive_every: int = 0      # 0 disables

@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig
    train: TrainConfig
    match: MatchConfig
    viz: VizConfig

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    paths = PathsConfig(**cfg.get("paths", {}))
    model = ModelConfig(**cfg.get("model", {}))
    train = TrainConfig(**cfg.get("train", {}))
    match = MatchConfig(**cfg.get("match", {}))
    viz = VizConfig(**cfg.get("viz", {}))
    return Config(paths=paths, model=model, train=train, match=match, viz=viz)

def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
