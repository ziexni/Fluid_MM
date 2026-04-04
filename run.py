"""
FindRec - Baseline Runner
베이스라인 조건에 맞춘 실행 스크립트
"""

import warnings
warnings.filterwarnings('ignore')

import yaml
from main_fluid import train


def load_config(config_path='config.yaml'):
    """Load config from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    # Load config
    config = load_config('config.yaml')
    
    print("=" * 70)
    print("FindRec Training")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"Seed: {config['seed']}")
    print(f"Hidden dim: {config['id_embedding_dim']}")
    print(f"Num heads: {config['num_attention_heads']}")
    print(f"Dropout: {config['dropout_prob']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max seq len: {config['max_seq_len']}")
    print(f"Eval: {config['topk']}-candidate (1 + {config['num_neg']} neg)")
    print("=" * 70)
    
    # Train
    train(config)
