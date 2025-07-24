#!/usr/bin/env python3

import wandb
import subprocess
import os

# 색상 및 출력 관련 모든 기능 비활성화
os.environ["WANDB_CONSOLE"] = "off"

os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "20"
os.environ["NO_COLOR"] = "1"


# Sweep 설정
sweep_config = {
    'method': 'bayes',  # 베이지안 최적화
    'metric': {
        'name': 'Few/final_test_auc',
        'goal': 'maximize'
    },
    'parameters': {
        'source_lr': {'values': [0.001, 0.0001, 0.00001]},
        'source_lr_few': {'values': [0.001, 0.0001, 0.00001]},
        'hidden_dim': {'values': [256, 512, 768]},
        'num_layers': {'values': [2, 3, 4]},
        'n_heads': {'values': [2, 4, 8]},
        'dropout_rate': {'values': [0.1, 0.2, 0.3]},
        'random_seed': {'values': [42, 44, 46, 48, 50]},
        'attn_type': {'values': ['gat_v1']},  # gat_v1은 따로 실행
        'embed_type': {'value': 'carte'},
        'edge_type': {'value': 'mlp'},
        'source_data': {'value': 'heart'},
        'few_shot': {'values': [4, 8, 16, 32, 64]},
        'train_epochs': {'value': 1000},
        'model_type': {'value': 'TabularFLM'},
        'base_dir': {'value': 'wandbtest20250722'}
    }
}

def train():
    # Wandb run 초기화
    run = wandb.init()
    config = wandb.config
    
    # 실험 실행 명령어 생성
    cmd = [
        'python', 'main_G.py',
        '--random_seed', str(config.random_seed),
        '--source_data', config.source_data,
        '--base_dir', config.base_dir,
        '--embed_type', config.embed_type,
        '--edge_type', config.edge_type,
        '--attn_type', config.attn_type,
        '--few_shot', str(config.few_shot),
        '--train_epochs', str(config.train_epochs),
        '--model_type', config.model_type,
        '--source_lr', str(config.source_lr),
        '--source_lr_few', str(config.source_lr_few),
        '--hidden_dim', str(config.hidden_dim),
        '--num_layers', str(config.num_layers),
        '--n_heads', str(config.n_heads),
        '--dropout_rate', str(config.dropout_rate),
        '--use_wandb',
        '--wandb_project', 'tabular-hyperparams-sweep-gat_v1'
    ]
    
    # GPU 설정
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '4'
    
    print(f"Running: {' '.join(cmd)}")
    
    # 실험 실행
    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print("Experiment completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        print(f"Error output: {e.stderr}")
        wandb.log({"experiment_failed": True})

if __name__ == "__main__":
    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project="tabular-hyperparams-sweep-gat_v1")
    print(f"Sweep created: {sweep_id}")
    print(f"View at: https://wandb.ai/your-entity/tabular-hyperparams-sweep-gat_v1/sweeps/{sweep_id}")
    
    # Sweep 실행 (200번 실험)
    wandb.agent(sweep_id, train, count=200)