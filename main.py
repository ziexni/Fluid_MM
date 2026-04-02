import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from vlgraph_data_utils import MicroVideoVLDataset
from vlgraph_model import VLGraph


# ──────────────────────────────────────────────────────────────────────────────
# 평가 함수
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model, dataset, mode='valid', num_neg=100, topk=10,
             batch_size=64, device='cuda'):
    """
    101개 후보 기반 평가 (정답 1 + negative 100)
    베이스라인 프로토콜:
    - valid: 유저 수 > 1000  → 1000명 랜덤 샘플링
    - test : 유저 수 > 10000 → 10000명 랜덤 샘플링
    - test 컨텍스트: train + valid 시퀀스
    """
    model.eval()

    eval_dict = dataset.user_valid if mode == 'valid' else dataset.user_test

    # 평가 가능한 유저 필터링
    all_valid_users = [u for u, items in eval_dict.items()
                       if len(items) > 0 and len(dataset.user_train[u]) >= 1]

    # 베이스라인과 동일한 유저 샘플링
    sample_limit = 1000 if mode == 'valid' else 10000
    if len(all_valid_users) > sample_limit:
        rng = np.random.RandomState(42)
        valid_users = rng.choice(all_valid_users, size=sample_limit, replace=False).tolist()
    else:
        valid_users = all_valid_users

    # 평가용 Dataset (모드별 시퀀스 구성)
    eval_dataset = MicroVideoVLDataset(
        dataset.interactions_df,
        dataset.image_feat[1:],  # 0-based로 다시 변환 (Dataset 내부에서 +1)
        dataset.title_feat[1:],
        max_seq_len=dataset.max_seq_len,
        mode=mode
    )
    # valid_users만 필터링
    valid_user_set = set(valid_users)
    eval_dataset.samples = [(u, seq, t) for u, seq, t in eval_dataset.samples
                            if u in valid_user_set]

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    # 유저별 표현 수집
    user_reprs = {}
    for batch in eval_loader:
        user_ids  = batch['user_id'].tolist()
        user_repr = model.get_user_repr(batch)  # (B, dim)
        for i, u in enumerate(user_ids):
            user_reprs[u] = user_repr[i]

    # 아이템 임베딩 (1-based)
    item_emb = model.embedding.weight[1:dataset.num_items].to(device)  # (num_items-1, dim)

    NDCG = HR = MRR = 0.0
    count = 0
    np.random.seed(42)

    for u in valid_users:
        if u not in user_reprs:
            continue
        u_repr = user_reprs[u]  # (dim,)

        target = eval_dict[u][0]['item_id']            # 1-based
        rated  = dataset.train_item_dict[u] | {target}

        # 후보 구성 (정답 1 + negative 100)
        cands = [target - 1]  # 0-based for item_emb
        for _ in range(num_neg):
            neg = np.random.randint(1, dataset.num_items)
            while neg in rated:
                neg = np.random.randint(1, dataset.num_items)
            cands.append(neg - 1)

        cands_tensor = torch.tensor(cands, dtype=torch.long, device=device)
        cand_emb     = item_emb[cands_tensor]           # (101, dim)
        scores       = torch.matmul(cand_emb, u_repr)   # (101,)

        rank = (-scores).argsort().argsort()[0].item()
        count += 1
        MRR   += 1 / (rank + 1)
        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HR   += 1

    N = max(count, 1)
    return NDCG / N, HR / N, MRR / N


# ──────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────────────────────────────────────
def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────
    print("Loading data...")
    interactions_df = pd.read_parquet(config['interaction_path'])
    item_df         = pd.read_parquet(config['item_path'])
    title_feat      = np.load(config['title_npy_path'])
    image_feat      = np.stack(item_df['video_feature'].values)

    config['num_items'] = interactions_df['item_id'].nunique() + 1  # 1-based max

    # ── Dataset / DataLoader ──────────────────────────────────────────────
    train_dataset = MicroVideoVLDataset(
        interactions_df, image_feat, title_feat,
        max_seq_len=config['max_seq_len'], mode='train'
    )
    # interactions_df 참조 보관 (evaluate에서 재사용)
    train_dataset.interactions_df = interactions_df

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers']
    )

    # ── 모델 초기화 ───────────────────────────────────────────────────────
    model = VLGraph(config, train_dataset.image_feat, train_dataset.title_feat).to(device)

    best_val_ndcg  = 0.0
    best_test_ndcg = 0.0
    best_test_hr   = 0.0
    best_test_mrr  = 0.0
    num_decreases  = 0

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            model.optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.optimizer.step()
            total_loss += loss.item()

        model.scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Loss: {avg_loss:.4f}")

        # 주기적 평가
        if (epoch + 1) % config['eval_every'] == 0:
            val_ndcg, val_hr, val_mrr = evaluate(
                model, train_dataset, mode='valid',
                num_neg=config['num_neg'], topk=config['topk'],
                batch_size=config['eval_batch_size'], device=device
            )
            print(f"  [Valid] NDCG@{config['topk']}: {val_ndcg:.4f} | "
                  f"HR@{config['topk']}: {val_hr:.4f} | "
                  f"MRR@{config['topk']}: {val_mrr:.4f}")

            test_ndcg, test_hr, test_mrr = evaluate(
                model, train_dataset, mode='test',
                num_neg=config['num_neg'], topk=config['topk'],
                batch_size=config['eval_batch_size'], device=device
            )
            print(f"  [Test]  NDCG@{config['topk']}: {test_ndcg:.4f} | "
                  f"HR@{config['topk']}: {test_hr:.4f} | "
                  f"MRR@{config['topk']}: {test_mrr:.4f}")

            # Best 모델 저장 & Early Stopping
            if val_ndcg > best_val_ndcg:
                best_val_ndcg  = val_ndcg
                best_test_ndcg = test_ndcg
                best_test_hr   = test_hr
                best_test_mrr  = test_mrr
                num_decreases  = 0
                torch.save(model.state_dict(), config['save_path'])
                print(f"  ✓ Best model saved (Val NDCG: {best_val_ndcg:.4f})")
            else:
                num_decreases += 1
                if num_decreases >= config['patience']:
                    print("Early stopping.")
                    break

    print(f"\n{'='*50}")
    print(f"Best Valid NDCG@{config['topk']}: {best_val_ndcg:.4f}")
    print(f"Best Test  NDCG@{config['topk']}: {best_test_ndcg:.4f} | "
          f"HR@{config['topk']}: {best_test_hr:.4f} | "
          f"MRR: {best_test_mrr:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    config = {
        # 데이터 경로
        'interaction_path': './interaction.parquet',
        'item_path':        './item_used.parquet',
        'title_npy_path':   './title_emb.npy',
        'save_path':        './vlgraph_best.pt',

        # 모델 구조
        'embedding_size': 128,      # 임베딩 차원
        'n_layer':          2,      # GNN 레이어 수
        'aggregator':   'rgat',     # rgat / hete_attention / kv_attention / gcn / graphsage / gat
        'max_relid':       10,      # 엣지 타입 수 (1~10)
        'alpha':          0.2,      # LeakyReLU negative slope
        'max_seq_len':     50,      # 최대 시퀀스 길이
        'fusion_type':   'gate',    # HeteAttenLayer용 (aggregator=hete_attention일 때)

        # 학습 설정
        'lr':            1e-3,      # 학습률
        'l2':            1e-5,      # L2 정규화
        'lr_dc':          0.1,      # LR decay 비율
        'lr_dc_step':      3,       # LR decay 주기 (에폭)
        'batch_size':     128,      # 학습 배치 크기
        'num_workers':      3,      # DataLoader 워커 수
        'num_epochs':     100,      # 최대 학습 에폭
        'eval_every':       5,      # 평가 주기
        'topk':            10,      # 평가 지표 K
        'patience':         3,      # Early stopping patience
        'num_neg':        100,      # negative 샘플 수
        'eval_batch_size':  64,     # 평가 배치 크기
        'dropout_local':   0.2,     # GNN dropout
        'dropout_atten':   0.2,     # Attention dropout

        # auxiliary info (node_type / pos 임베딩 사용 여부)
        'auxiliary_info': ['node_type', 'pos'],
    }

    train(config)
