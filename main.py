import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import load_our_data, Data
from model_vlgraph import VLGraph, train_epoch, evaluate_101


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    set_seed(42)
    print("Loading data...")

    (train_data, test_data, num_items,
     image_cluster_feature, text_cluster_feature,
     item_image_list, item_text_list,
     user_train, user_valid, user_test,
     train_item_dict) = load_our_data(
        config['interaction_path'],
        config['item_path'],
        config['title_npy_path'],
        max_seq_len=config['max_seq_len'],
        K=config['num_cluster']
    )

    print(f"Items: {num_items}")
    config['num_item'] = num_items + 1  # 1-based

    max_len = config['max_seq_len']

    train_dataset = Data(train_data, num_items, max_len, link_k=config['link_k'])
    test_dataset  = Data(test_data,  num_items, max_len, link_k=config['link_k'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=config['num_workers'])
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])

    model = VLGraph(config, image_cluster_feature, text_cluster_feature,
                    item_image_list, item_text_list).cuda()

    best_val_ndcg  = 0.0
    best_test_ndcg = 0.0
    best_test_hr   = 0.0
    best_test_mrr  = 0.0
    num_decreases  = 0

    for epoch in range(config['num_epochs']):
        avg_loss = train_epoch(model, train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % config['eval_every'] == 0:
            # valid 평가: train 데이터셋 (train_seq → valid 아이템 예측)
            val_ndcg, val_hr, val_mrr = evaluate_101(
                model, train_loader, train_item_dict, num_items,
                topk=config['topk'], num_neg=config['num_neg']
            )
            print(f"  [Valid] NDCG@{config['topk']}: {val_ndcg:.4f} | "
                  f"HR@{config['topk']}: {val_hr:.4f} | "
                  f"MRR: {val_mrr:.4f}")

            # test 평가: test 데이터셋 (train+valid_seq → test 아이템 예측)
            test_ndcg, test_hr, test_mrr = evaluate_101(
                model, test_loader, train_item_dict, num_items,
                topk=config['topk'], num_neg=config['num_neg']
            )
            print(f"  [Test]  NDCG@{config['topk']}: {test_ndcg:.4f} | "
                  f"HR@{config['topk']}: {test_hr:.4f} | "
                  f"MRR: {test_mrr:.4f}")

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
        # ── 데이터 경로 ───────────────────────────────────────────────────
        'interaction_path': './interaction.parquet',
        'item_path':        './item_used.parquet',
        'title_npy_path':   './title_emb.npy',
        'save_path':        './best_vlgraph.pt',

        # ── 클러스터링 ────────────────────────────────────────────────────
        'num_cluster': 10,   # K-means K (image/text 각각)
        'link_k':       1,   # 아이템당 cluster 수

        # ── VLGraph 모델 파라미터 ─────────────────────────────────────────
        'embedding_size':      128,
        'aggregator':         'rgat',    # rgat, hete_attention, gcn, graphsage, gat
        'fusion_type':        'gate',    # cat, gate, asy_mask
        'n_layer':              2,
        'max_relid':           10,       # 엣지 타입 수 (1~10)
        'alpha':               0.2,
        'dropout_local':       0.1,
        'dropout_atten':       0.1,
        'auxiliary_info':     ['node_type', 'pos'],
        'modality_prediction': True,
        'seq_pooling':        'last',    # last, mean, attention

        # ── 학습 파라미터 ─────────────────────────────────────────────────
        'lr':            0.001,
        'weight_decay':  1e-5,
        'lr_dc':         0.1,
        'lr_dc_step':    3,
        'batch_size':    128,
        'num_workers':     2,
        'num_epochs':    100,
        'eval_every':      1,
        'topk':           10,           # NDCG@10, HR@10, MRR
        'patience':        3,
        'num_neg':        100,          # 101개 후보 (정답 1 + negative 100)
        'max_seq_len':    50,
    }

    main(config)
