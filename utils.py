import sys
import copy
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Queue, Process


# ══════════════════════════════════════════════════════════════════════════════
# 인덱스 빌드
# ══════════════════════════════════════════════════════════════════════════════

def build_index(dataset_name):
    df = pd.read_parquet(dataset_name)
    intr = df[['user_id', 'item_id']]

    n_users = intr['user_id'].max()
    n_items = intr['item_id'].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for row in intr.itertuples(index=False):
        u2i_index[int(row.user_id)].append(int(row.item_id))
        i2u_index[int(row.item_id)].append(int(row.user_id))

    return u2i_index, i2u_index


# ══════════════════════════════════════════════════════════════════════════════
# 멀티모달 피처 로딩
# ══════════════════════════════════════════════════════════════════════════════

def load_item_features(interaction_path, item_path, title_npy_path):
    """
    실제 데이터셋 구조에 맞게 피처 룩업 테이블을 구성합니다.

    interaction.parquet : user_id, item_id, timestamp, watch_ratio, watch_seconds
    item_used.parquet   : item_id, video_feature (이미지/비디오), category_id (다중 리스트)
    title_emb.npy       : shape (num_items, title_dim) — item_df 행 순서와 동일

    반환
    -------
    text_feat     : np.ndarray (max_item_id + 1, title_dim)   — 0번 인덱스 = padding
    image_feat    : np.ndarray (max_item_id + 1, image_dim)   — 0번 인덱스 = padding
    category_feat : np.ndarray (max_item_id + 1, num_cats)    — multi-hot, 0번 = padding
    title_dim, image_dim, num_cats
    """
    item_df   = pd.read_parquet(item_path)
    title_raw = np.load(title_npy_path)          # (num_items, title_dim)

    # data_partition에서 item_id += 1 처리되므로 동일하게 1-based로 변환
    item_df = item_df.copy().reset_index(drop=True)
    item_df['item_id'] = item_df['item_id'] + 1
    max_item_id = int(item_df['item_id'].max())

    # ── 텍스트 (title_emb.npy) ───────────────────────────────────────────────
    title_dim = title_raw.shape[1]
    text_feat = np.zeros((max_item_id + 1, title_dim), dtype=np.float32)
    for raw_idx, row in item_df.iterrows():
        iid = int(row['item_id'])
        if raw_idx < len(title_raw):
            text_feat[iid] = title_raw[raw_idx].astype(np.float32)

    # ── 이미지/비디오 (video_feature 컬럼) ───────────────────────────────────
    sample_img = item_df['video_feature'].iloc[0]
    image_dim  = len(sample_img)
    image_feat = np.zeros((max_item_id + 1, image_dim), dtype=np.float32)
    for _, row in item_df.iterrows():
        iid = int(row['item_id'])
        image_feat[iid] = np.array(row['video_feature'], dtype=np.float32)

    # ── 카테고리 (category_id, 다중 리스트 → multi-hot) ─────────────────────
    all_cats = set()
    for cats in item_df['category_id']:
        if isinstance(cats, (list, np.ndarray)):
            all_cats.update(int(c) for c in cats)
        else:
            all_cats.add(int(cats))
    num_cats = max(all_cats) + 1  # 0-based category_id 기준

    category_feat = np.zeros((max_item_id + 1, num_cats), dtype=np.float32)
    for _, row in item_df.iterrows():
        iid  = int(row['item_id'])
        cats = row['category_id']
        if isinstance(cats, (list, np.ndarray)):
            for c in cats:
                category_feat[iid, int(c)] = 1.0
        else:
            category_feat[iid, int(cats)] = 1.0

    print(f"[load_item_features] title_dim={title_dim}, image_dim={image_dim}, "
          f"category_dim(multi-hot)={num_cats}, max_item_id={max_item_id}")

    return text_feat, image_feat, category_feat, title_dim, image_dim, num_cats


# ══════════════════════════════════════════════════════════════════════════════
# 샘플러
# ══════════════════════════════════════════════════════════════════════════════

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[uid][-1]
        idx = maxlen - 1
        ts  = set(user_train[uid])

        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        pos[maxlen - 1] = nxt
        neg[maxlen - 1] = random_neq(1, itemnum + 1, ts)

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids    = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors   = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(User, usernum, itemnum, batch_size, maxlen,
                              self.result_queue, np.random.randint(2e9))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 분할
# ══════════════════════════════════════════════════════════════════════════════

def data_partition(fname):
    df = pd.read_parquet(fname)
    df = df.sort_values(by=['user_id', 'timestamp'])

    df['user_id'] = df['user_id'] + 1
    df['item_id'] = df['item_id'] + 1

    usernum = df['user_id'].max()
    itemnum = df['item_id'].max()
    User    = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}

    for u, i in zip(df['user_id'], df['item_id']):
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user]  = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user]  = [User[user][-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]


# ══════════════════════════════════════════════════════════════════════════════
# 평가 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _make_feat_batch(seq, text_feat, image_feat, category_feat):
    """(L,) int 시퀀스 → 배치 차원 추가된 FloatTensor (1, L, dim)"""
    text_t  = torch.FloatTensor(text_feat[seq]).unsqueeze(0)     if text_feat     is not None else None
    image_t = torch.FloatTensor(image_feat[seq]).unsqueeze(0)    if image_feat    is not None else None
    cat_t   = torch.FloatTensor(category_feat[seq]).unsqueeze(0) if category_feat is not None else None
    return text_t, image_t, cat_t


def _make_cand_feat(item_idx, text_feat, image_feat, category_feat):
    """후보 아이템 리스트 → FloatTensor (num_cands, dim)"""
    arr     = np.array(item_idx)
    text_t  = torch.FloatTensor(text_feat[arr])     if text_feat     is not None else None
    image_t = torch.FloatTensor(image_feat[arr])    if image_feat    is not None else None
    cat_t   = torch.FloatTensor(category_feat[arr]) if category_feat is not None else None
    return text_t, image_t, cat_t


# ══════════════════════════════════════════════════════════════════════════════
# 평가 함수
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, dataset, args,
             text_feat=None, image_feat=None, category_feat=None):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG, HT, MRR = 0.0, 0.0, 0.0
    valid_user = 0.0

    users = (random.sample(range(1, usernum + 1), 10000)
             if usernum > 10000 else range(1, usernum + 1))

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # 시퀀스: train + valid 마지막 아이템
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if valid[u]:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated    = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        seq_text,  seq_image,  seq_cat  = _make_feat_batch(seq,      text_feat, image_feat, category_feat)
        cand_text, cand_image, cand_cat = _make_cand_feat(item_idx,  text_feat, image_feat, category_feat)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]],
            text_feat_seqs=seq_text, image_feat_seqs=seq_image, category_seqs=seq_cat,
            cand_text=cand_text,     cand_image=cand_image,     cand_cat=cand_cat,
        )
        predictions = predictions[0]
        rank        = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT   += 1
        MRR += 1 / (rank + 1)
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, MRR / valid_user


def evaluate_valid(model, dataset, args,
                   text_feat=None, image_feat=None, category_feat=None):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG, HT, MRR = 0.0, 0.0, 0.0
    valid_user = 0.0

    users = (random.sample(range(1, usernum + 1), 1000)
             if usernum > 1000 else range(1, usernum + 1))

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated    = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        seq_text,  seq_image,  seq_cat  = _make_feat_batch(seq,      text_feat, image_feat, category_feat)
        cand_text, cand_image, cand_cat = _make_cand_feat(item_idx,  text_feat, image_feat, category_feat)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]],
            text_feat_seqs=seq_text, image_feat_seqs=seq_image, category_seqs=seq_cat,
            cand_text=cand_text,     cand_image=cand_image,     cand_cat=cand_cat,
        )
        predictions = predictions[0]
        rank        = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT   += 1
        MRR += 1 / (rank + 1)
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, MRR / valid_user
