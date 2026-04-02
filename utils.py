import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


def data_partition(interaction_path):
    """
    원본 data_partition() 대체 — interaction.parquet 로드
    leave-two-out 분리 (원본과 동일한 구조 반환)

    반환:
        user_train : {user_id: [item_id, ...]}  1-based
        user_valid : {user_id: [item_id]}
        user_test  : {user_id: [item_id]}
        usernum    : 유저 수
        itemnum    : 아이템 수
    """
    df = pd.read_parquet(interaction_path)
    df = df.sort_values(['user_id', 'timestamp'])

    # 1-based 변환 (원본과 동일)
    df['user_id'] = df['user_id'] + 1
    df['item_id'] = df['item_id'] + 1

    usernum = df['user_id'].max()
    itemnum = df['item_id'].max()

    User = defaultdict(list)
    for u, i in zip(df['user_id'], df['item_id']):
        User[u].append(i)

    user_train = {}
    user_valid = {}
    user_test  = {}

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

    return user_train, user_valid, user_test, usernum, itemnum


def build_item_modality(item_path, title_feat, itemnum):
    """
    아이템별 image/title 노드 ID 매핑 구성

    원본 VLGraph에서 image_input[i] = [img_cluster_id, ...]  (K개)
    우리는 아이템당 image 1개, title 1개 → [[img_node_id]], [[title_node_id]] 형태로 맞춤

    image 노드 ID 공간: itemnum + item_id  (item과 겹치지 않게 offset)
    title 노드 ID 공간: itemnum*2 + item_id
    """
    # item_id는 1-based (data_partition과 동일하게 맞춤)
    item_to_image = {}  # {item_id(1-based): [img_node_id]}
    item_to_title = {}  # {item_id(1-based): [title_node_id]}

    item_df = pd.read_parquet(item_path)
    for idx, row in item_df.iterrows():
        item_id = int(row['item_id']) + 1  # 1-based
        img_node_id   = itemnum + item_id   # image 노드 offset
        title_node_id = itemnum * 2 + item_id  # title 노드 offset
        item_to_image[item_id] = [img_node_id]
        item_to_title[item_id] = [title_node_id]

    return item_to_image, item_to_title


class Data(Dataset):
    """
    원본 Data 클래스 구조 유지
    변경점:
    - inputs      : user_train 시퀀스 (세션 대체)
    - image_inputs: 아이템별 image 노드 ID 리스트
    - text_inputs : 아이템별 title 노드 ID 리스트
    - targets     : 다음 아이템 ID (train: 마지막 아이템, valid/test: 해당 아이템)
    """
    def __init__(self, data, item_to_image, item_to_title, link_k, train_len=None):
        # data = (inputs, image_inputs, text_inputs, targets)
        # 원본과 동일한 구조
        self.data_length = len(data[0])
        self.inputs       = data[0]
        self.image_inputs = data[1]
        self.text_inputs  = data[2]
        self.targets      = data[3]
        self.inputs, self.max_len = self._handle_data(data[0], train_len)

        self.k = link_k
        self.item_to_image = item_to_image
        self.item_to_title = item_to_title

    def _handle_data(self, inputData, train_len=None):
        """원본과 동일"""
        len_data = [len(d) for d in inputData]
        if train_len is None:
            max_len = max(len_data)
        else:
            max_len = train_len

        us_pois = []
        for upois, le in zip(inputData, len_data):
            _ = list(upois) if le < max_len else list(upois[:max_len])
            us_pois.append(_)
        return us_pois, max_len

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        """
        원본 __getitem__과 동일한 그래프 구성 로직
        u_input      : 아이템 ID 시퀀스
        image_input  : [[img_node_id], [img_node_id], ...] (각 아이템당 1개)
        text_input   : [[title_node_id], [title_node_id], ...] (각 아이템당 1개)
        """
        u_input      = self.inputs[index]
        image_input  = self.image_inputs[index]
        text_input   = self.text_inputs[index]
        target       = self.targets[index]

        le = len(u_input)

        # ── 노드 집합 구성 (원본과 동일) ─────────────────────────────────
        u_nodes = np.unique(u_input).tolist()
        i_nodes = np.unique([y for x in image_input for y in x]).tolist()
        t_nodes = np.unique([y for x in text_input  for y in x]).tolist()
        nodes   = u_nodes + i_nodes + t_nodes
        nodes   = np.asarray(nodes + (self.max_len - len(nodes)) * [0])

        u_node_num = len(u_nodes)
        i_node_num = len(i_nodes)
        t_node_num = len(t_nodes)
        node_type_mask = [1]*u_node_num + [2]*i_node_num + [3]*t_node_num
        node_type_mask = node_type_mask + (self.max_len - len(node_type_mask)) * [0]

        # ── 인접 행렬 구성 (원본과 동일) ─────────────────────────────────
        adj = np.zeros((self.max_len, self.max_len))

        for i in np.arange(le):
            item         = u_input[i]
            item_idx     = np.where(nodes == item)[0][0]
            adj[item_idx][item_idx] = 1  # self-loop

            image_bundle = image_input[i]
            for img in image_bundle:
                img_idx = np.where(nodes == img)[0][0]
                adj[img_idx][img_idx]   = 1
                adj[item_idx][img_idx]  = 5
                adj[img_idx][item_idx]  = 6

            text_bundle = text_input[i]
            for txt in text_bundle:
                txt_idx = np.where(nodes == txt)[0][0]
                adj[txt_idx][txt_idx]   = 1
                adj[item_idx][txt_idx]  = 7
                adj[txt_idx][item_idx]  = 8

            for img in image_bundle:
                for txt in text_bundle:
                    img_idx = np.where(nodes == img)[0][0]
                    txt_idx = np.where(nodes == txt)[0][0]
                    adj[img_idx][txt_idx] = 9
                    adj[txt_idx][img_idx] = 10

        for i in np.arange(le - 1):
            prev_item = u_input[i]
            next_item = u_input[i + 1]
            u = np.where(nodes == prev_item)[0][0]
            v = np.where(nodes == next_item)[0][0]
            if u == v or adj[u][v] == 4:
                pass
            elif adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

            prev_image_bundle = image_input[i]
            next_image_bundle = image_input[i + 1]
            for prev_img in prev_image_bundle:
                for next_img in next_image_bundle:
                    u = np.where(nodes == prev_img)[0][0]
                    v = np.where(nodes == next_img)[0][0]
                    if u == v or adj[u][v] == 4:
                        continue
                    if adj[v][u] == 2:
                        adj[u][v] = 4
                        adj[v][u] = 4
                    else:
                        adj[u][v] = 2
                        adj[v][u] = 3

            prev_text_bundle = text_input[i]
            next_text_bundle = text_input[i + 1]
            for prev_txt in prev_text_bundle:
                for next_txt in next_text_bundle:
                    u = np.where(nodes == prev_txt)[0][0]
                    v = np.where(nodes == next_txt)[0][0]
                    if u == v or adj[u][v] == 4:
                        continue
                    if adj[v][u] == 2:
                        adj[u][v] = 4
                        adj[v][u] = 4
                    else:
                        adj[u][v] = 2
                        adj[v][u] = 3

        # ── alias 인덱스 (원본과 동일) ────────────────────────────────────
        alias_inputs = []
        for item in u_input:
            item_idx = np.where(nodes == item)[0][0]
            alias_inputs.append(item_idx)

        alias_img_inputs = [[0] * self.k for _ in range(self.max_len)]
        for i, img_bundle in enumerate(image_input):
            for j, img in enumerate(img_bundle):
                img_idx = np.where(nodes == img)[0][0]
                alias_img_inputs[i][j] = img_idx

        alias_txt_inputs = [[0] * self.k for _ in range(self.max_len)]
        for i, txt_bundle in enumerate(text_input):
            for j, txt in enumerate(txt_bundle):
                txt_idx = np.where(nodes == txt)[0][0]
                alias_txt_inputs[i][j] = txt_idx

        alias_inputs = alias_inputs + [0] * (self.max_len - le)
        u_input      = u_input      + [0] * (self.max_len - le)
        us_msks      = [1] * le + [0] * (self.max_len - le) if le < self.max_len else [1] * self.max_len

        # ── node_pos_matrix (원본과 동일) ─────────────────────────────────
        node_pos_matrix = np.zeros((self.max_len, self.max_len))
        n_idx = 0
        for item in u_nodes:
            pos_idx = [p for p, x in enumerate(u_input) if item == x]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1
        for image in i_nodes:
            pos_idx = [p for p, sublist in enumerate(image_input) if image in sublist]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1
        for text in t_nodes:
            pos_idx = [p for p, sublist in enumerate(text_input) if text in sublist]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1

        return [torch.tensor(adj),             torch.tensor(nodes),
                torch.tensor(node_type_mask),  torch.tensor(node_pos_matrix),
                torch.tensor(us_msks),         torch.tensor(target),
                torch.tensor(u_input),         torch.tensor(alias_inputs),
                torch.tensor(alias_img_inputs),torch.tensor(alias_txt_inputs)]
