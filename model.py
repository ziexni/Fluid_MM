import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from aggregator import RGATLayer, GCNLayer, SAGELayer, GATLayer, KVAttentionLayer, HeteAttenLayer


class VLGraph(Module):
    """
    VLGraph — Sequential Recommendation 버전
    원래 세션 기반 구조를 유지하되, 세션 = user_train 시퀀스로 대체
    노드 타입: item(1) / image(2) / title(3)
    """
    def __init__(self, config, image_feat, title_feat):
        """
        image_feat : np.ndarray (num_items+1, img_dim)  — 1-based, 0은 padding
        title_feat : np.ndarray (num_items+1, title_dim) — 1-based, 0은 padding
        """
        super(VLGraph, self).__init__()
        self.config      = config
        self.batch_size  = config['batch_size']
        self.num_items   = config['num_items']   # 실제 아이템 수 (1-based max)
        self.dim         = config['embedding_size']
        self.n_layer     = config['n_layer']
        self.aggregator  = config['aggregator']
        self.max_relid   = config['max_relid']   # 10 (엣지 타입 수)
        self.max_seq_len = config['max_seq_len']
        self.max_node_len = self.max_seq_len * 3  # item + image + title

        self.dropout_local = config['dropout_local']
        self.dropout_atten = config['dropout_atten']

        # ── Aggregator 선택 ───────────────────────────────────────────────
        if self.aggregator == 'rgat':
            self.local_agg = RGATLayer(self.dim, self.max_relid, config['alpha'], dropout=self.dropout_atten)
        elif self.aggregator == 'hete_attention':
            self.local_agg = HeteAttenLayer(config, self.dim, self.max_relid, alpha=0.1, dropout=self.dropout_atten)
        elif self.aggregator == 'kv_attention':
            self.local_agg = KVAttentionLayer(self.dim, self.max_relid, alpha=0.1, dropout=self.dropout_atten)
        elif self.aggregator == 'gcn':
            self.local_agg = GCNLayer(self.dim, self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)
        elif self.aggregator == 'graphsage':
            self.local_agg = SAGELayer(self.dim, self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)
        elif self.aggregator == 'gat':
            self.local_agg = GATLayer(self.dim, self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)

        # ── 노드 임베딩 ───────────────────────────────────────────────────
        # 노드 ID 공간: item(1~num_items) / image(num_items+1~2*num_items) / title(2*num_items+1~3*num_items)
        self.embedding = nn.Embedding(self.num_items * 3 + 1, self.dim, padding_idx=0)

        # ── 위치/타입 임베딩 ──────────────────────────────────────────────
        self.pos_embedding       = nn.Embedding(200, self.dim)
        self.node_type_embedding = nn.Embedding(4,   self.dim)  # 0(pad)/1(item)/2(image)/3(title)

        # ── 시퀀스 표현 파라미터 (원래 VLGraph 그대로) ───────────────────
        self.w_1       = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2       = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_pos_type = nn.Parameter(torch.Tensor(3 * self.dim, self.dim))  # node+type+pos concat

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        # ── 모달리티 융합 ─────────────────────────────────────────────────
        # item + image + title 표현을 합쳐서 최종 유저 표현 생성
        self.fusion_layer = nn.Linear(self.dim * 3, self.dim)

        # ── 피처 프로젝션 레이어 ──────────────────────────────────────────
        # 사전학습 피처를 임베딩 차원으로 변환
        self.image_proj = nn.Linear(image_feat.shape[1], self.dim)
        self.title_proj = nn.Linear(title_feat.shape[1], self.dim)

        # ── 손실 함수 ─────────────────────────────────────────────────────
        # BPR Loss (베이스라인 프로토콜: 정답 vs negative)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config['lr_dc_step'], gamma=config['lr_dc']
        )

        self.reset_parameters()

        # ── 사전학습 피처로 임베딩 초기화 ────────────────────────────────
        # image 노드: 인덱스 num_items+1 ~ 2*num_items
        # title 노드: 인덱스 2*num_items+1 ~ 3*num_items
        with torch.no_grad():
            img_projected   = self.image_proj(torch.tensor(image_feat[1:], dtype=torch.float))
            title_projected = self.title_proj(torch.tensor(title_feat[1:], dtype=torch.float))
            self.embedding.weight[self.num_items + 1: 2 * self.num_items + 1].copy_(img_projected)
            self.embedding.weight[2 * self.num_items + 1: 3 * self.num_items + 1].copy_(title_projected)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, nodes, node_type_mask, node_pos_matrix, stage='train'):
        """
        이종 그래프 위에서 노드 표현 계산
        adj             : (B, max_node_len, max_node_len) 엣지 타입 행렬
        nodes           : (B, max_node_len) 노드 ID
        node_type_mask  : (B, max_node_len) 노드 타입
        node_pos_matrix : (B, max_node_len, max_seq_len) 위치 행렬
        """
        # 노드 임베딩 조회
        h_nodes = self.embedding(nodes)  # (B, max_node_len, dim)

        # 노드 타입 임베딩 추가
        node_type_emb = self.node_type_embedding(node_type_mask)  # (B, max_node_len, dim)

        # 위치 임베딩 추가 (node_pos_matrix로 평균 풀링)
        L       = node_pos_matrix.shape[-1]
        pos_emb = self.pos_embedding.weight[:L]                    # (L, dim)
        pos_emb = torch.matmul(node_pos_matrix, pos_emb)          # (B, max_node_len, dim)
        pos_num = node_pos_matrix.sum(dim=-1, keepdim=True)
        pos_emb = pos_emb / (pos_num + 1e-9)                      # 평균
        pos_emb = pos_emb * torch.clamp(node_type_mask, max=1).unsqueeze(-1)  # padding 마스킹

        # node + type + pos concat → 선형 변환
        h_nodes = torch.cat([h_nodes, node_type_emb, pos_emb], dim=-1)  # (B, max_node_len, dim*3)
        h_nodes = torch.matmul(h_nodes, self.w_pos_type)                # (B, max_node_len, dim)

        # ── GNN 레이어 반복 ───────────────────────────────────────────────
        for _ in range(self.n_layer):
            h_nodes = self.local_agg(h_nodes, adj, node_type_mask, stage)
            h_nodes = F.dropout(h_nodes, self.dropout_local, training=self.training)
            h_nodes = h_nodes * torch.clamp(node_type_mask, max=1).unsqueeze(-1)  # padding 마스킹

        return h_nodes  # (B, max_node_len, dim)

    def get_sequence_representation(self, seq_hiddens, mask):
        """
        시퀀스 표현 계산 — attention pooling (원래 VLGraph 방식)
        seq_hiddens : (B, L, dim)
        mask        : (B, L)  1=유효, 0=padding
        """
        batch_size = seq_hiddens.shape[0]
        L          = seq_hiddens.shape[1]
        mask_float = mask.float().unsqueeze(-1)  # (B, L, 1)

        # 위치 임베딩
        pos_emb = self.pos_embedding.weight[:L].unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, dim)

        # 전체 시퀀스 평균 (global context)
        hs = torch.sum(seq_hiddens * mask_float, dim=1) / torch.sum(mask_float, dim=1)  # (B, dim)
        hs = hs.unsqueeze(1).expand(-1, L, -1)  # (B, L, dim)

        # attention 가중치 계산
        nh   = torch.matmul(torch.cat([pos_emb, seq_hiddens], dim=-1), self.w_1)  # (B, L, dim)
        nh   = torch.tanh(nh)
        nh   = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)   # (B, L, 1)
        beta = beta * mask_float             # padding 위치 0
        hiddens = torch.sum(beta * seq_hiddens, dim=1)  # (B, dim)
        return hiddens

    def compute_scores(self, node_hiddens, alias_inputs, us_msks):
        """
        유저 표현 계산 (item/image/title 각각 → fusion)
        node_hiddens  : (B, max_node_len, dim)
        alias_inputs  : (B, max_seq_len) 시퀀스 위치별 노드 행렬 인덱스
        us_msks       : (B, max_seq_len) 시퀀스 마스크
        Returns:
            user_repr : (B, dim)
        """
        B   = node_hiddens.shape[0]
        L   = alias_inputs.shape[1]
        dim = node_hiddens.shape[2]

        # ── item 시퀀스 표현 ──────────────────────────────────────────────
        alias_exp    = alias_inputs.unsqueeze(-1).expand(B, L, dim)  # (B, L, dim)
        item_hiddens = node_hiddens.gather(1, alias_exp)              # (B, L, dim)
        item_repr    = self.get_sequence_representation(item_hiddens, us_msks)  # (B, dim)

        # ── image 시퀀스 표현 ─────────────────────────────────────────────
        # image 노드 인덱스 = item 노드 인덱스 + max_seq_len (노드 행렬 내 offset)
        img_alias    = (alias_inputs + self.max_seq_len).clamp(max=node_hiddens.shape[1] - 1)
        img_alias_exp = img_alias.unsqueeze(-1).expand(B, L, dim)
        img_hiddens  = node_hiddens.gather(1, img_alias_exp)
        img_repr     = self.get_sequence_representation(img_hiddens, us_msks)   # (B, dim)

        # ── title 시퀀스 표현 ─────────────────────────────────────────────
        # title 노드 인덱스 = item 노드 인덱스 + 2*max_seq_len
        txt_alias    = (alias_inputs + 2 * self.max_seq_len).clamp(max=node_hiddens.shape[1] - 1)
        txt_alias_exp = txt_alias.unsqueeze(-1).expand(B, L, dim)
        txt_hiddens  = node_hiddens.gather(1, txt_alias_exp)
        txt_repr     = self.get_sequence_representation(txt_hiddens, us_msks)   # (B, dim)

        # ── 모달리티 융합 ─────────────────────────────────────────────────
        user_repr = self.fusion_layer(
            torch.cat([item_repr, img_repr, txt_repr], dim=-1)
        )  # (B, dim)
        return user_repr

    def compute_loss(self, batch):
        """
        BPR Loss 계산 (정답 아이템 score > negative 아이템 score)
        """
        adj             = batch['adj'].cuda().float()
        nodes           = batch['nodes'].cuda()
        node_type_mask  = batch['node_type_mask'].cuda()
        node_pos_matrix = batch['node_pos_matrix'].cuda()
        alias_inputs    = batch['alias_inputs'].cuda()
        us_msks         = batch['us_msks'].cuda()
        target          = batch['target'].cuda()    # (B,) 1-based
        negative        = batch['negative'].cuda()  # (B,) 1-based

        node_hiddens = self.forward(adj, nodes, node_type_mask, node_pos_matrix, stage='train')
        user_repr    = self.compute_scores(node_hiddens, alias_inputs, us_msks)  # (B, dim)

        # 아이템 임베딩: item 노드는 1-based 그대로 embedding 조회
        pos_emb = self.embedding(target)    # (B, dim)
        neg_emb = self.embedding(negative)  # (B, dim)

        pos_score = (user_repr * pos_emb).sum(dim=-1)  # (B,)
        neg_score = (user_repr * neg_emb).sum(dim=-1)  # (B,)

        # BPR Loss
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        return loss

    def get_user_repr(self, batch):
        """평가 시 유저 표현 반환"""
        adj             = batch['adj'].cuda().float()
        nodes           = batch['nodes'].cuda()
        node_type_mask  = batch['node_type_mask'].cuda()
        node_pos_matrix = batch['node_pos_matrix'].cuda()
        alias_inputs    = batch['alias_inputs'].cuda()
        us_msks         = batch['us_msks'].cuda()

        with torch.no_grad():
            node_hiddens = self.forward(adj, nodes, node_type_mask, node_pos_matrix, stage='test')
            user_repr    = self.compute_scores(node_hiddens, alias_inputs, us_msks)
        return user_repr  # (B, dim)
