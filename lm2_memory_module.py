import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast


class LM2MemoryModule_Explan(nn.Module):
    """
    논문 2.1, 2.2절 구조.
    - 메모리 모듈 shape: (N, d, d)
      각 슬롯 M_r = I_{dxd}로 초기화
    - time_step마다 cross_attention & gate_update
    """
    def __init__(self, 
                 d_model: int, 
                 num_slots: int,
                 memory_rank: int,
                 lambda_pram: float=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.lambda_pram = lambda_pram

        # Memory 파라미터(슬롯), 논문에서 "M_r = I_{d×d}"로 초기화.
        mem_init = torch.eye(d_model).unsqueeze(0).repeat(num_slots,1,1) # eye(d_model) = (d, d)
        self.memory = nn.Parameter(mem_init, requires_grad=True) # 학습 가능하게 등록
        
        # 크로스 어텐션 Q, K, V 투영 행렬 <- eq.(1)
        self.W_Q = nn.Linear(d_model, d_model)
        # 메모리 (d,d)를 flatten하여 (d*d) -> d 로 매핑
        self.W_K = nn.Linear(d_model*d_model, d_model)
        self.W_V = nn.Linear(d_model*d_model, d_model)

        # 게이트 파라미터터 <- eq.(2)
        self.W_out = nn.Linear(d_model, d_model)
        # 게이트 파라미터터 <- eq.(4), (5)
        #     - (d->d) 만 사용 => 게이트가 (B, d)
        #       => broadcast => (B, N, d, d)
        self.W_forget = nn.Linear(d_model, d_model)
        self.W_in = nn.Linear(d_model, d_model)

    def forward(self, 
                E_t: torch.Tensor, 
                M_t: torch.Tensor=None):
        """
        E_t: (Batch_size, Seq_len, d_model) => time-step loop로 S개 토큰 각각 처리
        M_t: (Batch_size, Num_slots, d_model, d_model) => 이전 memory 상태
        return:
          E_out: (Batch_size, Seq_len, d_model)
          M_out: (Batch_size, Num_slots, d_model, d_model)
        """
        batch_size, seq_len, d_model = E_t.shape
        B, S, d = batch_size, seq_len, d_model
        N = self.num_slots

        if M_t is None:
            # batch마다 독립된 텐서로 복제 (B, N, d, d)
            M_out = self.memory.unsqueeze(0) \
                .expand(batch_size, N, d_model, d_model) \
                .clone()
        else:
            M_out = M_t.clone()

        E_out = torch.zeros_like(E_t) # (B, S, d)

        # time_step loop (각 토큰마다 메모리 업데이트)
        for t in range(seq_len):
            ### Cross Attention ###
            # 현재 토큰 (batch) => Q
            e_t = E_t[:, t, :] # (B, d)

            Q = self.W_Q(e_t) # (B, d)

            # flatten(M_out) => K, V
            M_flat = M_out.view(batch_size, N, d_model*d_model) # (B, N, d*d)

            K_ = self.W_K(M_flat) # (B, N, d)
            V_ = self.W_V(M_flat) # (B, N, d)

            # Q => (B, d) => reshape => (B,1,d) for batch matmul
            Q_3d = Q.unsqueeze(1) # (B, 1, d)

            # attn_score => (B, 1, N)
            # attn_score = (torch.bmm(Q_3d, K_.transpose(2, 1)) 
            #               / (d_model**0.5))
            attn_score = (torch.bmm(Q_3d, K_.permute(0, 2, 1)) 
                          / (d_model**0.5))
            attn_probs = F.softmax(attn_score, dim=-1) # (B, 1, N)
            # # NaN 체크
            # if torch.isnan(attn_probs).any():
            #     print("경고: NaN detected in attention probabilities!")

            # E_mem_3d = attn_probs * V_ : (B, 1, d) <- resultant attention output
            E_mem_3d = torch.bmm(attn_probs, V_) # 배치 행렬 곱
            E_mem = E_mem_3d.squeeze(1) # (B, d)

            ### Output Gate ###
            g_out = torch.sigmoid(self.W_out(E_mem)) # (B, d)
            # e_mem_gated = g_out * (self.lambda_pram * M_out 
            #                        + (1-self.lambda_pram) * E_mem)
            e_mem_gated = g_out * E_mem
            e_t_new = e_t + e_mem_gated  # skip connection

            ### Memory Update ###
            #   g_in       = sigma(e_t * W_in)      => (B, d)
            #   g_forget   = sigma(E_mem * W_forget)=> (B, d)
            #   new_info   = tanh(E_mem)            => (B, d)
            #
            #   -> reshape => (B,1,d,1) => expand => (B,N,d,d)
            #   -> M_{t+1} = g_in * new_info + g_forget * M_t

            g_in_vec = torch.sigmoid(self.W_in(e_t))     # (B, d)
            g_forget_vec = torch.sigmoid(self.W_forget(E_mem))  # (B, d)
            new_info_vec = torch.tanh(E_mem)             # (B, d)

            # reshape & expand
            g_in_4d = g_in_vec.view(B, 1, d, 1).expand(B, N, d, d)
            g_forget_4d = g_forget_vec.view(B, 1, d, 1).expand(B, N, d, d)
            new_info_4d = new_info_vec.view(B, 1, d, 1).expand(B, N, d, d)

            # 최종 업데이트
            M_out = g_in_4d * new_info_4d + g_forget_4d * M_out

            E_out[:, t, :] = e_t_new

        return E_out, M_out
    

class LM2MemoryModule(nn.Module):
    def __init__(self,
                 d_model: int, 
                 num_slots: int,
                 memory_rank: int):
        super().__init__()
        self.num_slots = num_slots
        self.memory_rank = memory_rank if memory_rank else d_model

        if memory_rank: # 저랭크 근사 U @ V^T
            U = torch.randn(num_slots, d_model, memory_rank) * 0.02
            V = torch.randn(num_slots, memory_rank, d_model) * 0.02
            # nn.init.xavier_uniform_(U)
            # nn.init.xavier_uniform_(V)
            self.U = nn.Parameter(U, requires_grad=True)
            self.V = nn.Parameter(V, requires_grad=True)
        else: # 단위 행렬 초기화
            mem_init = torch.eye(d_model).unsqueeze(0).repeat(num_slots,1,1)
            self.memory = nn.Parameter(mem_init, requires_grad=True)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model*d_model, d_model)
        self.W_V = nn.Linear(d_model*d_model, d_model)

        self.W_out = nn.Linear(d_model, d_model)
        self.W_forget = nn.Linear(d_model, d_model)
        self.W_in = nn.Linear(d_model, d_model)

    def forward(self,
                E_t: torch.Tensor, 
                M_t: torch.Tensor=None):
        
        B, S, d = E_t.shape
        N = self.num_slots
        # r = self.memory_rank
        r = d
        
        if M_t is None:
            M_out = torch.bmm(self.U, self.V).unsqueeze(0) \
                .expand(B, N, d, r) \
                .clone()
        else:
            M_out = M_t.clone()
        
        Q = self.W_Q(E_t) # (B, S, d)
        M_flat = M_out.view(B, N, d * r) # (B, N, d*r)
        K_ = self.W_K(M_flat) # (B, N, d)
        V_ = self.W_V(M_flat) # (B, N, d)
        
        # E_mem = checkpoint(self.compute_attention, Q, K_, V_, d) # (B, S, d)
        E_mem = self.compute_attention(Q, K_, V_, d) # (B, S, d)

        ### Output Gate ###
        g_out = torch.sigmoid(self.W_out(E_mem)) # (B, S, d)
        E_mem_gated = g_out * E_mem
        E_out = E_t + E_mem_gated

        ### Memory Update ###
        g_in_vec = torch.sigmoid(self.W_in(E_t))
        new_info_vec = torch.tanh(E_mem)
        g_forget_vec = torch.sigmoid(self.W_forget(E_mem))
        
        M_out = self.update_memory(g_in_vec, new_info_vec, g_forget_vec, M_out)

        return E_out, M_out

        
    def compute_attention(self, Q, K, V, d):
        attn_score = torch.bmm(Q, K.transpose(2, 1)) / (d ** 0.5) # (B, S, N)
        # attn_score = torch.bmm(Q, K.permute(0, 2, 1)) / (d ** 0.5)
        attn_probs = F.softmax(attn_score, dim=-1) # (B, S, N)
        # # Nan 체크
        # if torch.isnan(attn_probs).any():
        #     print("경고: NaN detected in attention probabilities!")

        return torch.bmm(attn_probs, V) # (B, S, d)
        
    def update_memory(self, g_in, new_info, g_forget, M_out): # (B,S,d) * (B,N,d,r)
        # 차원을 어떻게 맞춰서 계산했는지 정말 모르겠음. 
        # 논문에도 안 나오고 저자도 코드 올려준다고 하고 잠수 탐.
        return g_in * new_info + g_forget * M_out#