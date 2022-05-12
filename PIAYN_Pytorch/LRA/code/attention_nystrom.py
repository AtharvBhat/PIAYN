
import torch
import torch.nn as nn
import math

class NystromAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.num_landmarks = config["num_landmarks"]
        self.seq_len = config["max_seq_len"]
        
        if "inv_coeff_init_option" in config:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):

        Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        print('Q after masking: ', Q.shape)
        print('K after masking: ', K.shape)

        if self.num_landmarks == self.seq_len:
            print('Shape of Q: ', Q.shape)
            a = K.transpose(-1, -2)
            print('Shape of Other Matrix: ', a.shape)
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            
            print('Q_Landmark without mean: ', Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).shape)
            print('K_landmark without mean: ', K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).shape)
            print('Q_landmarks', Q_landmarks.shape)
            print('K_landmarks', K_landmarks.shape)
            print('K_landmarks.transpose(-1, -2)', K_landmarks.transpose(-1, -2).shape)

            #kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(0, -1)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            #kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(K, Q_landmarks.transpose(0, -1))) # - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            print('kernel_1: ', kernel_1.shape)
            print('self.interactive_inv(kernel_2): ', self.iterative_inv(kernel_2).shape)
            print('kernel_3: ', kernel_3.shape)
            print('V: ', V.shape)

            #X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))
            b = torch.matmul(kernel_1, self.iterative_inv(kernel_2).transpose(0, -2))
            print('b.shape', b.shape)
            c = torch.matmul(kernel_3, V)
            print('c.shape', c.shape)

            X = torch.matmul(b, c)

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])

        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'
