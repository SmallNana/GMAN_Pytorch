import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda')


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu, bn_decay=None):
        """
        卷积
        :param input_dims: 输入维度
        :param output_dims: 输出维度
        :param kernel_size: 卷积盒大小
        :param stride:步长
        :param padding:pad小大
        :param use_bias:是否使用偏置值
        :param activation:激活函数
        :param bn_decay:标准化过程中的稳定系数
        """
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]

        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_normal_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        :param x:  shape [num_samples, hid_squence, num_nodes, C]，[N,W,H,C]
        :return:
        """
        x = x.permute(0, 3, 2, 1).to(device)  # [N,C,H,W]
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        # 前两个是对W进行pad(左边填充数， 右边填充数)，后两个是对H进行pad(上边填充数，下边填充数)

        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)

        return x.permute(0, 3, 2, 1)  # [N,W,H,C]


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        """
        全连接层
        :param input_dims: 输入维度
        :param units: 输出维度
        :param activations: 激活函数
        :param bn_decay: 标准化过程中的稳定系数
        :param use_bias: 是否使用偏置值
        """
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)

        assert type(units) == list

        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation, bn_decay=bn_decay)
            for input_dim, num_unit, activation in zip(input_dims, units, activations)])

    def forward(self, x):
        # [num_samples, hid_squence, num_nodes, C]，[N,W,H,C]
        for conv in self.convs:
            x = conv(x)

        return x


class STEmbedding(nn.Module):
    """时空间嵌入"""
    def __init__(self, D, bn_decay):
        """
        时空嵌入
        :param D:输出维度 K * d
        :param bn_decay: 标准化过程中的稳定系数
        """
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):
        """
        过程都是过2层FC
        :param SE: 由node2vec得到的空间嵌入 [num_vertex, D]  [H, D]
        :param TE: [num_samples, num_hid + num_prod, 2]  [N, W ,D]
        :param T: 将一天分为288个5分钟step
        :return:
        """
        # 空间embedding
        SE = SE.unsqueeze(0).unsqueeze(0)  # [1, 1, H, D]
        SE = self.FC_se(SE)  # [1,1,H,D]

        # 时间embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)

        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [num_samples, num_squences, 295] (N, W, C)
        TE = TE.unsqueeze(dim=2)  # [N,W,1,C]
        TE = self.FC_te(TE)

        del dayofweek, timeofday
        return SE + TE  # [N,W,H,C]


class spatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, K, d, bn_decay):
        """
        :param K: Heads数量
        :param d: 单个head的维度
        :param bn_decay: 标准化过程中的稳定系数
        """
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, STE):
        """
        :param X:[batch_size, num_steps, num_nodes, channels]，[N,W,H,C]
        :param STE: [N,W,H,C]
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_steps, num_nodes, 2D]

        query = self.FC_q(X)  # [batch_size, num_steps, num_nodes, K * d]
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # [K * batch_size, num_steps, num_nodes, d]

        attention = torch.matmul(query, key.transpose(2, 3))  # [K*batch_size, num_steps, num_nodes, num_nodes]
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)  # [K*batch_size, num_steps, num_nodes, num_nodes]

        X = torch.matmul(attention, value)  # [K*batch_size, num_steps, num_nodes, d]
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)   # [batch_size, num_steps ,num_nodes, d*K=D]
        X = self.FC(X)
        del query, key, attention

        return X


class temporalAttention(nn.Module):
    """时间注意力"""
    def __init__(self, K, d, bn_decay, mask=True):
        """
        :param K: Heads数量
        :param d: 单个head的维度
        :param bn_decay: 标准化过程中的稳定系数
        :param mask: 是否使用mask，对于时间注意力我们需要mask掉后面的时间step
        """
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, STE):
        """
        :param X: [batch_size, num_steps, num_nodes, channels]，[N,W,H,C]
        :param STE: [N,W,H,C]
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_steps, num_nodes, 2D]

        query = self.FC_q(X)  # [batch_size, num_steps, num_nodes, K * d]
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # [K * batch_size, num_steps, num_nodes, d]

        query = query.permute(0, 2, 1, 3)  # [K * batch_size, num_nodes, num_steps, d]
        key = key.permute(0, 2, 3, 1)  # [K * batch_size, num_nodes, d, num_steps]
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)  # [K * batch_size, num_nodes, num_steps, num_steps]
        attention /= (self.d ** 0.5)

        if self.mask:
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)  # 取下三角矩阵，包括对角线
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # 【1，1，num_step, num_step】
            mask = mask.repeat(self.K * batch_size, num_nodes, 1, 1)
            # [K * batch_size, num_nodes, num_steps, num_nodes]
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)

        # softmax
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)  # [K * batch_size, num_nodes, num_steps, d]
        X = X.permute(0, 2, 1, 3)  # [K * batch_size, num_steps, num_nodes, d]
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # [batch_size, num_steps, num_nodes, d*K]
        X = self.FC(X)
        del query, key, value, attention

        return X


class gatedFusion(nn.Module):
    def __init__(self, D, bn_decay):
        """
        :param D: 输出维度 K * d
        :param bn_decay: 标准化过程中的稳定系数
        """
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=False)

        self.FC_xt = FC(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=True)

        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, HS, HT):
        """
        :param HS: [batch_size, num_steps, num_nodes, channels]，[N,W,H,C]
        :param HT: [batch_size, num_steps, num_nodes, channels]，[N,W,H,C]
        """
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)

        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1-z, HT))
        H = self.FC_h(H)

        del XS, XT, z

        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        """
        :param K: Heads数量
        :param d: 单个head的维度
        :param bn_decay: 标准化过程中的稳定系数
        :param mask: 是否使用mask，对于时间注意力我们需要mask掉后面的时间step
        """
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT= self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)

        del HS, HT

        return torch.add(X, H)


class transformAttention(nn.Module):
    def __init__(self, K, d, bn_decay):
        """
        :param K: Heads数量
        :param d: 单个head的维度
        :param bn_decay: 标准化过程中的稳定系数
        :param mask: 是否使用mask，对于时间注意力我们需要mask掉后面的时间step
        """
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        """
        :param X:[batch_size, num_his, num_nodes, K*d]
        :param STE_his:[batch_size, num_his, num_nodes, K*d]
        :param STE_pred:[batch_size, num_pred, num_nodes, K*d]
        :return:
        """
        batch_size = X.shape[0]

        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  # [K * batch_size, num_pred, num_nodes, d]
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  # [K * batch_size, num_his, num_nodes, d]
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)  # [K * batch_size, num_his, num_nodes, d]

        query = query.permute(0, 2, 1, 3)  # [K * batch_size, num_nodes, num_pred, d]
        key = key.permute(0, 2, 3, 1)  # [K * batch_size, num_nodes, d ,num_his]
        value = value.permute(0, 2, 1, 3)  # [K * batch_size, num_nodes, num_his, d]

        attention = torch.matmul(query, key)  # [K * batch_size, num_nodes, num_pred, num_his]
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)  # [K * batch_size, num_nodes, num_pred, num_his]

        X = torch.matmul(attention, value)  # [K * batch_size, num_nodes, num_pred, d]
        X = X.permute(0, 2, 1, 3)  # [K * batch_size, num_pred, num_nodes, d]
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # [batch_size, num_pred, num_nodes, d * K]
        X = self.FC(X)
        del query, key, value, attention

        return X


class GMAN(nn.Module):
    def __init__(self, SE, args, bn_decay):
        """
        GMAN模型
        :param SE:node2vec得到的图嵌入 [num_nodes, D=K * d]
        :param args.L: STAttention层数
        :param args.K: Head数量
        :param args.d：单个head的维度
        :param args.hum_his：历史时间step长度
        :param bn_decay：标准化过程中的稳定系数
        """
        super(GMAN, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        self.num_his = args.num_his
        self.SE = SE

        self.STEmbedding = STEmbedding(D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, X, TE):
        X = torch.unsqueeze(X, -1).to(device)  # [num_samples, hid_squence, num_nodes, 1]，[N,W,H,C]
        X = self.FC_1(X)  # [num_samples, hid_squence, num_nodes, K * d]，[N,W,H,C]

        # 时空间嵌入
        STE = self.STEmbedding(self.SE, TE)  # [num_samples, hid_squence + pred_squence, num_nodes, K * d]
        STE_his = STE[:, :self.num_his]  # [num_samples, hid_squence, num_nodes, K * d]
        STE_pred = STE[:, self.num_his:]  # [num_samples, pred_squence, num_nodes, K * d]

        # Encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)  # [num_samples, hid_squence, num_nodes, K * d]

        # TransAtt
        X = self.transformAttention(X, STE_his, STE_pred)  # [num_samples, pred_squence, num_nodes, K * d]

        # Decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)  # [num_samples, pred_squence, num_nodes, K * d]

        # Output
        X = self.FC_2(X)  # [num_samples, pred_squence, num_nodes, 1]
        del STE, STE_his, STE_pred

        return torch.squeeze(X, 3)

























