"""正确实现的增强模块：backbone 后插 1 个（transition 位置）+ 冻结前层。

参照 F:\\archive\\chest_xray8_models\\models\\attention_enhanced.py 的正确实现：
  backbone → [单个注意力模块 SE/CBAM/BAM] → projection → pooling → classifier
（此前 tb_9class_multi_benchmark.py 的错误实现是"每 stage 插 4 个"，已废弃）

冻结策略（统一，符合申请书"冻结前70%" + Wang[2] 借鉴）：
  冻结 conv1/bn1/relu/maxpool/layer1/layer2，微调 layer3/layer4 + 增强模块 + 分类头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

MODEL_TYPES = ["baseline", "se", "cbam", "bam", "gcn", "gat"]


# ======================================================================
# 注意力模块（SE / CBAM / BAM，标准实现）
# ======================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class BAMBlock(nn.Module):
    """标准 Bottleneck Attention Module：分支内 BN，最后 Mc+Ms 相加统一 sigmoid，输出 F + F*M"""
    def __init__(self, channels, reduction=16, dilation=4):
        super().__init__()
        r = channels // reduction
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, r, 1, bias=False),
            nn.BatchNorm2d(r),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.BatchNorm2d(channels))
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.BatchNorm2d(r),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, r, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(r),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, r, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(r),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, 1, 1, bias=False),
            nn.BatchNorm2d(1))

    def forward(self, x):
        Mc = self.channel_att(x)
        Ms = self.spatial_att(x)
        M = torch.sigmoid(Mc + Ms)
        return x + x * M


# ======================================================================
# 图模块（GCN / GAT，layer4 特征图 → 网格节点 → 图卷积/图注意力 → mean pool）
# ======================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.bmm(adj, x)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return F.relu(x)


class GCNBlock(nn.Module):
    def __init__(self, in_channels, num_nodes=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_proj = nn.Conv2d(in_channels, 512, 1)
        self.gcn1 = GraphConvLayer(512, 512)
        self.gcn2 = GraphConvLayer(512, 512)

    def _grid_adj(self, N, device):
        side = self.num_nodes
        A = torch.zeros(N, N, device=device)
        for i in range(N):
            A[i, i] = 1.0
            r, c = i // side, i % side
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    A[i, nr * side + nc] = 1.0
        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(D.pow(-0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.node_proj(x)
        x = F.adaptive_avg_pool2d(x, (self.num_nodes, self.num_nodes))
        N = self.num_nodes * self.num_nodes
        x = x.view(B, 512, N).transpose(1, 2)
        adj = self._grid_adj(N, x.device).unsqueeze(0).expand(B, -1, -1)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x.mean(dim=1)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def _prepare(self, h):
        B, N, C = h.size()
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        return torch.cat([h_i, h_j], dim=-1)

    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.W(x)
        e = self.leakyrelu(self.a(self._prepare(h)).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        h_prime = torch.bmm(attention, h)
        return F.elu(h_prime)


class GATBlock(nn.Module):
    def __init__(self, in_channels, num_nodes=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_proj = nn.Conv2d(in_channels, 512, 1)
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(512, 512 // num_heads, dropout=dropout)
            for _ in range(num_heads)])

    def _grid_adj(self, N, device):
        side = self.num_nodes
        A = torch.zeros(N, N, device=device)
        for i in range(N):
            A[i, i] = 1.0
            r, c = i // side, i % side
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    A[i, nr * side + nc] = 1.0
        return A

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.node_proj(x)
        x = F.adaptive_avg_pool2d(x, (self.num_nodes, self.num_nodes))
        N = self.num_nodes * self.num_nodes
        x = x.view(B, 512, N).transpose(1, 2)
        adj = self._grid_adj(N, x.device).unsqueeze(0).expand(B, -1, -1)
        head_outputs = [att(x, adj) for att in self.attentions]
        x = torch.cat(head_outputs, dim=-1)
        return x.mean(dim=1)


# ======================================================================
# 统一模型：ResNet-50 骨架 + 可选增强模块（backbone 后插 1 个）+ 冻结前层
# ======================================================================
class Chest9Classifier(nn.Module):
    def __init__(self, model_type="baseline", num_classes=9, feature_depth=512):
        super().__init__()
        assert model_type in MODEL_TYPES, f"未知 model_type: {model_type}"
        self.model_type = model_type

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                    backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 增强模块：backbone 后插 1 个（transition 位置），非"每 stage 插"
        if model_type == "se":
            self.attention = SEBlock(2048)
        elif model_type == "cbam":
            self.attention = CBAMBlock(2048)
        elif model_type == "bam":
            self.attention = BAMBlock(2048)
        elif model_type == "gcn":
            self.gnn = GCNBlock(2048, num_nodes=8)
        elif model_type == "gat":
            self.gnn = GATBlock(2048, num_nodes=8, num_heads=4)

        head_in = 512 if model_type in ("gcn", "gat") else 2048
        self.projection = nn.Sequential(
            nn.Linear(head_in, feature_depth),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.classifier = nn.Linear(feature_depth, num_classes)

        # 冻结前层：conv1/bn1/relu/maxpool/layer1/layer2
        self._freeze_layers()

    def _freeze_layers(self):
        for p in self.layer1.parameters():   # 含 conv1/bn1/relu/maxpool/layer1
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, 7, 7)

        if self.model_type == "se":
            x = self.attention(x)
        elif self.model_type == "cbam":
            x = self.attention(x)
        elif self.model_type == "bam":
            x = self.attention(x)
        elif self.model_type == "gcn":
            x = self.gnn(x)          # (B, 512)
            x = self.projection(x)
            return self.classifier(x)
        elif self.model_type == "gat":
            x = self.gnn(x)          # (B, 512)
            x = self.projection(x)
            return self.classifier(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)    # (B, 2048)
        x = self.projection(x)       # (B, 512)
        return self.classifier(x)


if __name__ == "__main__":
    # 快速自检：6 模型前向 + 冻结状态
    for mt in MODEL_TYPES:
        m = Chest9Classifier(model_type=mt, num_classes=9)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        assert out.shape == (2, 9), f"{mt} 输出形状错误: {out.shape}"
        # 检查冻结：layer1/layer2 无 requires_grad，layer3/layer4 有
        frozen = all(not p.requires_grad for p in m.layer1.parameters()) and \
                 all(not p.requires_grad for p in m.layer2.parameters())
        unfrozen = any(p.requires_grad for p in m.layer3.parameters()) and \
                   any(p.requires_grad for p in m.layer4.parameters())
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in m.parameters())
        print(f"{mt}: out={tuple(out.shape)} frozen={frozen} unfrozen={unfrozen} "
              f"trainable={n_train:,}/{n_total:,}")
    print("全部模型自检通过")
