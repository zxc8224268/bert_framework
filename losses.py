import torch
from torch import nn, Tensor
import torch.nn.functional as F

# -------------------------------------------------------
#   Triplet distance metric
# -------------------------------------------------------
class TripletDistanceMetric():
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

# -------------------------------------------------------
#   Triplet loss (hard margin)
# -------------------------------------------------------
class TripletLoss(nn.Module):
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN, margin=5):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, emb_anchor, emb_pos, emb_neg):
        distance_pos = self.distance_metric(emb_anchor, emb_pos)
        distance_neg = self.distance_metric(emb_anchor, emb_neg)
        losses = F.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()

# -------------------------------------------------------
#   Triplet loss (soft margin)
# -------------------------------------------------------
class SoftMarginTripletLoss(nn.Module):
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN):
        super(SoftMarginTripletLoss, self).__init__()
        self.distance_metric = distance_metric

    def forward(self, emb_anchor, emb_pos, emb_neg):
        distance_pos = self.distance_metric(emb_anchor, emb_pos)
        distance_neg = self.distance_metric(emb_anchor, emb_neg)
        tl = torch.log1p(torch.exp(distance_pos - distance_neg))
        soft_margin_triplet_loss = tl.mean()
        return soft_margin_triplet_loss