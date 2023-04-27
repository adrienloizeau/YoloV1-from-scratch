import torch
import torch.nn as nn
from src.utils import intersection_over_union
from src import config


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_noobj= 0.5
        self.lambda_coord= 5
        self.C = config.C
        self.B = config.B
        self.S = config.S

    def forward(self, predictions, target):

        #? 
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B *5)

        # Questions :
        # pq que deux boxs ?
        # C'est quoi C déjà ?
        # C'est quoi exits box  ?
        # C'est quoi best box  ?

        iou_b1 = intersection_over_union(predictions[...,21:25], target[...,21:25])
        iou_b2 = intersection_over_union(predictions[...,25:30], target[...,26:30])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim =0)
        ious_maxes, bestbox = torch.max(ious, dim = 0) 
        exists_box = target[...,20].unsqueeze(3) # identity of object I 

        # ================== #
        #     BOX COORDS     #
        # ================== #
        box_predictions = exists_box * (
            (
                bestbox * predictions[...,26:30]
                +(1 - bestbox) * predictions[...,21:25]
            )
         )
        box_targets = exists_box* target[...,21:25]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(
            torch.abs(box_predictions[..., 2:4]+ 1e-6)
        )

        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # (N, S, S, 25) -> (N*S*S,4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # ================== #
        #     OBJECT LOSS    #
        # ================== #
        pred_box = (
            bestbox * predictions[..., 25:26]+ (1-bestbox) * predictions[...,20:21]
        )
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21])
        )

        # ================== #
        #  NO OBJECT LOSS    #
        # ================== #
        # (N,S,S,1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box)* predictions[...,20:21], start_dim= 1),
            torch.flatten((1 - exists_box)* target[...,20:21], start_dim= 1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box)* predictions[...,25:26], start_dim= 1),
            torch.flatten((1 - exists_box)* target[...,20:21], start_dim= 1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[...,:20], end_dim = -2),
            torch.flatten(exists_box * target[...,:20], end_dim = -2)
        )

        loss = (
            self.lambda_coord * box_loss # First two rows of the loss in the paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss 