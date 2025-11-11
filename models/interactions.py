import torch
from torch import nn

class FeedForwardDataInteraction(nn.Module):
    def __init__(self, backbone, d_input, use_output_proj=True):
        super(FeedForwardDataInteraction, self).__init__()
        self.backbone = backbone
        self.d_input = d_input
        self.use_output_proj = use_output_proj
        self.output_proj = nn.LazyLinear(self.d_input, bias=False) if use_output_proj else nn.Identity()

    def forward(self, x, hidden_state, aux_inputs=None):
        assert aux_inputs is None, f"aux_inputs not supported for FeedForwardDataInteraction"
        features = self.backbone(x).flatten(1)
        output = self.output_proj(features)
        return output, None

class CIFARFSDataInteraction(FeedForwardDataInteraction):
    def __init__(self, backbone, d_input):
        super().__init__(backbone, d_input, use_output_proj=True)

    def forward(self, x, hidden_state=None, aux_inputs=None):
        assert aux_inputs is not None, f"aux_inputs must be provided for CIFARFSDataInteraction"
        
        features = self.backbone(x).flatten(1)
        features = torch.cat([features, aux_inputs], dim=-1)
        output = self.output_proj(features)

        return output, None