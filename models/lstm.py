import torch
from torch import Tensor, nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, dropout: float = 0, layer_norm: bool = False, T_max: int | None = None):
    super().__init__()
    assert 0 <= dropout < 1, "Dropout should be >= 0 and < 1"
    assert T_max is None or T_max >= 2, "T_max should be >= 2 if set"
    self.hidden_size, self.layer_norm, self.T_max = hidden_size, layer_norm, T_max
    self.h_0, self.c_0 = nn.Parameter(torch.zeros(hidden_size)), nn.Parameter(torch.zeros(hidden_size))  # Learnable initial hidden states
    self.weight_h = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
    self.weight_x = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
    self.bias = nn.Parameter(torch.empty(4 * hidden_size))
    self.dropout = nn.Dropout(p=dropout)  # Recurrent dropout for hidden state updates, from "Recurrent Dropout without Memory Loss"
    if layer_norm:
      self.ln_1 = nn.LayerNorm(4 * hidden_size)  # First LayerNorm configuration from "Layer Normalization"
      self.ln_2 = nn.LayerNorm(4 * hidden_size)
      self.ln_3 = nn.LayerNorm(hidden_size)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight_x)
    nn.init.orthogonal_(self.weight_h)
    nn.init.zeros_(self.bias)
    if self.T_max:
      bias_f = torch.log(torch.zeros(self.hidden_size).uniform_(1, self.T_max - 1))  # If T_max is given, use chrono initialisation from "Can recurrent neural networks warp time?"
      self.bias.data[: self.hidden_size], self.bias.data[self.hidden_size : 2 * self.hidden_size] = bias_f, -bias_f
    else:
      nn.init.ones_(self.bias[: self.hidden_size])  # Initialise high forget gate bias, from "An Empirical Exploration of Recurrent Network Architectures"

  def _calculate_gates(self, input: Tensor, h_0: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # noqa: A002
    if self.layer_norm:  # noqa: SIM108
      gates = self.ln_1(h_0 @ self.weight_h) + self.ln_2(input @ self.weight_x) + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = LN(W_h⋅h_t−1; α_1, β_1) + LN(W_x⋅x_t; α_2, β_2) + b
    else:
      gates = h_0 @ self.weight_h + input @ self.weight_x + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = W_h⋅h_t−1 + W_x⋅x_t + b
    return gates.chunk(4, dim=1)

  def _calculate_cell_hidden_updates(self, c_0: Tensor, forget_gate: Tensor, in_gate: Tensor, out_gate: Tensor, cell_gate: Tensor) -> tuple[Tensor, Tensor]:
    c_1 = torch.sigmoid(forget_gate) * c_0 + torch.sigmoid(in_gate) * self.dropout(torch.tanh(cell_gate))  # c_t = σ(f_t) ⊙ c_t−1 + σ(i_t) ⊙ d(tanh(g_t))
    if self.layer_norm:  # noqa: SIM108
      h_1 = torch.sigmoid(out_gate) * torch.tanh(self.ln_3(c_1))  # h_t = σ(o_t) ⊙ tanh(LN(c_t; α_3, β_3))
    else:
      h_1 = torch.sigmoid(out_gate) * torch.tanh(c_1)  # h_t = σ(o_t) ⊙ tanh(c_t)
    return h_1, c_1

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0, c_0 = state if state is not None else (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0))

    forget_gate, in_gate, out_gate, cell_gate = self._calculate_gates(input, h_0)
    h_1, c_1 = self._calculate_cell_hidden_updates(c_0, forget_gate, in_gate, out_gate, cell_gate)
    return h_1, (h_1, c_1)

# Plastic LSTM, from "Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity"
class PlasticLSTMCell(LSTMCell):
  def __init__(self, input_size: int, hidden_size: int, dropout: float = 0, layer_norm: bool = False, T_max: int | None = None, neuromodulation: str | None = None):
    super().__init__(input_size, hidden_size, dropout=dropout, layer_norm=layer_norm, T_max=T_max)
    self.neuromodulation = neuromodulation
    self.alpha = nn.Parameter(torch.full((hidden_size, hidden_size), 0.01))  # Plasticity coefficient
    if neuromodulation:
      self.M = nn.Sequential(nn.Linear(hidden_size, hidden_size * hidden_size), nn.Tanh())  # Neuromodulation module
    if neuromodulation is None or neuromodulation == "retroactive":
      self.eta = nn.Parameter(torch.empty((1, )))  # Plasticity/eligibility trace learning rate
    self.reset_parameters()

  def reset_parameters(self):
    super().reset_parameters()
    if hasattr(self, "neuromodulation"):
      if self.neuromodulation:
        nn.init.orthogonal_(self.M[0].weight)
        nn.init.zeros_(self.M[0].bias)
      if self.neuromodulation is None or self.neuromodulation == "retroactive":
        nn.init.constant_(self.eta, 0.01)

  def _calculate_gates(self, input: Tensor, h_0: Tensor, Hebb_0: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # noqa: A002
    weight_h = self.weight_h.unsqueeze(dim=0) + torch.cat([torch.zeros(input.shape[0], self.hidden_size, 3 * self.hidden_size, device=input.device), self.alpha.unsqueeze(dim=0) * Hebb_0], dim=2)  # Add plastic weights for the cell gate: W_h + α⋅Hebb_t-1
    if self.layer_norm:  # noqa: SIM108
      gates = self.ln_1(torch.einsum("bi,bih->bh", h_0, weight_h)) + self.ln_2(input @ self.weight_x) + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = LN((W_h + α⋅Hebb_t-1)⋅h_t−1; α_1, β_1) + LN(W_x⋅x_t; α_2, β_2) + b
    else:
      gates = torch.einsum("bi,bih->bh", h_0, weight_h) + input @ self.weight_x + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = (W_h + α⋅Hebb_t-1)⋅h_t−1 + W_x⋅x_t + b
    return gates.chunk(4, dim=1)

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor, ...] | None) -> tuple[Tensor, tuple[Tensor, Tensor, ...]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    B = input.shape[0]
    if self.neuromodulation == "retroactive":
      h_0, c_0, Hebb_0, E_0 = state if state is not None else (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0), torch.zeros(B, self.hidden_size, self.hidden_size, device=input.device), torch.zeros(B, self.hidden_size, self.hidden_size, device=input.device))
    else:
      h_0, c_0, Hebb_0 = state if state is not None else (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0), torch.zeros(B, self.hidden_size, self.hidden_size, device=input.device))

    forget_gate, in_gate, out_gate, cell_gate = self._calculate_gates(input, h_0, Hebb_0)
    h_1, c_1 = self._calculate_cell_hidden_updates(c_0, forget_gate, in_gate, out_gate, cell_gate)

    if self.neuromodulation == "simple":
      Hebb_1 = torch.clip(Hebb_0 + self.M(h_0).view(-1, self.hidden_size, self.hidden_size) * torch.einsum("bh,bi->bhi", h_0, torch.tanh(cell_gate)), min=-1, max=1)  # Hebb_i,j_t = Clip(Hebb_i,j_t-1 + M_t-1⋅x_i_t-1 ⊗ x_j_t)
      return h_1, (h_1, c_1, Hebb_1)
    elif self.neuromodulation == "retroactive":
      Hebb_1 = torch.clip(Hebb_0 + self.M(h_0).view(-1, self.hidden_size, self.hidden_size) * E_0, min=-1, max=1)  # Hebb_t = Clip(Hebb_t-1 + M_t-1⋅E_t-1)
      E_1 = (1 - self.eta) * E_0 + self.eta * torch.einsum("bh,bi->bhi", h_0, torch.tanh(cell_gate))  # E_i,j_t = (1 − η)E_i,j_t-1 + η⋅x_i_t-1 ⊗ x_j_t
      return h_1, (h_1, c_1, Hebb_1, E_1)
    else:
      Hebb_1 = torch.clip(Hebb_0 + self.eta * torch.einsum("bh,bi->bhi", h_0, torch.tanh(cell_gate)), min=-1, max=1)  # Hebb_i,j_t = Clip(Hebb_i,j_t-1 + η⋅x_i_t-1 ⊗ x_j_t)
      return h_1, (h_1, c_1, Hebb_1)

class LSTM(nn.Module):
  def __init__(self,
               data_interaction,
               out_dims,
               d_model,
               d_input,
               dropout=0,
               layer_norm=False,
               num_lstm_layers=1,
               plastic=False
               ):
    super().__init__()

    self.data_interaction = data_interaction
    self.out_dims = out_dims
    self.d_model = d_model
    self.plastic = plastic

    cell = PlasticLSTMCell if plastic else LSTMCell
    self.lstm_layers = nn.ModuleList([
        cell(d_input if layer_idx == 0 else d_model, d_model, dropout=dropout, layer_norm=layer_norm)
        for layer_idx in range(num_lstm_layers)
    ])

    self.output_projector = nn.Sequential(
        nn.Linear(in_features=d_model, out_features=512, bias=True),
        nn.LayerNorm(512),
        nn.Linear(in_features=512, out_features=out_dims, bias=True),
    )

  def forward(self, x, aux_inputs=None):

    B, iterations, *input_shape = x.shape

    device = x.device

    # --- Prepare Storage for Outputs per Iteration ---
    predictions = torch.empty(B, self.out_dims, iterations, device=device, dtype=torch.float32)

    # --- Initialize LSTM States for All Layers ---
    lstm_states = [None] * len(self.lstm_layers)

    # --- Recurrent Loop  ---
    for stepi in range(iterations):
      
        # --- Featurise Input Data ---
        hidden_state, attn_weights = self.data_interaction(x[:, stepi], None, aux_inputs=aux_inputs[:, stepi] if aux_inputs is not None else None)

        # --- Loop through LSTM Layers ---
        for layer_idx, lstm_layer in enumerate(self.lstm_layers):

            # Save residual for skip connection
            residual = hidden_state

            # Forward through layer
            hidden_state, lstm_states[layer_idx] = lstm_layer(hidden_state, lstm_states[layer_idx])

            # For layers after the first, apply residual connection and layer norm
            if layer_idx > 0:
                hidden_state = F.layer_norm(hidden_state + residual, normalized_shape=(self.d_model,))

        # --- Get Predictions from Final Layer Output ---
        predictions[..., stepi] = self.output_projector(hidden_state)

    return predictions
