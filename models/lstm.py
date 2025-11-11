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


class LSTM(nn.Module):
  def __init__(self,
               data_interaction,
               out_dims,
               d_model,
               d_input,
               dropout=0,
               layer_norm=False,
               num_lstm_layers=1,
               ):
    super().__init__()

    self.data_interaction = data_interaction
    self.out_dims = out_dims
    self.d_model = d_model

    self.lstm_layers = nn.ModuleList([
        LSTMCell(d_input if layer_idx == 0 else d_model, d_model, dropout=dropout, layer_norm=layer_norm)
        for layer_idx in range(num_lstm_layers)
    ])

    self.output_projector = nn.Sequential(
        nn.Linear(in_features=d_model, out_features=512, bias=True),
        nn.Linear(in_features=512, out_features=out_dims, bias=False)
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
