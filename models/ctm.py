import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SuperLinear(nn.Module):
    def __init__(self, in_dims, out_dims, N, do_norm=False, dropout=0, bias=True, requires_grad=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=requires_grad)
        )
        if bias:
            self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=requires_grad))
        else:
            self.register_parameter('b1', None)

    def forward(self, x):
        out = self.dropout(x)
        out = self.layernorm(out)
        out = torch.einsum('BDM,MHD->BDH', out, self.w1)
        if self.b1 is not None:
            out = out + self.b1
        out = out.squeeze(-1)
        return out


class PlasticLinear(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, do_layernorm=False):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.weights_fixed = nn.Parameter(torch.empty(out_features, out_features))
        nn.init.kaiming_uniform_(self.weights_fixed, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.proj_to_d_model = nn.LazyLinear(hidden_features)

        self.do_layernorm = do_layernorm
        if self.do_layernorm:
            self.layernorm = nn.LayerNorm(out_features, elementwise_affine=True)

    def randomize_weights(self):
        bound = math.sqrt(1 / (2 * self.in_features))
        with torch.no_grad():
            self.weights_fixed.uniform_(-bound, bound)

    def forward(self, x, synapse_weights_plastic=None, synapse_bias_plastic=None):

        B = x.shape[0]

        x = self.proj_to_d_model(x) # (B, _) -> (B, hidden_features)

        weights_fixed = self.weights_fixed.unsqueeze(0).expand(B, -1, -1)
        
        if synapse_weights_plastic is None:
            weights_normalized = weights_fixed / torch.norm(weights_fixed, p='fro', dim=(-2, -1), keepdim=True)
            output = torch.bmm(x.unsqueeze(1), weights_normalized).squeeze(1)
        else:
            combined_weights = weights_fixed + synapse_weights_plastic
            weights_normalized = combined_weights / torch.norm(combined_weights, p='fro', dim=(-2, -1), keepdim=True)
            output = torch.bmm(x.unsqueeze(1), weights_normalized).squeeze(1)

        if self.bias is not None:
            bias = self.bias.unsqueeze(0).expand(B, -1)
            if synapse_bias_plastic is not None:
                output = output + bias + synapse_bias_plastic
            else:
                output = output + bias

        # do layernorm
        if self.do_layernorm:
            output = self.layernorm(output)

        return output

class Synchronizer(nn.Module):
    def __init__(self, d_model, do_layernorm=True):
        super(Synchronizer, self).__init__()
        self.synch_representation_size = d_model * d_model

        self.do_layernorm = do_layernorm
        if do_layernorm:
            self.layernorm = nn.LayerNorm(self.synch_representation_size)

        self.register_buffer('neuron_indices_left', torch.arange(d_model).repeat_interleave(d_model))
        self.register_buffer('neuron_indices_right', torch.arange(d_model).repeat(d_model))
        self.register_parameter(f'decay_params', nn.Parameter(torch.rand(self.synch_representation_size) * 2, requires_grad=True))

    def get_decay_weights(self, B):
        self.decay_params.data = torch.clamp(self.decay_params, 0, 15)
        decay_weights = torch.exp(-self.decay_params).unsqueeze(0).repeat(B, 1)
        return decay_weights

    def forward(self, state, decay_alpha, decay_beta, decay_weights):
        left = state[:, self.neuron_indices_left]
        right = state[:, self.neuron_indices_right]
        pairwise_product = left * right

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = decay_weights**2 * decay_alpha + pairwise_product
            decay_beta = decay_weights**2 * decay_beta + 1

        synchronisation = decay_alpha / (torch.sqrt(decay_beta))

        if self.do_layernorm:
            synchronisation = self.layernorm(synchronisation)

        return synchronisation, decay_alpha, decay_beta

class ContinuousThoughtMachineCell(nn.Module):
    def __init__(self,
                 d_model,
                 d_input,
                 memory_length,
                 hidden_dim_nlm,
                 do_layernorm_nlm,
                 bias_nlm,
                 dropout_nlm,
                ):
        super(ContinuousThoughtMachineCell, self).__init__()

        # --- Core Parameters ---
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.hidden_dim_nlm = hidden_dim_nlm

        # --- Core CTM Modules ---
        self.synapses = PlasticLinear(in_features=(d_input+d_model), hidden_features=d_model, out_features=d_model, bias=False, do_layernorm=True)
        self.nlms = nn.Sequential(
                        nn.Sequential(
                            SuperLinear(in_dims=memory_length, out_dims=2 * hidden_dim_nlm, N=d_model, do_norm=False, dropout=dropout_nlm, bias=bias_nlm),
                            nn.GLU(),
                            SuperLinear(in_dims=hidden_dim_nlm, out_dims=2, N=d_model,do_norm=do_layernorm_nlm, dropout=dropout_nlm, bias=bias_nlm),
                            nn.GLU(),
                            Squeeze(-1)
                        )
                    )
        self.synchronizer = Synchronizer(d_model=d_model, do_layernorm=True)

        #  --- Start States ---
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((d_model, memory_length))))
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((d_model))))

    def initialize_state(self, B):
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)
        activated_state_trace = activated_state.unsqueeze(-1) # Shape: (B, H, 1)
        decay_alpha = decay_beta = None
        initial_synch = torch.einsum('bd,be->bde', activated_state, activated_state).view(B, -1)  # Shape: (B, D**2)
        return (state_trace, activated_state_trace, decay_alpha, decay_beta), initial_synch

    def forward(self, x, ctm_layer_state, synapse_weights_plastic=None):

        B = x.shape[0]

        pre_activations, post_activations, decay_alpha, decay_beta  = ctm_layer_state
        state_trace = pre_activations[:,:,-self.memory_length:]
        activated_state = post_activations[:,:,-1]

        # --- Apply Synapses ---
        state = self.synapses(torch.concat([x, activated_state], dim=-1), synapse_weights_plastic)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        # --- Apply Neuron-Level Models ---
        activated_state = self.nlms(state_trace)

        # --- Compute Synchronization ---
        decay_weights = self.synchronizer.get_decay_weights(B)
        synchronisation, decay_alpha, decay_beta = self.synchronizer(activated_state, decay_alpha, decay_beta, decay_weights)

        updated_pre_activations = torch.cat((pre_activations, state.unsqueeze(-1)), dim=-1)
        updated_post_activations = torch.cat((post_activations, activated_state.unsqueeze(-1)), dim=-1)

        update_ctm_layer_state = (updated_pre_activations, updated_post_activations, decay_alpha, decay_beta)

        return synchronisation, activated_state, update_ctm_layer_state

class ContinuousThoughtMachine(nn.Module):               
    def __init__(self,
                 data_interaction,
                 out_dims,
                 d_model,
                 d_input,
                 memory_length,
                 hidden_dim_nlm,
                 do_layernorm_nlm,
                 bias_nlm,
                 dropout=0,
                 num_ctm_layers=1,
                ):
        super(ContinuousThoughtMachine, self).__init__()

        # --- Core Parameters ---
        self.d_model = d_model
        self.d_input = d_input
        self.data_interaction = data_interaction
        self.memory_length = memory_length
        self.out_dims = out_dims
        self.num_ctm_layers = num_ctm_layers
        self.output_projector = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=512, bias=True),
            nn.Linear(in_features=512, out_features=out_dims, bias=False)
        )
                
        # --- CTM Layers ---
        self.ctm_layers = nn.ModuleList([
            ContinuousThoughtMachineCell(
                d_model=d_model,
                d_input=d_input if i == 0 else d_model,
                memory_length=memory_length,
                hidden_dim_nlm=hidden_dim_nlm,
                do_layernorm_nlm=do_layernorm_nlm,
                bias_nlm=bias_nlm,
                dropout_nlm=dropout,
            )
            for i in range(num_ctm_layers)
        ])

    def synch_reshape(self, synch, do_layernorm=True):
        B = synch.size(0)
        if do_layernorm:
            synch = F.layer_norm(synch, normalized_shape=(self.d_model ** 2,))
        return synch.view(B, self.d_model, self.d_model)

    def forward(self, x, aux_inputs=None):

        B, iterations, *input_shape = x.shape

        device = x.device

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent States for All Layers ---
        ctm_states = []
        synchronizations = []
        for layer in self.ctm_layers:
            state, synch = layer.initialize_state(B)
            ctm_states.append(state)
            synchronizations.append(synch)

        # --- Recurrent Loop  ---
        for stepi in range(iterations):

            # --- Featurise Input Data ---
            activated_state, attn_weights = self.data_interaction(x[:, stepi], None, aux_inputs=aux_inputs[:, stepi] if aux_inputs is not None else None)

            # --- Loop through CTM Layers ---
            for layer_idx, ctm_layer in enumerate(self.ctm_layers):

                residual = activated_state

                # Forward through layer
                synchronizations[layer_idx], activated_state, ctm_states[layer_idx] = ctm_layer(activated_state, ctm_layer_state=ctm_states[layer_idx], synapse_weights_plastic=self.synch_reshape(synchronizations[layer_idx]))

                # For layers after the first, apply residual connection and layer norm
                if layer_idx > 0:
                    activated_state = F.layer_norm(residual + activated_state, normalized_shape=(self.d_model,))

            # --- Get Predictions from Final Layer Output ---
            predictions[..., stepi] = self.output_projector(activated_state)

        return predictions
