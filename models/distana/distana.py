import torch as th
import torch.nn as nn
from conv_lstm import ConvLSTMCell


class DISTANACell(nn.Module):
    """
    This class contains the kernelized network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, dyn_channels, lat_channels, hidden_channels, kernel_size,
                 batch_size, data_size, bias, device):

        super(DISTANACell, self).__init__()

        # Initialize DISTANA's states for lateral data exchange and lstm h,c
        #s = th.zeros(batch_size, hidden_channels, *data_size, device=device)
        self.l = th.zeros(batch_size, lat_channels, *data_size, device=device)
        self.h = th.zeros(batch_size, hidden_channels, *data_size, device=device)
        self.c = th.zeros(batch_size, hidden_channels, *data_size, device=device)

        self.device = device
        self.dyn_channels = dyn_channels

        dimensions = len(data_size)
        conv_function = th.nn.Conv1d if dimensions == 1 else th.nn.Conv2d

        # Lateral input convolution layer
        self.lat_in_conv_layer = conv_function(
            in_channels=lat_channels,
            out_channels=lat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        ).to(device=device)

        # Dynamic and lateral input preprocessing layer
        self.pre_layer = conv_function(
            in_channels=dyn_channels + lat_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        ).to(device=device)

        # Central LSTM layer
        self.clstm = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=bias,
            dimensions=dimensions
        ).to(device=device)

        # Postprocessing layer
        self.post_layer = conv_function(
            in_channels=hidden_channels,
            out_channels=dyn_channels + lat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        ).to(device=device)

    def forward(self, dyn_in, state=None):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel for
        a given input
        :param dyn_in: The dynamic input for the PKs
        :param state: The latent state of the module [l, h, c]
        """

        if state is None:
            lat_in, lstm_h, lstm_c = self.l, self.h, self.c
        else:
            lat_in, lstm_h, lstm_c = state

        # Compute the lateral input as convolution of the latent lateral
        # outputs from the previous timestep
        lat_in = self.lat_in_conv_layer(lat_in)

        # Forward the dynamic and lateral inputs through the preprocessing
        # layer
        dynlat_in = th.cat(tensors=(dyn_in, lat_in), dim=1)
        pre_act = th.tanh(self.pre_layer(dynlat_in))

        # Feed the preprocessed data through the lstm
        lstm_h, lstm_c = self.clstm(input_tensor=pre_act,
                                    cur_state=[lstm_h, lstm_c])

        # Pass the lstm output through the postprocessing layer
        post_act = self.post_layer(lstm_h)

        # Split the post activation into dynamic and latent lateral outputs
        dyn_out = post_act[:, :self.dyn_channels]

        # Lateral output
        lat_out = th.tanh(post_act[:, self.dyn_channels:])

        # State update
        self.l = lat_out
        self.h = lstm_h
        self.c = lstm_c

        return dyn_out
        
    def reset(self, batch_size):
        #s = th.zeros(batch_size, *self.l.shape[1:], device=self.device)
        self.l = th.zeros(batch_size, *self.l.shape[1:], device=self.device)
        self.h = th.zeros(batch_size, *self.h.shape[1:], device=self.device)
        self.c = th.zeros(batch_size, *self.c.shape[1:], device=self.device)


class DISTANA(nn.Module):
    """
    DISTANA class to model the 1D Burgers equation, the two 1D diffusion
    sorption equations or the 2d diffusion reaction equations.
    """
    def __init__(self, config, device):
        """
        Constructor
        """
        super(DISTANA, self).__init__()

        self.layers = th.nn.ModuleList()

        for ch_idx in range(len(config.model.dynamic_channels)):
            layer = DISTANACell(
                dyn_channels=config.model.dynamic_channels[ch_idx],
                lat_channels=config.model.lateral_channels[ch_idx],
                hidden_channels=config.model.hidden_channels[ch_idx],
                kernel_size=config.model.kernel_size,
                batch_size=config.training.batch_size,
                data_size=config.model.field_size,
                bias=True,
                device=device
            )
            self.layers.append(layer)

    def forward(self, input_tensor, cur_state_list=None):
        """
        Forward pass of the ConvLSTM_Burger class
        """

        next_state_list = []

        for layer_idx, layer in enumerate(self.layers):
            dyn_out = layer.forward(
                dyn_in=input_tensor,
                state=None if cur_state_list is None else cur_state_list[layer_idx]
            )

            next_state_list.append([layer.l, layer.h, layer.c])
            input_tensor = dyn_out

        self.state_list = next_state_list

        return dyn_out, next_state_list

    def reset(self, batch_size):
        """
        Set all states back to zero using a given batch_size
        """
        for layer in self.layers:
            layer.reset(batch_size=batch_size)