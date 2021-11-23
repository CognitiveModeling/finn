from torch.autograd import Variable
import torch as th


class ConvLSTMCell(th.nn.Module):
    """
    A Convolutional LSTM implementation taken from
    https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, bias,
                 dimensions):
        """
        Initialize ConvLSTM cell.
        :param input_channels: Number of channels of input tensor.
        :param hidden_channels: Number of channels of hidden state.
        :param kernel_size: Size of the convolutional kernel.
        :param bias: Whether or not to add the bias.
        :param dimensions: The spatial dimensions of the data
        """

        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        conv_function = th.nn.Conv1d if dimensions == 1 else th.nn.Conv2d
        self.conv = conv_function(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = th.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = th.split(
            combined_conv, self.hidden_channels, dim=1
        )
        i = th.sigmoid(cc_i)
        f = th.sigmoid(cc_f)
        o = th.sigmoid(cc_o)
        g = th.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * th.tanh(c_next)

        return h_next, c_next


class ConvLSTM(th.nn.Module):
    """
    ConvLSTM class to model the 1D Burgers equation, the two 1D diffusion
    sorption equations or the 2d diffusion reaction equations.
    """

    def __init__(self, config, device):
        """
        Constructor method to initialize the TCN instance.
        :param config: The configuration class of the model
        :param device: The device (GPU or CPU) which processes the tensors
        """
        super(ConvLSTM, self).__init__()

        self.layers = th.nn.ModuleList()
        self.state_list = []

        for ch_idx in range(1, len(config.model.channels)):
            self.layers.append(ConvLSTMCell(
                input_channels=config.model.channels[ch_idx - 1],
                hidden_channels=config.model.channels[ch_idx],
                kernel_size=config.model.kernel_size,
                bias=True,
                dimensions=len(config.model.field_size)
            ))

            # Initialize an exemplary state
            s = th.zeros(config.training.batch_size,
                         config.model.channels[ch_idx],
                         *config.model.field_size,
                         device=device)

            # Append a pair of h and c states to the state list
            self.state_list.append([s.clone(), s.clone()])

    def forward(self, input_tensor, cur_state_list=None):
        """
        Forward pass of the ConvLSTMModel class
        """

        if cur_state_list is None:
            cur_state_list = self.state_list

        next_state_list = []

        for layer_idx, layer in enumerate(self.layers):
            h, c = layer.forward(input_tensor=input_tensor,
                                 cur_state=cur_state_list[layer_idx])

            next_state_list.append([h, c])
            input_tensor = h

        self.state_list = next_state_list

        return h, next_state_list

    def reset(self, batch_size, device):
        """
        Resets the ConvLSTM's states to zero by potentially changing the
        batch_size.
        """
        for layer_idx, state_pair in enumerate(self.state_list):
            s = th.zeros(batch_size, *state_pair[0][0].shape, device=device)
            self.state_list[layer_idx] = [s.clone(), s.clone()]
