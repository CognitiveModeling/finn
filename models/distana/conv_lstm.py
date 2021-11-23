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
