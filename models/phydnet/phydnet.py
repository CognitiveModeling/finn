#!/usr/bin/env python3

"""
Physics-derivatives learning layer. Implementation taken and modified from
https://github.com/vincent-leguen/PhyDNet
"""

import torch
import torch.nn as nn
import numpy as np

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        """
        input_dim: int
            Number of channels of input tensor.
        F_hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int for 1D, (int, int) for 2D
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = tuple(np.floor_divide(self.kernel_size,2))
        self.bias = bias
        
        self.F = nn.Sequential()
        if len(self.kernel_size) == 1:
            self.F.add_module('conv1', nn.Conv1d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,)*len(self.kernel_size), padding=self.padding))
            self.F.add_module('bn1',nn.GroupNorm( 7 ,F_hidden_dim))        
            self.F.add_module('conv2', nn.Conv1d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,)*len(self.kernel_size), stride=(1,)*len(self.kernel_size), padding=(0,)*len(self.kernel_size)))
            
            self.convgate = nn.Conv1d(in_channels=self.input_dim + self.input_dim,
                                  out_channels= self.input_dim,
                                  kernel_size=(3,)*len(self.kernel_size),
                                  padding=(1,)*len(self.kernel_size), bias=self.bias)
        else:
            assert len(self.kernel_size) == 2, "Input dimension has to be either 1 or 2."
            self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,)*len(self.kernel_size), padding=self.padding))
            self.F.add_module('bn1',nn.GroupNorm( 7 ,F_hidden_dim))        
            self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,)*len(self.kernel_size), stride=(1,)*len(self.kernel_size), padding=(0,)*len(self.kernel_size)))
            
            self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels= self.input_dim,
                                  kernel_size=(3,)*len(self.kernel_size),
                                  padding=(1,)*len(self.kernel_size), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)        # prediction
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden

class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        """
        input_shape: int for 1D, (int, int) for 2D
            Shape of input tensor: a single integer for 1D input, a tuple of
            (height, width) for 2D.
        input_dim: int
            Number of channels of input tensor.
        F_hidden_dim: int
            Number of channels of hidden state.
        n_layers: int
            Number of PhyCell blocks.
        kernel_size: int for 1D, (int, int) for 2D
            Size of the convolutional kernel.
        device: torch.device
            Device to perform calculation (cpu or cuda)
        """
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        if type(self.input_shape) is not tuple:
            self.input_shape = (self.input_shape,)
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        if type(self.kernel_size) is not tuple:
            self.kernel_size = (self.kernel_size,)
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros((batch_size, self.input_dim) + self.input_shape).to(self.device) )

    def setHidden(self, H):
        self.H = H

        
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: int for 1D, (int, int) for 2D
            Shape of input tensor: a single integer for 1D input, a tuple of
            (height, width) for 2D.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: array with dimension [n_layers]
            Number of channels of each hidden layer.
        kernel_size: int for 1D, (int, int) for 2D
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = tuple(np.floor_divide(self.kernel_size,2))
        self.bias        = bias
        
        if len(self.kernel_size) == 1:
            self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding, bias=self.bias)
        else:
            assert len(self.kernel_size) == 2, "Input dimension has to be either 1 or 2."
            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        """
        input_shape: int for 1D, (int, int) for 2D
            Shape of input tensor: a single integer for 1D input, a tuple of
            (height, width) for 2D.
        input_dim: int
            Number of channels of input tensor.
        hidden_dims: array with dimension [n_layers]
            Number of channels of each hidden layer.
        n_layers: int
            Number of ConvLSTM_Cell blocks.
        kernel_size: int for 1D, (int, int) for 2D
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        if type(self.input_shape) is not tuple:
            self.input_shape = (self.input_shape,)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        if type(self.kernel_size) is not tuple:
            self.kernel_size = (self.kernel_size,)
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            # print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros((batch_size,self.hidden_dims[i]) + self.input_shape).to(self.device) )
            self.C.append( torch.zeros((batch_size,self.hidden_dims[i])+ self.input_shape).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride, padding=1, _1d=False, small=True):
        super(dcgan_conv, self).__init__()
        if small:
            groups = 4
        else:
            groups = 16
            
        if _1d:
            self.main = nn.Sequential(
                    nn.Conv1d(in_channels=nin, out_channels=nout, kernel_size=3, stride=stride, padding=padding),
                    nn.GroupNorm(groups,nout),
                    nn.LeakyReLU(0.2, inplace=True),
                    )
        else:
            self.main = nn.Sequential(
                    nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=stride, padding=padding),
                    nn.GroupNorm(groups,nout),
                    nn.LeakyReLU(0.2, inplace=True),
                    )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride, _1d=False, even=(1,), small=True):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = even
        else:
            output_padding = 0
        
        if small:
            groups = 4
        else:
            groups = 16
        
        if _1d:
            self.main = nn.Sequential(
                    nn.ConvTranspose1d(in_channels=nin,out_channels=nout,kernel_size=3, stride=stride,padding=1,output_padding=output_padding),
                    nn.GroupNorm(groups,nout),
                    nn.LeakyReLU(0.2, inplace=True),
                    )
        else:
            self.main = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=3, stride=stride,padding=1,output_padding=output_padding),
                    nn.GroupNorm(groups,nout),
                    nn.LeakyReLU(0.2, inplace=True),
                    )

    def forward(self, input):
        return self.main(input)
        

class encoder_E(nn.Module):
    def __init__(self, nc=1, nf=32, _1d=False, small=True):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2, padding=0, _1d=_1d, small=small) # (nf) x (input_shape,)//2
        self.c2 = dcgan_conv(nf, nf, stride=1, _1d=_1d, small=small) # (nf) x (input_shape,)//2
        self.c3 = dcgan_conv(nf, 2*nf, stride=2, _1d=_1d, small=small) # (2*nf) x (input_shape,)//4

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3

class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32, _1d=False, even1=(1,), even2=(1,), small=True):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2*nf, nf, stride=2, _1d=_1d, even=even2, small=small) #(nf) x (input_shape,)//2
        self.upc2 = dcgan_upconv(nf, nf, stride=1, _1d=_1d, small=small) #(nf) x (input_shape,)//2
        if _1d:
            self.upc3 = nn.ConvTranspose1d(in_channels=nf,out_channels=nc,kernel_size=3,stride=2,padding=1,output_padding=even1)  ##(nf) x (input_shape,)
        else:
            self.upc3 = nn.ConvTranspose2d(in_channels=nf,out_channels=nc,kernel_size=3,stride=2,padding=1,output_padding=even1)  ##(nf) x (input_shape,)

    def forward(self, input):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)
        d3 = self.upc3(d2) 
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64, _1d=False, small=True):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1, _1d=_1d, small=small) #(nf) x (input_shape,)//4
        self.c2 = dcgan_conv(nf, nf, stride=1, _1d=_1d, small=small) #(nf) x (input_shape,)//4

    def forward(self, input):
        h1 = self.c1(input)  
        h2 = self.c2(h1)     
        return h2

class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64, _1d=False, small=True):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1, _1d=_1d, small=small) #(nf) x (input_shape,)//4
        self.upc2 = dcgan_upconv(nf, nc, stride=1, _1d=_1d, small=small) #(nf) x (input_shape,)//2
        
    def forward(self, input):
        d1 = self.upc1(input) 
        d2 = self.upc2(d1)  
        return d2       

        
class EncoderRNN(torch.nn.Module):
    def __init__(self, phycell, convcell, input_channels, input_dim, _1d, bc,
                 device, sigmoid=True, small=True):
        """
        phycell: object
            The PhyCell object.
        convcell: object
            The ConvLSTM object.
        input_channels: int
            Number of input channels (number of unknown variables).
        input_dim: int for 1D, (int, int) for 2D
            Shape of the image (number of spatial cells).
        _1d: bool
            To determine whether the problem is 1D or 2D.
        bc: torch.tensor
            A tensor containing the boundary condition values.
        device: torch.device
            Device to perform calculation (CPU or CUDA).
        sigmoid: bool
            Whether to use sigmoid or tanh in the last output layer.
        """
        
        super(EncoderRNN, self).__init__()
        
        assert phycell.input_dim == convcell.input_dim, "Input channels for PhyCell and ConvLSTM_Cell has to be equal."
        
        # Check if the first downsampling stems from an evenly-shaped input
        even1 = tuple((np.remainder(input_dim,2)==0)*1)
        # Check if the second downsampling stems from an evenly-shaped input
        even2 = tuple((np.remainder(np.ceil(np.divide(input_dim,2)),2)==0)*1)
        
        self.encoder_E = encoder_E(nc=input_channels, nf=phycell.input_dim//2, _1d=_1d, small=small)   # general encoder 64x64x1 -> 32x32x32
        self.decoder_D = decoder_D(nc=input_channels, nf=phycell.input_dim//2, _1d=_1d, even1=even1, even2=even2, small=small)  # general decoder 32x32x32 -> 64x64x1 
        self.encoder_E = self.encoder_E.to(device)
        self.decoder_D = self.decoder_D.to(device)
        
        if not small:
            self.encoder_Ep = encoder_specific(nc=phycell.input_dim, nf=phycell.input_dim, _1d=_1d, small=small) # specific image encoder 32x32x32 -> 16x16x64
            self.encoder_Er = encoder_specific(nc=convcell.input_dim, nf=convcell.input_dim, _1d=_1d, small=small) 
            self.decoder_Dp = decoder_specific(nc=phycell.input_dim, nf=phycell.input_dim, _1d=_1d, small=small) # specific image decoder 16x16x64 -> 32x32x32 
            self.decoder_Dr = decoder_specific(nc=convcell.input_dim, nf=convcell.input_dim, _1d=_1d, small=small)     
            self.encoder_Ep = self.encoder_Ep.to(device) 
            self.encoder_Er = self.encoder_Er.to(device) 
            self.decoder_Dp = self.decoder_Dp.to(device) 
            self.decoder_Dr = self.decoder_Dr.to(device)  
            
        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)
        
        self._1d = _1d
        self.bc = bc
        self.sigmoid = sigmoid
        self.small = small

    def forward(self, input, first_timestep=False, decoding=False):
        
        # Pad boundary conditions
        
        if self._1d:
            input = torch.cat((self.bc[:1,:,:1],input,self.bc[:1,:,1:]),dim=2)
        else:
            Nx = input.size(-2)
            input = torch.cat((self.bc[:1,:,:1].repeat(1,1,Nx).unsqueeze(-1),input,
                               self.bc[:1,:,1:2].repeat(1,1,Nx).unsqueeze(-1)),dim=3)
            Ny = input.size(-1)
            input = torch.cat((self.bc[:1,:,2:3].repeat(1,1,Ny).unsqueeze(-2),input,
                               self.bc[:1,:,3:4].repeat(1,1,Ny).unsqueeze(-2)),dim=2)
            
        input = self.encoder_E(input) # general encoder 64x64x1 -> 32x32x32
        
        if self.small:
            if decoding:  # input=None in decoding phase
                input_phys = None
            else:
                input_phys = input
            input_conv = input
    
            hidden1, output1 = self.phycell(input_phys, first_timestep)
            hidden2, output2 = self.convcell(input_conv, first_timestep)
    
            out_phys = torch.sigmoid(output1[-1]) # partial reconstructions for vizualization
            out_conv = torch.sigmoid(output2[-1])
    
            concat = out_phys + out_conv
            if self.sigmoid:
                output_image = torch.sigmoid( self.decoder_D(concat ))
            else:
                output_image = torch.tanh( self.decoder_D(concat ))
        
        else:
            if decoding:  # input=None in decoding phase
                input_phys = None
            else:
                input_phys = self.encoder_Ep(input)
                
            input_conv = self.encoder_Er(input)     
    
            hidden1, output1 = self.phycell(input_phys, first_timestep)
            hidden2, output2 = self.convcell(input_conv, first_timestep)
    
            decoded_Dp = self.decoder_Dp(output1[-1])
            decoded_Dr = self.decoder_Dr(output2[-1])
            
            out_phys = torch.sigmoid(self.decoder_D(decoded_Dp)) # partial reconstructions for vizualization
            out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))
    
            concat = decoded_Dp + decoded_Dr
            
            if self.sigmoid:
                output_image = torch.sigmoid( self.decoder_D(concat ))
            else:
                output_image = torch.tanh( self.decoder_D(concat ))
        
        return out_phys, hidden1, output_image, out_phys, out_conv 