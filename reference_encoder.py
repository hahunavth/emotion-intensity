import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import torch

class SpatialDropout1D(nn.Module):
    def __init__(self, drop_rate):
        super(SpatialDropout1D, self).__init__()
        
        self.dropout = nn.Dropout2d(drop_rate)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        inputs = self.dropout(inputs.unsqueeze(2)).squeeze(2)
        inputs = inputs.permute(0, 2, 1)
        
        return inputs

class Conv_Net(nn.Module,):
    def __init__( self, channels=[80, 128], 
        conv_kernels=[3,], conv_strides=[1,], 
        maxpool_kernels=[2,], maxpool_strides=[2,], dropout=0.2):
        
        super(Conv_Net, self).__init__()
        
        convs = []
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.maxpool_kernels = maxpool_kernels
        self.maxpool_strides = maxpool_strides
        
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            conv = [
                nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=self.conv_kernels[i],
                    stride=self.conv_strides[i],
                    padding=self.conv_kernels[i]-1),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU()
            ]            
            convs += conv
        
        self.conv_net = nn.ModuleList(convs)
        self.dropout = SpatialDropout1D(dropout)
        
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length + 2*(kernel_size-1) - kernel_size) // stride + 1
        
        def _maxpool_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for index in range(len(self.conv_kernels)):
            input_lengths = _conv_out_length(input_lengths, self.conv_kernels[index], self.conv_strides[index])
            # input_lengths = _maxpool_out_length(input_lengths, self.maxpool_kernels[index], self.maxpool_strides[index])
            
        return input_lengths.to(torch.long)
    
    def _get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
                    
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

        # inputs_lengths = torch.sum(~masks, dim=-1)
        # outputs_lengths = self._get_feat_extract_output_lengths(inputs_lengths)
        # masks = self._get_mask_from_lengths(outputs_lengths)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        for conv in self.conv_net:
            inputs = conv(inputs)
        inputs = self.dropout(inputs)
        inputs = inputs.permute(0, 2, 1)
        
        return inputs

class AdditiveAttention(nn.Module):
    def __init__(self, dropout,
                 query_vector_dim,
                 candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector),dim=1)
        candidate_weights = self.dropout(candidate_weights)
            
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target

class Reference_Encoder(nn.Module):
    def __init__(self, config) -> None:
        super(Reference_Encoder, self).__init__()
        
        self.conv_net = Conv_Net(
            channels=[80, 64, 128, 128, 256, 128], 
            conv_kernels=[3, 3, 3, 3, 3], conv_strides=[2, 1, 2, 1, 2], 
            dropout=config["reference_encoder"]["dropout"]
        )
        self.dropout = nn.Dropout(config["reference_encoder"]["dropout"])
        
        self.attn_head = AdditiveAttention(
            dropout=config["reference_encoder"]["dropout"],
            query_vector_dim=128,
            candidate_vector_dim=128)

    def forward(self, inputs):
        outputs= self.conv_net(inputs)
        outputs = self.attn_head(outputs)
        outputs = self.dropout(outputs)
        
        return outputs


if __name__ == "__main__":
    import yaml
    config = yaml.load(open("/home/hahunavth/Desktop/VLSP2023_ESS/repo/config/model.yaml", "r"), Loader=yaml.FullLoader)

    model = Reference_Encoder(config)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    inputs = torch.randn(8, 512, 80)
    masks = torch.ones(8, 512)

    output = model(inputs * masks.unsqueeze(-1))
    print(output[0].shape)
    print(output[1].shape)
    print(output.shape)