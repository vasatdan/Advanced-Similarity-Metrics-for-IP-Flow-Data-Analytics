import torch
import torch.nn.functional as F
import torch.nn as nn

class GeM(nn.Module):
    """
    Code taken from
    https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference
    """
    def __init__(self, kernel_size, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p) # type: ignore
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        #print('foward x', x.shape)
        return self.gem(x, p=self.p, eps=self.eps) # type: ignore

    def gem(self, x, p=3, eps=1e-6):
        # print(self.kernel_size, x.shape)
        output_gem = F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)
        # print(output_gem.shape)
        return output_gem

    def __repr__(self):
        return self.__class__.__name__ + \
                "(" + "kernel_size=" + str(self.kernel_size) + ", p=" + "{:.4f}".format(self.p.data.tolist()[0]) + \
                ", eps=" + str(self.eps) + ")"

class MultimodalClassificationModel(nn.Module):
    """
    The deep neural network model
    """
    def __init__(self, cnn_final_len:int = 10, ppi_unit1:int = 150, ppi_unit2:int = 300, cnn_num_hidden:int = 2, 
                 cnn_phist_final_len:int = 3, phist_unit1:int = 50, phist_unit2:int = 100, others_unit:int = 300, cnn_phist_num_hidden:int = 2,
                 common_unit:int = 300 , others_num_hidden:int = 2, dropout_rate:float = 0.2, 
                 classes_to_predict = 154, ppi_shape = (5,30), phists_shape = (4,8), other_shape = (55,)):
        super().__init__()
        self.layers_for_grad = []
        self.temperature: Optional[torch.Tensor] = None
        # save dimensions
        self.ppi_shape = ppi_shape
        self.phists_shape = phists_shape
        self.other_shape = other_shape

        #---------------------------------
        # PPI
        self.cnn_ppi_pkt = nn.Sequential(
            nn.Conv1d(in_channels = self.ppi_shape[0], out_channels = ppi_unit1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(ppi_unit1),
            
            *(nn.Sequential(
                nn.Conv1d(in_channels = ppi_unit1, out_channels = ppi_unit1, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=False),
                nn.BatchNorm1d(ppi_unit1),) for _ in range(cnn_num_hidden)),

            nn.Conv1d(in_channels = ppi_unit1, out_channels = ppi_unit2, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(ppi_unit2),

            nn.Conv1d(in_channels = ppi_unit2, out_channels = ppi_unit2, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(ppi_unit2),

            nn.Conv1d(in_channels = ppi_unit2, out_channels = ppi_unit2, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),
        )
        #---------------------------------
        # PHIST
        self.cnn_phist_pkt = nn.Sequential(
            nn.Conv1d(in_channels = self.phists_shape[0], out_channels = phist_unit1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(phist_unit1),
            
            *(nn.Sequential(
                nn.Conv1d(in_channels = phist_unit1, out_channels = phist_unit1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),
                nn.BatchNorm1d(phist_unit1),) for _ in range(cnn_phist_num_hidden)),

            nn.Conv1d(in_channels = phist_unit1, out_channels = phist_unit2, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
        )
        
        self.cnn_global_pooling = nn.Sequential(
            GeM(kernel_size=cnn_final_len),
            nn.Flatten(),
            nn.BatchNorm1d(ppi_unit2*(10//cnn_final_len)),
            nn.Dropout(dropout_rate),
        )
        self.cnn_phist_global_pooling = nn.Sequential(
            GeM(kernel_size=cnn_phist_final_len),
            nn.Flatten(),
            nn.BatchNorm1d(phist_unit2*(4//cnn_phist_final_len)),
            nn.Dropout(dropout_rate),
        )
        #---------------------------------
        # OTHER
        self.fc_other = nn.Sequential(
            nn.Linear(self.other_shape[0], others_unit),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(others_unit),

            *(nn.Sequential(
                nn.Linear(others_unit, others_unit),
                nn.ReLU(inplace=False),
                nn.BatchNorm1d(others_unit)) for _ in range(others_num_hidden)),

            nn.Linear(others_unit, others_unit),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(others_unit),
            nn.Dropout(dropout_rate),
        )  
        #--------------------------------
        # FINAL part
        self.fc_shared = nn.Sequential(
            nn.Linear(ppi_unit2*(10//cnn_final_len)+phist_unit2*(4//cnn_phist_final_len)+others_unit, common_unit),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(common_unit),
            nn.Dropout(dropout_rate),
        )
        self.out = nn.Linear(common_unit, classes_to_predict)

        for layer in self.layers_for_grad:
            setattr(self.fc_shared[layer], "samples_grad", True)
            
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        assert isinstance(self.temperature, torch.Tensor)
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def forward(self, x_ppi, x_phist, x_other, react_threshold=None):
        # ppi
        out_cnn_ppi_pkt = self.cnn_ppi_pkt(x_ppi)
        out_cnn_ppi_pkt = self.cnn_global_pooling(out_cnn_ppi_pkt)
        # phists
        out_cnn_phist_pkt = self.cnn_phist_pkt(x_phist)
        out_cnn_phist_pkt = self.cnn_phist_global_pooling(out_cnn_phist_pkt)
        # other
        out_fc_other = self.fc_other(x_other)
        # combine
        out = torch.column_stack([out_cnn_ppi_pkt, out_cnn_phist_pkt, out_fc_other])
        # final part
        out = self.fc_shared(out)
        
        if react_threshold is not None:
            out = torch.min(out, react_threshold) # threshold per activation
        logits = self.out(out)
        
        if not self.training and self.temperature:
            logits = self.temperature_scale(logits)
            
        return logits