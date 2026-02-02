import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, r3d_18, R2Plus1D_18_Weights, R3D_18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class I3D(nn.Module):
    def __init__(self, num_subjects, num_actions, hidden_dim):
        super(I3D, self).__init__()
        self.num_subjects = num_subjects
        self.num_actions = num_actions
        self.I3D_model = InceptionI3d(num_classes=self.num_subjects, num_actions=self.num_actions, hidden_dim=hidden_dim)
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs, actions, features = self.I3D_model(inputs)
        return outputs, actions, features
        
class R3DBackbone(nn.Module):
    def __init__(self):
        super(R3DBackbone, self).__init__()
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights) 
        self.hidden_dim = 256
        self.R3D_model = model
        self.extractor = create_feature_extractor(model, return_nodes={
            "layer3": "feat3",  
            "layer4": "feat4" 
        })
        
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(256 + 512, 512, kernel_size=1), 
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 14, 14], stride=(1, 1, 1))
        self.name = 'r3d'

       
    def forward(self, inputs):
        features = self.extractor(inputs)
        feat3 = features["feat3"]
        feat4 = features["feat4"]
        feat3_upsampled = F.interpolate(feat3, size=feat4.shape[2:], mode='trilinear', align_corners=False)
        fused_features = torch.cat([feat3_upsampled, feat4], dim=1)
        fused_features = self.feature_fusion(fused_features)
        features = self.avg_pool(fused_features)
        return features    
        
