import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Upsample

def _layer_norm(channels, eps=1e-5):
    "Helper function to create a LayerNorm layer for a given number of channels."
    return nn.GroupNorm(1, channels, eps=eps)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, output_stride=16):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True)))

        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                _layer_norm(out_channels),
                nn.ReLU(inplace=True)))

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d((len(modules) + 1) * out_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True))
        
        #self.upsample = nn.Upsample(scale_factor=output_stride, mode='bilinear', align_corners=False)

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        size = x.shape[-2:]
        global_feat = self.global_avg_pool(x)
        scale_factor = [size[0] / global_feat.size(2), size[1] / global_feat.size(3)]
        global_feat = F.interpolate(global_feat, scale_factor=scale_factor, mode='bilinear', align_corners=False)
       # global_feat = self.upsample(global_feat)
        
        res.append(global_feat)
        #print([i.shape for i in res])
        res = torch.cat(res, dim=1)
        return self.project(res)
    
class AASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.atrous_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False)
            for rate in atrous_rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        # 添加一个类似self.project的模块来整合和减少通道数
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) + 2 * out_channels, out_channels, 1, bias=False),  # 注意调整输入通道数
            nn.LayerNorm([out_channels, 1, 1]),  # 根据需要调整LayerNorm的参数
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        global_feat = self.global_avg_pool(x)
        # 动态计算scale_factor
        scale_factor = [size[0] / global_feat.size(2), size[1] / global_feat.size(3)]
        global_feat_upsampled = F.interpolate(global_feat, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        atrous_feats = [conv(x) for conv in self.atrous_convs]
        atrous_feats.append(global_feat_upsampled)
        atrous_feats.append(x)  # 假设还想加入原始特征
        #print([i.shape for i in atrous_feats])
        combined = torch.cat(atrous_feats, dim=1)
        
        return self.project(combined)



class FeatureFusion(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels, enhanced_channels, attended_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.aspp = ASPP(high_channels, out_channels, atrous_rates=[6, 12, 18])
        
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv_mid = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv_enhanced = nn.Sequential(
            nn.Conv2d(enhanced_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv_attended = nn.Sequential(
            nn.Conv2d(attended_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, low_feat, mid_feat, high_feat, enhanced_feat, attended_feat):
        high_feat = self.aspp(high_feat)
        
        low_feat = self.conv_low(low_feat)
        mid_feat = self.conv_mid(mid_feat)
        enhanced_feat = self.conv_enhanced(enhanced_feat)
        attended_feat = self.conv_attended(attended_feat)
        
        target_size = low_feat.size()[2:]
        high_feat = F.interpolate(high_feat, size=target_size, mode='bilinear', align_corners=True)
        mid_feat = F.interpolate(mid_feat, size=target_size, mode='bilinear', align_corners=True)
        enhanced_feat = F.interpolate(enhanced_feat, size=target_size, mode='bilinear', align_corners=True)
        attended_feat = F.interpolate(attended_feat, size=target_size, mode='bilinear', align_corners=True)
        #print(low_feat.shape,mid_feat.shape, high_feat.shape,enhanced_feat.shape,attended_feat.shape)
        feat = low_feat + mid_feat + high_feat + enhanced_feat + attended_feat
        return feat

# Other modules like FeatureEnhancementModule, PartSegmentationHead, KeyPointHead, MattingHead, and FinteNet
# remain the same in structure but replace BatchNorm2d with LayerNorm using _layer_norm function where applicable.

    
class PartSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(PartSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, num_parts, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    
class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureEnhancementModule, self).__init__()
        self.in_channels=in_channels
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print("feautre ",x.shape)
        #print("channels=",self.in_channels)
        channel_weights = self.channel_attention(x)
        spatial_weights = self.spatial_attention(x)
        #print(channel_weights.shape)
        #print(spatial_weights.shape)
        enhanced_features = x * channel_weights * spatial_weights
        return enhanced_features

class KeyPointHead(nn.Module):
    def __init__(self, in_channels, num_keypoints=13, num_parts=6, upsample_ratio=4):
        super(KeyPointHead, self).__init__()
        self.num_parts = num_parts
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, kernel_size=1)
        )
        #self.channel_matching_conv = nn.Conv2d(adjusted_attention_weights.size()[1], features.size()[1], kernel_size=1)
        self.part_segmentation_head = PartSegmentationHead(in_channels, num_parts)
        #self.feature_enhancement_modules = nn.ModuleList([FeatureEnhancementModule(in_channels) for _ in range(num_parts)])
        #print("in_channels in keypoint",in_channels)
        # 这些错误出现的地方都是通道不匹配
        self.feature_enhancement_modules = nn.ModuleList([FeatureEnhancementModule(in_channels//2) for _ in range(num_parts)])
        self.upsample = nn.Upsample(scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
        self.attention_refine = nn.Sequential(
            nn.Conv2d(num_keypoints, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_matching_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.channel_adjust = nn.Conv2d(num_keypoints, in_channels, kernel_size=1)
        
    def forward(self, x, features):
        keypoint_heatmaps = self.conv_layers(x)
        part_segmentation = self.part_segmentation_head(x)
        
        part_masks = torch.argmax(part_segmentation, dim=1)
        enhanced_features = torch.zeros_like(features)
        
        for i in range(self.num_parts):
            part_mask = (part_masks == i).unsqueeze(1).float()
            upsample_layer = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            part_mask_upsampled = upsample_layer(part_mask)
            #print(features.shape,part_mask_upsampled.shape)
            part_features = features * part_mask_upsampled
            #print("part future is ",part_features.shape)
            enhanced_part_features = self.feature_enhancement_modules[i](part_features)
            #print("enhanced_part_features is ",enhanced_part_features.shape)
            enhanced_features += enhanced_part_features
        
        upsampled_heatmaps = self.upsample(keypoint_heatmaps)
        attention_weights = self.attention_refine(upsampled_heatmaps)
        adjusted_attention_weights = self.channel_adjust(attention_weights)
        #print("final",features.shape,adjusted_attention_weights.shape)
        # 假设 features 和 adjusted_attention_weights 已经定义并且有上述尺寸

        # 1x1卷积以匹配通道数
        
        #channel_matching_conv = nn.Conv2d(adjusted_attention_weights.size()[1], features.size()[1], kernel_size=1)
        # 确定设备
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #@!!!!!
        # 将卷积层移动到设备上
        #channel_matching_conv = channel_matching_conv.to(device)

        # 现在，你可以安全地对位于GPU上的张量进行操作，不会出现设备不匹配的错误

        adjusted_attention_weights_matched = self.channel_matching_conv(adjusted_attention_weights)

        # 上采样features以匹配空间维度
        features_upsampled = F.interpolate(features, size=adjusted_attention_weights_matched.size()[2:], mode='bilinear', align_corners=True)
    
        #attended_features = features * adjusted_attention_weights
        
        attended_features = features_upsampled * adjusted_attention_weights_matched
        return upsampled_heatmaps, enhanced_features, attended_features


class MattingHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(MattingHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.sigmoid(self.conv2(x))
        return x

class FiniteNet(nn.Module):
    def __init__(self, num_keypoints=13, num_parts=6, pretrained=True):
        super(FiniteNet, self).__init__()
        self.mobilenetv3 = models.mobilenet_v3_large(pretrained=pretrained)
        self.features = self.mobilenetv3.features
        
        # Indices for low, mid, and high-level features based on MobileNetV3 structure
        self.low_level_features_index = 3 ##4
        self.mid_level_features_index = 7
        self.high_level_features_index = 14
        
        self.keypoint_head = KeyPointHead(in_channels=160, num_keypoints=num_keypoints, num_parts=num_parts)
        self.feature_fusion = FeatureFusion(low_channels=24, mid_channels=80, high_channels=160, enhanced_channels=80, attended_channels=80, out_channels=256)
        self.matting_head = MattingHead(in_channels=256)  # Assuming fused features have 256 channels
        
    def forward(self, x, reference=True):
        low_level_feat, mid_level_feat, high_level_feat = None, None, None
        
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i == self.low_level_features_index:
                low_level_feat = x
            elif i == self.mid_level_features_index:
                mid_level_feat = x
            elif i == self.high_level_features_index:
                high_level_feat = x
        #print(low_level_feat.shape,mid_level_feat.shape,high_level_feat.shape)
        upsampled_heatmaps, enhanced_features, attended_features = self.keypoint_head(high_level_feat, mid_level_feat)
        fused_features = self.feature_fusion(low_level_feat, mid_level_feat, high_level_feat, enhanced_features, attended_features)
        
        pre_matte = self.matting_head(fused_features)
        
        # Only return the pre_matte and upsampled_heatmaps as specified
        pre_matte = F.interpolate(pre_matte, scale_factor=4, mode='bilinear', align_corners=False)
        return pre_matte, upsampled_heatmaps, pre_matte


