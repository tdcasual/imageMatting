import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Upsample
import numpy as np
def _layer_norm(channels, eps=1e-5):
    "Helper function to create a LayerNorm layer for a given number of channels."
    return nn.GroupNorm(1, channels, eps=eps)
def compute_max_dilation_rate(output_stride, max_receptive_field_ratio=0.5):
    """
    计算所需的最大膨胀率,以使最大感受野范围覆盖输入特征图的一定比例。

    Args:
        output_stride (int): 主干网络输出特征图相对于输入图像的下采样率。
        max_receptive_field_ratio (float): 期望的最大感受野范围占输入特征图宽高的比例,介于0到1之间。

    Returns:
        int: 所需的最大膨胀率。
    """
    # 假设使用3x3的卷积核
    kernel_size = 3

    # 计算输入特征图的理论分辨率
    stride = 2 ** output_stride
    theoretical_input_resolution = (stride, stride)

    # 计算期望的最大感受野范围
    max_receptive_field = (
        theoretical_input_resolution[0] * max_receptive_field_ratio,
        theoretical_input_resolution[1] * max_receptive_field_ratio
    )

    # 计算所需的最大膨胀率
    max_dilation_rate = int(np.ceil(max(max_receptive_field) / kernel_size))

    return max_dilation_rate

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=False)))

        # 根据输入特征图的分辨率动态计算膨胀率
        max_dilation_rate = compute_max_dilation_rate(output_stride)
        dilation_rates = [6, 12, max_dilation_rate]
        #print('what are ou ')
        #print(dilation_rates)
        for rate in dilation_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                _layer_norm(out_channels),
                nn.ReLU(inplace=False)))

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=False))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d((len(modules) + 1) * out_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=False))
        
        #self.upsample = nn.Upsample(scale_factor=output_stride, mode='bilinear', align_corners=False)

    def forward(self, x):
        #print(x.shape)
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
    
class __ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _layer_norm(out_channels),
            nn.ReLU(inplace=True)))

        # 根据输入特征图的分辨率动态计算膨胀率
        max_dilation_rate = compute_max_dilation_rate(output_stride)
        dilation_rates = [6, 12, max_dilation_rate]
        #print('what are ou ')
        #print(dilation_rates)
        for rate in dilation_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                _layer_norm(out_channels),
                nn.ReLU(inplace=False)))

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
        #print(x.shape)
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
    

class FeatureFusion(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels, enhanced_channels, attended_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.aspp = ASPP(high_channels, out_channels)
        
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
    
class FeatureEnhancementModuleOptimized(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(FeatureEnhancementModuleOptimized, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = in_channels // reduction_ratio

        self.depthwise_separable_conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.ReLU(inplace=True),
            # Pointwise convolution to reduce channel dimensions
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            # Pointwise convolution to restore original channel dimensions
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.depthwise_separable_conv(y)
        return x * y.expand_as(x)


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
        #print(channel_weights.shape)
        spatial_weights = self.spatial_attention(x)
        
        #print(spatial_weights.shape)
        enhanced_features = x * channel_weights * spatial_weights
        return enhanced_features

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1,activation=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        self.activate = nn.ReLU(inplace=True) if not activation else activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activate(x)
        return x

class KeyPointHead(nn.Module):
    def __init__(self, in_channels,mid_channels,num_keypoints=13, num_parts=6, upsample_ratio=4,output_keypoints=True):
        super(KeyPointHead, self).__init__()
        self.num_parts = num_parts
        self.output_keypoints=output_keypoints
        self.num_keypoints=num_keypoints
       
        self.conv_layers = nn.Sequential(
                    DepthwiseSeparableConv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                    DepthwiseSeparableConv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
                    DepthwiseSeparableConv2d(in_channels // 4, num_keypoints, kernel_size=3, padding=1)
                    )
        self.upsample_ratio=upsample_ratio
        #self.channel_matching_conv = nn.Conv2d(adjusted_attention_weights.size()[1], features.size()[1], kernel_size=1)
        self.part_segmentation_head = PartSegmentationHead(in_channels, num_parts)
        #self.feature_enhancement_modules = nn.ModuleList([FeatureEnhancementModule(in_channels) for _ in range(num_parts)])
        #print("in_channels in keypoint",in_channels)
        # 这些错误出现的地方都是通道不匹配
        #这里采用了硬编码
        self.feature_enhancement_modules = nn.ModuleList([FeatureEnhancementModuleOptimized(mid_channels) for _ in range(num_parts)])
        self.upsample = nn.Upsample(scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
        self.org_attention_refine = nn.Sequential(
            nn.Conv2d(num_keypoints, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_refine = nn.Sequential(
            DepthwiseSeparableConv2d(num_keypoints, 64, kernel_size=1, activation=nn.ReLU(inplace=True)),
            DepthwiseSeparableConv2d(64, num_keypoints, kernel_size=1, activation=nn.Sigmoid())
        )
        #另一个和标准的不同的地方。
        #self.channel_matching_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.channel_adjust = nn.Conv2d(num_keypoints, mid_channels, kernel_size=1)
        
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
        #adjusted_attention_weights_matched = self.channel_matching_conv(adjusted_attention_weights)

        # 上采样features以匹配空间维度
        features_upsampled = F.interpolate(features, size=adjusted_attention_weights.size()[2:], mode='bilinear', align_corners=True)
    
        #attended_features = features * adjusted_attention_weights
        
        attended_features = features_upsampled *adjusted_attention_weights
        #print(keypoint_heatmaps.shape)
        #print(upsampled_heatmaps.shape)
        if self.output_keypoints:
            keypoint_coords = self.heatmaps_to_keypoints(keypoint_heatmaps)
        else:
            keypoint_coords = None
        return keypoint_coords, enhanced_features, attended_features
    
    def heatmaps_to_keypoints(self, heatmaps):
        batch_size, _, heatmap_height, heatmap_width = heatmaps.size()
        # 根据原始图像和热图的尺寸动态计算实际上采样比例
        actual_upsample_ratio_width = self.upsample_ratio*8
        actual_upsample_ratio_height = self.upsample_ratio*8
        i=heatmaps.shape[-2:]
        #print(i,heatmaps.shape)

        # 初始化关键点坐标和置信度
        keypoint_coords = torch.zeros(batch_size, self.num_keypoints, 2, device=heatmaps.device)
        keypoint_confidences = torch.zeros(batch_size, self.num_keypoints, device=heatmaps.device)

        for i in range(self.num_keypoints):
            heatmap = heatmaps[:, i, :, :]
            max_vals, max_indices = torch.max(heatmap.view(batch_size, -1), dim=1)
            max_indices = max_indices.float()  # 确保进行浮点数除法
            coords = torch.stack([max_indices % heatmap_width, max_indices // heatmap_width], dim=1)

            # 使用动态计算的实际上采样比例调整坐标
            coords[:, 0] *= actual_upsample_ratio_width
            coords[:, 1] *= actual_upsample_ratio_height

            keypoint_coords[:, i, :] = coords
            keypoint_confidences[:, i] = max_vals

        # 初始化一个新张量以存储坐标和置信度
        keypoints = torch.zeros(batch_size, self.num_keypoints, 3, device=heatmaps.device)
        keypoints[..., :2] = keypoint_coords  # 填充x和y坐标
        keypoints[..., 2] = keypoint_confidences  # 填充置信度

        return keypoints



class MattingHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(MattingHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 =_layer_norm(in_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.sigmoid(self.conv2(x))
        return x

class FiniteNet(nn.Module):
    def __init__(self, num_keypoints=17, num_parts=6, pretrained=True, output_heatmaps=True):
        super(FiniteNet, self).__init__()
        self.mobilenetv3 = models.mobilenet_v3_large(pretrained=pretrained)
        self.features = self.mobilenetv3.features
        
        # Indices for low, mid, and high-level features based on MobileNetV3 structure
        self.low_level_features_index = 1 ##4
        self.mid_level_features_index = 7
        self.high_level_features_index = 14
        self.keypoint_head = KeyPointHead(in_channels=160, mid_channels=80,num_keypoints=num_keypoints, num_parts=num_parts, output_keypoints= output_heatmaps, upsample_ratio=4)  # 假定高级特征层的通道数
        self.feature_fusion = FeatureFusion(low_channels=16, mid_channels=80, high_channels=160, enhanced_channels=80, attended_channels=80, out_channels=160)
        self.matting_head = MattingHead(in_channels=160)  # 假定融合特征有256个通道

        #self.keypoint_head = KeyPointHead(in_channels=160, num_keypoints=num_keypoints, num_parts=num_parts)
        #self.feature_fusion = FeatureFusion(low_channels=24, mid_channels=80, high_channels=160, enhanced_channels=80, attended_channels=80, out_channels=256)
        #self.matting_head = MattingHead(in_channels=256)  # Assuming fused features have 256 channels
        
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
        if self.keypoint_head.output_keypoints:
            upsampled_heatmaps, enhanced_features, attended_features = self.keypoint_head(high_level_feat, mid_level_feat)
        else:
            enhanced_features, attended_features = self.keypoint_head(high_level_feat, mid_level_feat)
        
        fused_features = self.feature_fusion(low_level_feat, mid_level_feat, high_level_feat, enhanced_features, attended_features)
        pre_matte = self.matting_head(fused_features)
        pre_matte = F.interpolate(pre_matte, scale_factor=2, mode='bilinear', align_corners=False)
        
        if self.keypoint_head.output_keypoints:
            return None, upsampled_heatmaps,pre_matte
        else:
            return None,None,pre_matte


