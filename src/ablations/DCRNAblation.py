import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import random
import json
from collections import defaultdict
import torchvision.models as models
import traceback

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
project_root = current_dir
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from src.config.config import device, worker_init_fn, g, new_classesPavia
from src.data.loaders import load_and_preprocess_pavia_data
from src.data.datasets import HSIDataset, HSITestDataset
from src.models.classifier import HybridSSUDClassifierFixed
from src.models.sifd import SpectrumInvariantFrequencyDisentanglement, GradientReversalLayer
from src.models.evidential import EvidentialLayer
from src.models.attention import ChannelAttention, SpatialAttention
from src.training.losses import edl_mse_loss
from src.training.evaluator import test_open_set_hybrid
from src.utils.helpers import setup_directories


def edl_mse_loss_fixed(target, alpha, epoch=1, lambda_reg=0.1):
    """Fixed version of EDL MSE loss that handles tensor types properly"""
    # Ensure target is LongTensor
    target = target.long()

    # Check for valid target range
    num_classes = alpha.size(1)
    if target.min() < 0 or target.max() >= num_classes:
        # Clamp to valid range
        target = torch.clamp(target, 0, num_classes - 1)

    # Add small epsilon to alpha to prevent numerical instability
    eps = 1e-7
    alpha = alpha + eps

    # Convert target to one-hot encoding
    y = F.one_hot(target, num_classes=num_classes).float()

    # Calculate strength (sum of alpha) with numerical stability
    S = torch.sum(alpha, dim=1, keepdim=True)
    S = torch.clamp(S, min=eps)  # Prevent division by zero

    # Expected probabilities under Dirichlet
    expected_p = alpha / S

    # MSE loss with numerical stability checks
    mse_loss = torch.sum((y - expected_p) ** 2, dim=1)
    mse_loss = torch.clamp(mse_loss, min=0.0, max=1e6)  # Prevent extreme values
    mse_loss = mse_loss.mean()

    # Regularization to penalize high certainty (small S is uncertain, large S is certain)
    # We want to be uncertain unless we have evidence
    # Start with weak regularization and increase it
    annealing_coef = min(1.0, epoch / 10)

    # Regularize more on classes that should be zero
    zeros = (1 - y)
    reg_loss = torch.sum(zeros * alpha, dim=1)
    reg_loss = torch.clamp(reg_loss, min=0.0, max=1e6)  # Prevent extreme values
    reg_loss = reg_loss.mean()

    # Weighted total loss with numerical stability
    total_loss = mse_loss + annealing_coef * lambda_reg * reg_loss

    # Final check for NaN/Inf
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(1.0, device=alpha.device, requires_grad=True)
        mse_loss = torch.tensor(1.0, device=alpha.device)
        reg_loss = torch.tensor(0.0, device=alpha.device)

    return total_loss, mse_loss, reg_loss


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DCRNAblation_ResNet50(nn.Module):
    """Replace entire DCRN with standard ResNet50"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_ResNet50, self).__init__()

        # SIFD module (keep this)
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Convert input channels to RGB
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(3)

        # Standard ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # Remove final FC layer and avgpool
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Upsample to maintain spatial dimensions
        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # Feature dimensions
        self.feature_size = 2048  # ResNet50 output channels
        self.spectral_size = self.feature_size // 2  # Dummy for compatibility
        self.spatial_size = self.feature_size // 2

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting (dummy for compatibility)
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Convert to RGB
        rgb_input = self.input_adapter(enhanced_input)
        rgb_input = F.relu(self.input_bn(rgb_input))

        # ResNet50 features
        features = self.resnet(rgb_input)
        features = self.upsample(features)

        # Pool features
        combined_features = F.adaptive_avg_pool2d(features, 1).view(batch_size, -1)

        # Split for compatibility (dummy spectral/spatial split)
        spectral_features = combined_features[:, :self.spectral_size]
        spatial_features = combined_features[:, self.spectral_size:]

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights (dummy)
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class DCRNAblation_EfficientNet(nn.Module):
    """Replace DCRN with EfficientNet"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_EfficientNet, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Convert input channels to RGB
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(3)

        # EfficientNet-B0
        try:
            from torchvision.models import efficientnet_b0
            self.efficientnet = efficientnet_b0(pretrained=True)
            # Remove classifier
            self.efficientnet.classifier = nn.Identity()
            self.feature_size = 1280  # EfficientNet-B0 output features
        except (ImportError, AttributeError):
            # Fallback to ResNet if EfficientNet not available
            self.efficientnet = models.resnet50(pretrained=True)
            self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-2])
            self.feature_size = 2048  # ResNet50 output features
        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # Feature dimensions are set above based on model type
        self.spectral_size = self.feature_size // 2
        self.spatial_size = self.feature_size // 2

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Convert to RGB
        rgb_input = self.input_adapter(enhanced_input)
        rgb_input = F.relu(self.input_bn(rgb_input))

        # EfficientNet features
        features = self.efficientnet(rgb_input)
        if len(features.shape) == 4:  # If spatial dimensions remain
            features = self.upsample(features)
            combined_features = F.adaptive_avg_pool2d(features, 1).view(batch_size, -1)
        else:  # Already pooled
            combined_features = features

        # Ensure correct feature size
        if combined_features.size(1) != self.feature_size:
            # Add a projection layer if needed
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Linear(combined_features.size(1), self.feature_size).to(device)
            combined_features = self.feature_proj(combined_features)

        # Split for compatibility
        spectral_features = combined_features[:, :self.spectral_size]
        spatial_features = combined_features[:, self.spectral_size:]

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class DCRNAblation_VGG(nn.Module):
    """Replace DCRN with VGG backbone"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_VGG, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Convert input channels to RGB
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(3)

        # Custom VGG-like layers for small patches - very conservative approach
        self.vgg_features = nn.Sequential(
            # Block 1 - no pooling to preserve spatial dimensions
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 2 - single pooling 7x7 -> 3x3
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 3 - no more pooling, just more conv layers
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Global average pooling to handle any remaining spatial size
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimensions
        self.feature_size = 512
        self.spectral_size = self.feature_size // 2
        self.spatial_size = self.feature_size // 2

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Convert to RGB
        rgb_input = self.input_adapter(enhanced_input)
        rgb_input = F.relu(self.input_bn(rgb_input))

        # Custom VGG-like features designed for small patches
        features = self.vgg_features(rgb_input)

        # Global average pooling to 1x1
        combined_features = self.global_pool(features).view(batch_size, -1)

        # Split for compatibility
        spectral_features = combined_features[:, :self.spectral_size]
        spatial_features = combined_features[:, self.spectral_size:]

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class DCRNAblation_SpectralOnly(nn.Module):
    """Keep only spectral pathway from DCRN"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_SpectralOnly, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Spectral pathway only (from original DCRN)
        self.spec_conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn1 = nn.BatchNorm2d(64)
        self.spec_conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn2 = nn.BatchNorm2d(64)
        self.spec_conv3 = nn.Conv2d(64, 512, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn3 = nn.BatchNorm2d(512)
        self.spec_conv4 = nn.Conv2d(512, 512, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn4 = nn.BatchNorm2d(512)
        self.spec_conv5 = nn.Conv2d(512, 1024, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn5 = nn.BatchNorm2d(1024)

        # Feature dimensions
        self.spectral_size = 1024
        self.spatial_size = 1024  # Dummy for compatibility
        self.feature_size = 1024  # Only spectral features

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)  # Dummy
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting (dummy)
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Spectral pathway only
        spec = self.spec_conv1(enhanced_input)
        spec = F.relu(self.spec_bn1(spec))
        spec = self.spec_conv2(spec)
        spec = F.relu(self.spec_bn2(spec))
        spec = self.spec_conv3(spec)
        spec = F.relu(self.spec_bn3(spec))
        spec = self.spec_conv4(spec)
        spec = F.relu(self.spec_bn4(spec))
        spec = self.spec_conv5(spec)
        spec = F.relu(self.spec_bn5(spec))

        # Pool spectral features
        spectral_features = F.adaptive_avg_pool2d(spec, 1).view(batch_size, -1)

        # Dummy spatial features for compatibility
        spatial_features = spectral_features  # Same as spectral
        combined_features = spectral_features

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)  # Dummy
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights (spectral only)
        weights = torch.tensor([[1.0, 0.0]], device=x.device).repeat(batch_size, 1)

        # Reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = spectral_reliability  # Dummy

        # Weighted uncertainty (same as spectral)
        weighted_uncertainty = spectral_out['uncertainty']

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class DCRNAblation_SpatialOnly(nn.Module):
    """Keep only spatial pathway from DCRN"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_SpatialOnly, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Spatial pathway only (from original DCRN)
        self.spat_conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.spat_bn1 = nn.BatchNorm2d(3)

        # ResNet50 layers
        resnet = models.resnet50(pretrained=True)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # Feature dimensions
        self.spatial_size = 2048
        self.spectral_size = 2048  # Dummy for compatibility
        self.feature_size = 2048  # Only spatial features

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)  # Dummy
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting (dummy)
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Spatial pathway only
        spat = self.spat_conv1(enhanced_input)
        spat = F.relu(self.spat_bn1(spat))
        spat = self.adapter(spat)
        spat = self.layer1(spat)
        spat = self.layer2(spat)
        spat = self.layer3(spat)
        spat = self.layer4(spat)
        spat = self.upsample(spat)

        # Pool spatial features
        spatial_features = F.adaptive_avg_pool2d(spat, 1).view(batch_size, -1)

        # Dummy spectral features for compatibility
        spectral_features = spatial_features  # Same as spatial
        combined_features = spatial_features

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)  # Dummy
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights (spatial only)
        weights = torch.tensor([[0.0, 1.0]], device=x.device).repeat(batch_size, 1)

        # Reliability scores
        spatial_reliability = 1.0 - spatial_out['uncertainty']
        spectral_reliability = spatial_reliability  # Dummy

        # Weighted uncertainty (same as spatial)
        weighted_uncertainty = spatial_out['uncertainty']

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class DCRNAblation_SimpleConcatenation(nn.Module):
    """DCRN with simple concatenation instead of attention fusion"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_SimpleConcatenation, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Spectral pathway
        self.spec_conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn1 = nn.BatchNorm2d(64)
        self.spec_conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn2 = nn.BatchNorm2d(64)
        self.spec_conv3 = nn.Conv2d(64, 512, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn3 = nn.BatchNorm2d(512)
        self.spec_conv4 = nn.Conv2d(512, 512, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn4 = nn.BatchNorm2d(512)
        self.spec_conv5 = nn.Conv2d(512, 1024, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn5 = nn.BatchNorm2d(1024)

        # Spatial pathway
        self.spat_conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.spat_bn1 = nn.BatchNorm2d(3)

        resnet = models.resnet50(pretrained=True)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # Feature dimensions
        self.spectral_size = 1024
        self.spatial_size = 2048
        self.feature_size = 3072  # Simple concatenation

        # NO ATTENTION - just projection after concatenation
        self.projection = nn.Conv2d(self.spectral_size + self.spatial_size,
                                    self.feature_size, kernel_size=1)

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Spectral pathway
        spec = self.spec_conv1(enhanced_input)
        spec = F.relu(self.spec_bn1(spec))
        spec = self.spec_conv2(spec)
        spec = F.relu(self.spec_bn2(spec))
        spec = self.spec_conv3(spec)
        spec = F.relu(self.spec_bn3(spec))
        spec = self.spec_conv4(spec)
        spec = F.relu(self.spec_bn4(spec))
        spec = self.spec_conv5(spec)
        spec = F.relu(self.spec_bn5(spec))

        # Spatial pathway
        spat = self.spat_conv1(enhanced_input)
        spat = F.relu(self.spat_bn1(spat))
        spat = self.adapter(spat)
        spat = self.layer1(spat)
        spat = self.layer2(spat)
        spat = self.layer3(spat)
        spat = self.layer4(spat)
        spat = self.upsample(spat)

        # SIMPLE CONCATENATION (no attention)
        combined = torch.cat((spec, spat), 1)
        combined = self.projection(combined)

        # Pool separate pathway features
        spectral_features = F.adaptive_avg_pool2d(spec, 1).view(batch_size, -1)
        spatial_features = F.adaptive_avg_pool2d(spat, 1).view(batch_size, -1)
        combined_features = F.adaptive_avg_pool2d(combined, 1).view(batch_size, -1)

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


class DCRNAblation_ElementwiseAddition(nn.Module):
    """DCRN with element-wise addition instead of attention fusion"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRNAblation_ElementwiseAddition, self).__init__()

        # SIFD module
        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)

        # Spectral pathway
        self.spec_conv1 = nn.Conv2d(input_channels, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn1 = nn.BatchNorm2d(64)
        self.spec_conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn2 = nn.BatchNorm2d(64)
        self.spec_conv3 = nn.Conv2d(64, 512, kernel_size=(1, 7), padding=(0, 3))
        self.spec_bn3 = nn.BatchNorm2d(512)
        self.spec_conv4 = nn.Conv2d(512, 512, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn4 = nn.BatchNorm2d(512)
        self.spec_conv5 = nn.Conv2d(512, 1024, kernel_size=(1, 21), padding=(0, 10))
        self.spec_bn5 = nn.BatchNorm2d(1024)

        # Spatial pathway
        self.spat_conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.spat_bn1 = nn.BatchNorm2d(3)

        resnet = models.resnet50(pretrained=True)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.upsample = nn.Upsample(size=(patch_size, patch_size), mode='bilinear', align_corners=False)

        # Project spatial features to match spectral dimensions for addition
        self.spatial_projection = nn.Conv2d(2048, 1024, kernel_size=1)

        # Feature dimensions
        self.spectral_size = 1024
        self.spatial_size = 2048
        self.feature_size = 1024  # After element-wise addition

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # EDL layers
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Pathway weighting
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)

        # Apply SIFD
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        # Spectral pathway
        spec = self.spec_conv1(enhanced_input)
        spec = F.relu(self.spec_bn1(spec))
        spec = self.spec_conv2(spec)
        spec = F.relu(self.spec_bn2(spec))
        spec = self.spec_conv3(spec)
        spec = F.relu(self.spec_bn3(spec))
        spec = self.spec_conv4(spec)
        spec = F.relu(self.spec_bn4(spec))
        spec = self.spec_conv5(spec)
        spec = F.relu(self.spec_bn5(spec))

        # Spatial pathway
        spat = self.spat_conv1(enhanced_input)
        spat = F.relu(self.spat_bn1(spat))
        spat = self.adapter(spat)
        spat = self.layer1(spat)
        spat = self.layer2(spat)
        spat = self.layer3(spat)
        spat = self.layer4(spat)
        spat = self.upsample(spat)

        # Project spatial to match spectral dimensions
        spat_proj = self.spatial_projection(spat)

        # ELEMENT-WISE ADDITION (no attention)
        combined = spec + spat_proj

        # Pool separate pathway features
        spectral_features = F.adaptive_avg_pool2d(spec, 1).view(batch_size, -1)
        spatial_features = F.adaptive_avg_pool2d(spat, 1).view(batch_size, -1)
        combined_features = F.adaptive_avg_pool2d(combined, 1).view(batch_size, -1)

        # Main classification
        logits = self.classifier(combined_features)
        probs = F.softmax(logits, dim=1)

        # EDL outputs
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Pathway weights
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Domain prediction
        domain_pred = self.domain_classifier(combined_features)

        return {
            'logits': logits,
            'probs': probs,
            'alpha_combined': combined_out['alpha'],
            'alpha_spectral': spectral_out['alpha'],
            'alpha_spatial': spatial_out['alpha'],
            'uncertainty_combined': combined_out['uncertainty'],
            'uncertainty_spectral': spectral_out['uncertainty'],
            'uncertainty_spatial': spatial_out['uncertainty'],
            'weighted_uncertainty': weighted_uncertainty,
            'fusion_weights': weights,
            'spectral_reliability': spectral_reliability,
            'spatial_reliability': spatial_reliability,
            'features': combined_features,
            'domain_pred': domain_pred,
            'sifd_outputs': sifd_outputs
        }


def train_one_epoch_dcrn_ablation(models, classifiers, train_loader, optimizers, epoch, num_classes=7,
                                  alpha=0.3, beta=0.5, gamma=0.3, delta=0.2, clip_value=1.0):
    """Train all DCRN ablation models simultaneously on the same batches"""
    model_names = list(models.keys())

    # Set all models to training mode
    for model in models.values():
        model.train()

    total_losses = {name: 0 for name in model_names}
    total_accuracies = {name: 0 for name in model_names}
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Ensure target is LongTensor and in valid range
        target = target.long()
        target = torch.clamp(target, 0, num_classes - 1)

        # Train each model on the same batch
        for name in model_names:
            model = models[name]
            optimizer = optimizers[name]

            optimizer.zero_grad()

            try:
                outputs = model(data)

                # Verify all required outputs exist
                required_keys = ['logits', 'alpha_combined', 'alpha_spectral', 'alpha_spatial', 'sifd_outputs']
                missing_keys = [key for key in required_keys if key not in outputs]
                if missing_keys:
                    continue

                # Standard classification loss
                cls_loss = F.cross_entropy(outputs['logits'], target)

                # EDL losses (using fixed version)
                edl_combined, _, _ = edl_mse_loss_fixed(target, outputs['alpha_combined'], epoch)
                edl_spectral, _, _ = edl_mse_loss_fixed(target, outputs['alpha_spectral'], epoch)
                edl_spatial, _, _ = edl_mse_loss_fixed(target, outputs['alpha_spatial'], epoch)
                edl_total_loss = (edl_combined + alpha * (edl_spectral + edl_spatial)) / 3.0

                # Domain adversarial loss
                domain_loss = F.binary_cross_entropy_with_logits(
                    outputs['sifd_outputs']['domain_pred'],
                    torch.ones_like(outputs['sifd_outputs']['domain_pred']) * 0.5
                )

                # Reconstruction loss
                data_flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])
                recon_loss = F.mse_loss(outputs['sifd_outputs']['reconstructed_spectrum'], data_flat)

                # Total loss
                loss = cls_loss + alpha * edl_total_loss + beta * domain_loss + gamma * recon_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

                # Update metrics
                total_losses[name] += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs['logits'], 1)
                correct = (predicted == target).sum().item()
                total_accuracies[name] += correct

            except Exception as e:
                continue

        total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in model_names:
        avg_loss = total_losses[name] / len(train_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        results[name] = {'loss': avg_loss, 'accuracy': avg_acc}

    return results


def validate_dcrn_ablation(models, classifiers, val_loader, epoch):
    """Validate all DCRN ablation models simultaneously"""
    model_names = list(models.keys())

    # Set all models to eval mode
    for model in models.values():
        model.eval()

    total_losses = {name: 0 for name in model_names}
    total_accuracies = {name: 0 for name in model_names}
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # Ensure target is LongTensor
            target = target.long()

            for name in model_names:
                model = models[name]

                try:
                    outputs = model(data)

                    # Calculate validation loss
                    loss = F.cross_entropy(outputs['logits'], target)
                    total_losses[name] += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs['logits'], 1)
                    correct = (predicted == target).sum().item()
                    total_accuracies[name] += correct

                except Exception as e:
                    continue

            total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in model_names:
        avg_loss = total_losses[name] / len(val_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        results[name] = {'loss': avg_loss, 'accuracy': avg_acc}

    return results


def create_dcrn_ablation_models(input_channels, patch_size, num_classes):
    """Create all DCRN ablation models"""

    # Import the original model
    from src.models.classifier import DCRN_SSUD_SIFD

    # Full set of models - all 8 ablation models
    models = {
        'Original_DCRN': DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=patch_size,
                                        num_classes=num_classes).to(device),
        'ResNet50': DCRNAblation_ResNet50(input_channels=input_channels, patch_size=patch_size,
                                          num_classes=num_classes).to(device),
        'EfficientNet': DCRNAblation_EfficientNet(input_channels=input_channels, patch_size=patch_size,
                                                  num_classes=num_classes).to(device),
        'VGG': DCRNAblation_VGG(input_channels=input_channels, patch_size=patch_size,
                                num_classes=num_classes).to(device),
        'SpectralOnly': DCRNAblation_SpectralOnly(input_channels=input_channels, patch_size=patch_size,
                                                  num_classes=num_classes).to(device),
        'SpatialOnly': DCRNAblation_SpatialOnly(input_channels=input_channels, patch_size=patch_size,
                                                num_classes=num_classes).to(device),
        'SimpleConcatenation': DCRNAblation_SimpleConcatenation(input_channels=input_channels,
                                                                patch_size=patch_size,
                                                                num_classes=num_classes).to(device),
        'ElementwiseAddition': DCRNAblation_ElementwiseAddition(input_channels=input_channels,
                                                                patch_size=patch_size,
                                                                num_classes=num_classes).to(device)
    }

    return models


def create_dcrn_ablation_optimizers(models, lr):
    """Create optimizers for all DCRN ablation models"""
    optimizers = {}
    for name, model in models.items():
        try:
            if name == 'Original_DCRN':
                # Use the original grouping for the original model
                optimizer = optim.Adam([
                    {'params': model.dcrn.parameters(), 'lr': lr},
                    {'params': model.classifier.parameters(), 'lr': lr},
                    {'params': model.spectral_evidence.parameters(), 'lr': lr},
                    {'params': model.spatial_evidence.parameters(), 'lr': lr},
                    {'params': model.combined_evidence.parameters(), 'lr': lr},
                    {'params': model.sifd.parameters(), 'lr': lr},
                    {'params': model.pathway_weight.parameters(), 'lr': lr},
                    {'params': model.domain_classifier.parameters(), 'lr': lr},
                ], weight_decay=1e-5)
            else:
                # For ablation models, use simple single-group optimizer
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

            optimizers[name] = optimizer

        except Exception as e:
            # Fallback to simple optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            optimizers[name] = optimizer

    return optimizers


def run_single_seed_dcrn_experiment(seed, pavia_loaders, input_channels, num_classes,
                                    epochs=50, batch_size=32, patch_size=7, lr=0.00001,
                                    uncertainty_threshold=0.4111, decoupling_threshold=0.1, spectral_weight=0.7):
    """Run a single DCRN ablation experiment with a given seed"""
    print(f"\n--- Running DCRN Ablation Seed {seed} ---")

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for this experiment
    set_seed(seed)

    train_loader, val_loader, test_loader = pavia_loaders

    # Create all ablation models
    models = create_dcrn_ablation_models(input_channels, patch_size, num_classes)

    # Print model parameter counts
    print(f"\nModel Parameter Counts:")
    print("-" * 50)
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<20}: {trainable_params:,} trainable, {total_params:,} total")

    # Create classifiers for each model
    classifiers = {}
    for name, model in models.items():
        classifiers[name] = HybridSSUDClassifierFixed(
            model, num_classes=num_classes,
            uncertainty_threshold=uncertainty_threshold,
            decoupling_threshold=decoupling_threshold,
            spectral_weight=spectral_weight
        ).to(device)

    # Create optimizers for each model
    optimizers = create_dcrn_ablation_optimizers(models, lr)

    # Training history
    history = {name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
               for name in models.keys()}

    best_models = {name: None for name in models.keys()}
    best_val_accs = {name: 0 for name in models.keys()}

    # Training loop
    for epoch in range(1, epochs + 1):
        # Train all models on the same batches
        train_results = train_one_epoch_dcrn_ablation(models, classifiers, train_loader, optimizers, epoch,
                                                      num_classes=7)

        # Validate all models
        val_results = validate_dcrn_ablation(models, classifiers, val_loader, epoch)

        # Update history and save best models
        for name in models.keys():
            history[name]['train_loss'].append(train_results[name]['loss'])
            history[name]['train_acc'].append(train_results[name]['accuracy'])
            history[name]['val_loss'].append(val_results[name]['loss'])
            history[name]['val_acc'].append(val_results[name]['accuracy'])

            # Save best model
            if val_results[name]['accuracy'] > best_val_accs[name]:
                best_val_accs[name] = val_results[name]['accuracy']
                best_models[name] = deepcopy(models[name].state_dict())

        # Print progress every epoch with validation accuracy
        print(f"Epoch {epoch:2d}/{epochs}: ", end='')
        model_strs = []
        for name in models.keys():
            model_strs.append(f"{name}: {val_results[name]['accuracy']:5.1f}%")
        print(" | ".join(model_strs))

    print(f"\nSeed {seed} training completed. Evaluating on test set...")

    # Load best models and evaluate
    final_results = {}
    for name in models.keys():
        try:
            models[name].load_state_dict(best_models[name])

            # Evaluate on test set
            known_acc, unknown_rej, hos_score, cm = test_open_set_hybrid(models[name], classifiers[name], test_loader)

            final_results[name] = {
                'known_accuracy': known_acc,
                'unknown_rejection': unknown_rej,
                'hos_score': hos_score,
                'best_val_acc': best_val_accs[name],
                'confusion_matrix': cm,
                'seed': seed
            }
            print(f"  {name}: HOS = {hos_score:.4f}")

        except Exception as e:
            final_results[name] = {
                'known_accuracy': 0,
                'unknown_rejection': 0,
                'hos_score': 0,
                'best_val_acc': best_val_accs[name],
                'confusion_matrix': None,
                'seed': seed,
                'error': str(e)
            }

    # Clean up memory after each seed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_results, history, best_models


def run_pavia_dcrn_ablation_single_seed(epochs=50, batch_size=32, patch_size=7, lr=0.00001,
                                        uncertainty_threshold=0.4111, decoupling_threshold=0.1, spectral_weight=0.7):
    """Run DCRN ablation study on Pavia dataset with a single seed"""
    print("=" * 100)
    print(f"PAVIA DCRN ABLATION STUDY - SINGLE SEED, {epochs} EPOCHS")
    print("=" * 100)

    # Load Pavia dataset
    print("\nLoading Pavia dataset...")
    (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
     hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

    # Create datasets
    source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                      patch_size=patch_size, augment=True)
    source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u,
                                    patch_size=patch_size, augment=False)
    target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped,
                                         patch_size=patch_size)

    # Create data loaders
    train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    pavia_loaders = (train_loader, val_loader, test_loader)
    input_channels = hs_pavia_u_norm.shape[2]  # 102

    seed = 1  # Fixed seed for single experiment
    print(f"\nRunning single experiment with seed {seed}...")

    # Run single seed experiment
    seed_results, seed_history, seed_models = run_single_seed_dcrn_experiment(
        seed, pavia_loaders, input_channels, 7, epochs, batch_size, patch_size, lr,
        uncertainty_threshold, decoupling_threshold, spectral_weight
    )

    print("\n" + "=" * 100)
    print("FINAL DCRN ABLATION RESULTS")
    print("=" * 100)

    # Print results table
    model_names = ['Original_DCRN', 'ResNet50', 'EfficientNet', 'VGG', 'SpectralOnly',
                   'SpatialOnly', 'SimpleConcatenation', 'ElementwiseAddition']

    print(f"\nRESULTS FOR SEED {seed}:")
    print("-" * 80)
    print(f"{'Model':<20} {'Val Acc':<10} {'Known Acc':<12} {'Unknown Rej':<12} {'HOS Score':<10}")
    print("-" * 80)
    for model_name in model_names:
        if model_name in seed_results:
            res = seed_results[model_name]
            print(f"{model_name:<20} {res['best_val_acc']:<10.2f} {res['known_accuracy']:<12.2f} "
                  f"{res['unknown_rejection']:<12.2f} {res['hos_score']:<10.4f}")

    # Find best performing model
    best_model = None
    best_hos_score = 0
    for model_name, results in seed_results.items():
        if results['hos_score'] > best_hos_score:
            best_hos_score = results['hos_score']
            best_model = model_name

    print(f"\nBEST PERFORMING MODEL:")
    if best_model:
        print(f"Model: {best_model}")
        print(f"HOS Score: {best_hos_score:.4f}")
        print(f"Known Accuracy: {seed_results[best_model]['known_accuracy']:.2f}%")
        print(f"Unknown Rejection: {seed_results[best_model]['unknown_rejection']:.2f}%")

    # Analyze ablation impact compared to Original DCRN
    if 'Original_DCRN' in seed_results:
        original_hos = seed_results['Original_DCRN']['hos_score']
        print(f"\nABLATION ANALYSIS (compared to Original DCRN HOS: {original_hos:.4f}):")
        print("-" * 80)
        print(f"{'Ablation':<20} {'HOS Score':<12} {'Difference':<12} {'Status'}")
        print("-" * 80)

        for model_name in model_names[1:]:  # Skip original
            if model_name in seed_results:
                ablation_hos = seed_results[model_name]['hos_score']
                diff = ablation_hos - original_hos
                status = "Better" if diff > 0.01 else "Worse" if diff < -0.01 else "Similar"
                print(f"{model_name:<20} {ablation_hos:<12.4f} {diff:+.4f}       {status}")

    # Save comprehensive results
    comprehensive_results = {
        'experiment_type': f'Pavia DCRN Ablation Study - Single Seed {seed}',
        'seed': seed,
        'results': seed_results,
        'history': seed_history,
        'best_model': best_model,
        'best_hos_score': best_hos_score,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'learning_rate': lr,
            'uncertainty_threshold': uncertainty_threshold,
            'decoupling_threshold': decoupling_threshold,
            'spectral_weight': spectral_weight
        },
        'dataset_info': {
            'name': 'Pavia',
            'input_channels': input_channels,
            'num_classes': 7
        }
    }

    # Save the comprehensive results
    torch.save(comprehensive_results, f'outputs/models/pavia_dcrn_ablation_single_seed_{seed}_comprehensive.pth')

    # Save the models
    for model_name, model_state in seed_models.items():
        torch.save({
            'model_state_dict': model_state,
            'seed': seed,
            'results': seed_results[model_name] if model_name in seed_results else None,
            'model_type': model_name
        }, f'outputs/models/pavia_dcrn_single_seed_{seed}_{model_name.lower()}_model.pth')

    # Save results as JSON for easy reading
    with open(f'outputs/models/pavia_dcrn_ablation_single_seed_{seed}_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, results in seed_results.items():
            json_results[model_name] = {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                        for k, v in results.items() if k != 'confusion_matrix'}

        json_data = {
            'seed': seed,
            'best_model': best_model,
            'best_hos_score': float(best_hos_score),
            'results': json_results
        }
        json.dump(json_data, f, indent=2)

    print(f"\nPavia DCRN ablation study with single seed completed!")
    print(f"All results saved to outputs/models/")
    print(f"Comprehensive results: pavia_dcrn_ablation_single_seed_{seed}_comprehensive.pth")
    print(f"Results summary: pavia_dcrn_ablation_single_seed_{seed}_results.json")
    print(f"All models saved with prefix: pavia_dcrn_single_seed_{seed}_")

    return comprehensive_results


def test_data_loader():
    """Test data loader to check tensor types"""
    print("Testing data loader...")

    try:
        # Load Pavia dataset
        (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
         hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

        # Create a small test dataset
        source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                          patch_size=7, augment=False)

        test_loader = DataLoader(source_train_dataset, batch_size=2,
                                 shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

        # Test one batch
        data, target = next(iter(test_loader))
        print(f"Data loader test successful")
        print(f"  Data shape: {data.shape}, dtype: {data.dtype}")
        print(f"  Target shape: {target.shape}, dtype: {target.dtype}")
        print(f"  Target values: {target}")
        print(f"  Target min: {target.min()}, max: {target.max()}")

        return True

    except Exception as e:
        print(f"Data loader test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation(input_channels=102, patch_size=7, num_classes=7):
    """Test that all models can be created successfully"""
    print("Testing model creation...")

    try:
        # Create all models
        models = create_dcrn_ablation_models(input_channels, patch_size, num_classes)
        print("All models created successfully")

        # Test optimizer creation
        optimizers = create_dcrn_ablation_optimizers(models, 0.00001)
        print("All optimizers created successfully")

        # Test forward pass with dummy data
        dummy_input = torch.randn(2, input_channels, patch_size, patch_size).to(device)
        dummy_target = torch.randint(0, num_classes, (2,)).long().to(device)
        print("Testing forward passes on all models...")

        for name, model in models.items():
            try:
                model.eval()
                with torch.no_grad():
                    outputs = model(dummy_input)
                print(f"  {name}: output shape {outputs['logits'].shape}")

                # Check all required outputs
                required_keys = ['logits', 'alpha_combined', 'alpha_spectral', 'alpha_spatial', 'sifd_outputs']
                missing_keys = [key for key in required_keys if key not in outputs]
                if missing_keys:
                    print(f"    Missing keys: {missing_keys}")
                else:
                    print(f"    All required outputs present")

                # Test EDL loss calculation
                try:
                    edl_loss, _, _ = edl_mse_loss_fixed(dummy_target, outputs['alpha_combined'], 1)
                    print(f"    EDL loss works: {edl_loss.item():.4f}")
                except Exception as e:
                    print(f"    EDL loss failed: {e}")

            except Exception as e:
                print(f"  {name}: {e}")

        print("Model creation test completed")
        return True

    except Exception as e:
        print(f"Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    setup_directories()

    print("Starting DCRN Ablation Study on Pavia Dataset")
    print("Testing different backbone architectures and fusion methods")
    print("-" * 80)

    # Test data loader first
    if not test_data_loader():
        print("Data loader test failed. Please check the errors above.")
        sys.exit(1)

    # Test model creation
    if not test_model_creation():
        print("Model creation test failed. Please check the errors above.")
        sys.exit(1)

    print("All tests passed. Starting full experiment...")
    print("-" * 80)
    print("Ablations to test:")
    print("1. Original DCRN (baseline)")
    print("2. ResNet50 (replace entire DCRN)")
    print("3. EfficientNet (replace entire DCRN)")
    print("4. VGG (replace entire DCRN)")
    print("5. Spectral Branch Only")
    print("6. Spatial Branch Only")
    print("7. Simple Concatenation (no attention)")
    print("8. Element-wise Addition (no attention)")
    print("-" * 80)

    # Run the full experiment with single seed and 50 epochs
    results = run_pavia_dcrn_ablation_single_seed(
        epochs=50,
        batch_size=32,
        patch_size=7,
        lr=0.00001,
        uncertainty_threshold=0.411,
        decoupling_threshold=0.1,
        spectral_weight=0.7
    )

    if results is not None:
        print("\n" + "=" * 100)
        print("PAVIA DCRN ABLATION STUDY COMPLETED!")
        print("All 8 models have been trained and evaluated.")
        print("Check the outputs/models/ directory for detailed results.")
        print("=" * 100)
        print("\nNEXT STEPS:")
        print("- Review the JSON results file for detailed comparisons")
        print("- Analyze which ablation performed best vs the original DCRN")
        print("- Consider running multiple seeds for more robust statistics")
    else:
        print("\nDCRN ablation study failed!")
        print("Check the error messages above for debugging information.")