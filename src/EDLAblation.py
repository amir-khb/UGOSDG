import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import json
from collections import defaultdict

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
project_root = current_dir
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from src.config.config import device, worker_init_fn, g, set_full_determinism
from src.data.loaders import load_and_preprocess_pavia_data
from src.data.datasets import HSIDataset, HSITestDataset
from src.models.dcrn import DCRN
from src.models.sifd import SpectrumInvariantFrequencyDisentanglement, GradientReversalLayer
from src.models.classifier import HybridSSUDClassifierFixed
from src.training.evaluator import test_open_set_hybrid
from src.utils.helpers import setup_directories


# ================================
# ALTERNATIVE UNCERTAINTY MODULES
# ================================

class SoftmaxConfidenceLayer(nn.Module):
    """Alternative to EDL: Use max softmax probability as confidence"""

    def __init__(self, in_features, num_classes):
        super(SoftmaxConfidenceLayer, self).__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1, keepdim=True)

        # Uncertainty = 1 - confidence
        uncertainty = 1.0 - max_probs

        # Create dummy alpha for compatibility
        alpha = probs + 1.0

        return {
            'evidence': probs,
            'alpha': alpha,
            'strength': torch.sum(alpha, dim=1, keepdim=True),
            'prob': probs,
            'uncertainty': uncertainty
        }


class EntropyBasedLayer(nn.Module):
    """Alternative to EDL: Use normalized entropy as uncertainty"""

    def __init__(self, in_features, num_classes):
        super(EntropyBasedLayer, self).__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)

        # Normalized entropy as uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        uncertainty = entropy / np.log(self.num_classes)

        # Create dummy alpha for compatibility
        alpha = probs + 1.0

        return {
            'evidence': probs,
            'alpha': alpha,
            'strength': torch.sum(alpha, dim=1, keepdim=True),
            'prob': probs,
            'uncertainty': uncertainty
        }


class MCDropoutLayer(nn.Module):
    """Alternative to EDL: Use Monte Carlo Dropout for uncertainty"""

    def __init__(self, in_features, num_classes, n_samples=10):
        super(MCDropoutLayer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features // 2, num_classes)
        )
        self.num_classes = num_classes
        self.n_samples = n_samples

    def forward(self, x):
        if self.training:
            # During training, single forward pass
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)

            # Simple entropy-based uncertainty during training
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
            uncertainty = entropy / np.log(self.num_classes)

        else:
            # During inference, MC sampling
            self.train()  # Enable dropout
            all_probs = []

            for _ in range(self.n_samples):
                with torch.no_grad():
                    logits = self.classifier(x)
                    sample_probs = F.softmax(logits, dim=1)
                    all_probs.append(sample_probs)

            self.eval()  # Disable dropout

            # Calculate mean and variance across samples
            all_probs = torch.stack(all_probs, dim=0)  # [n_samples, batch_size, num_classes]
            probs = torch.mean(all_probs, dim=0)

            # Uncertainty from predictive variance
            predictive_variance = torch.var(all_probs, dim=0)
            uncertainty = torch.mean(predictive_variance, dim=1, keepdim=True)

        # Create dummy alpha for compatibility
        alpha = probs + 1.0

        return {
            'evidence': probs,
            'alpha': alpha,
            'strength': torch.sum(alpha, dim=1, keepdim=True),
            'prob': probs,
            'uncertainty': uncertainty
        }


class DeepEnsembleLayer(nn.Module):
    """Alternative to EDL: Use ensemble disagreement as uncertainty"""

    def __init__(self, in_features, num_classes, n_models=3):
        super(DeepEnsembleLayer, self).__init__()
        self.n_models = n_models
        self.ensembles = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(in_features // 2, num_classes)
            ) for _ in range(n_models)
        ])
        self.num_classes = num_classes

    def forward(self, x):
        all_probs = []

        for model in self.ensembles:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        # Stack predictions
        all_probs = torch.stack(all_probs, dim=0)  # [n_models, batch_size, num_classes]

        # Mean prediction
        mean_probs = torch.mean(all_probs, dim=0)

        # Uncertainty from ensemble disagreement
        disagreement = torch.var(all_probs, dim=0)
        uncertainty = torch.mean(disagreement, dim=1, keepdim=True)

        # Create dummy alpha for compatibility
        alpha = mean_probs + 1.0

        return {
            'evidence': mean_probs,
            'alpha': alpha,
            'strength': torch.sum(alpha, dim=1, keepdim=True),
            'prob': mean_probs,
            'uncertainty': uncertainty
        }


class TemperatureScalingLayer(nn.Module):
    """Alternative to EDL: Use temperature-calibrated softmax"""

    def __init__(self, in_features, num_classes):
        super(TemperatureScalingLayer, self).__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.classifier(x)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)

        # Use max probability for uncertainty (inverse confidence)
        max_probs, _ = torch.max(probs, dim=1, keepdim=True)
        uncertainty = 1.0 - max_probs

        # Create dummy alpha for compatibility
        alpha = probs + 1.0

        return {
            'evidence': probs,
            'alpha': alpha,
            'strength': torch.sum(alpha, dim=1, keepdim=True),
            'prob': probs,
            'uncertainty': uncertainty
        }


# ================================
# MODIFIED MODEL CLASSES
# ================================

class DCRN_SSUD_SIFD_SoftmaxConf(nn.Module):
    """Model with Softmax Confidence instead of EDL"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRN_SSUD_SIFD_SoftmaxConf, self).__init__()

        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        self.spectral_size = self.dcrn.spectral_size
        self.spatial_size = self.dcrn.spatial_size
        self.feature_size = self.dcrn.total_output_channels

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Replace EDL with Softmax Confidence
        self.spectral_evidence = SoftmaxConfidenceLayer(self.spectral_size, num_classes)
        self.spatial_evidence = SoftmaxConfidenceLayer(self.spatial_size, num_classes)
        self.combined_evidence = SoftmaxConfidenceLayer(self.feature_size, num_classes)

        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

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
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        features_dict = self.dcrn(enhanced_input)
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        logits = self.classifier(combined_features)

        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        probs = F.softmax(logits, dim=1)
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


class DCRN_SSUD_SIFD_Entropy(nn.Module):
    """Model with Entropy-based uncertainty instead of EDL"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRN_SSUD_SIFD_Entropy, self).__init__()

        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        self.spectral_size = self.dcrn.spectral_size
        self.spatial_size = self.dcrn.spatial_size
        self.feature_size = self.dcrn.total_output_channels

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Replace EDL with Entropy-based
        self.spectral_evidence = EntropyBasedLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EntropyBasedLayer(self.spatial_size, num_classes)
        self.combined_evidence = EntropyBasedLayer(self.feature_size, num_classes)

        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

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
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        features_dict = self.dcrn(enhanced_input)
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        logits = self.classifier(combined_features)

        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        probs = F.softmax(logits, dim=1)
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


class DCRN_SSUD_SIFD_MCDropout(nn.Module):
    """Model with MC Dropout instead of EDL"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRN_SSUD_SIFD_MCDropout, self).__init__()

        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        self.spectral_size = self.dcrn.spectral_size
        self.spatial_size = self.dcrn.spatial_size
        self.feature_size = self.dcrn.total_output_channels

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Replace EDL with MC Dropout
        self.spectral_evidence = MCDropoutLayer(self.spectral_size, num_classes)
        self.spatial_evidence = MCDropoutLayer(self.spatial_size, num_classes)
        self.combined_evidence = MCDropoutLayer(self.feature_size, num_classes)

        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

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
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        features_dict = self.dcrn(enhanced_input)
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        logits = self.classifier(combined_features)

        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        probs = F.softmax(logits, dim=1)
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


class DCRN_SSUD_SIFD_DeepEnsemble(nn.Module):
    """Model with Deep Ensembles instead of EDL"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRN_SSUD_SIFD_DeepEnsemble, self).__init__()

        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        self.spectral_size = self.dcrn.spectral_size
        self.spatial_size = self.dcrn.spatial_size
        self.feature_size = self.dcrn.total_output_channels

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Replace EDL with Deep Ensembles
        self.spectral_evidence = DeepEnsembleLayer(self.spectral_size, num_classes)
        self.spatial_evidence = DeepEnsembleLayer(self.spatial_size, num_classes)
        self.combined_evidence = DeepEnsembleLayer(self.feature_size, num_classes)

        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

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
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        features_dict = self.dcrn(enhanced_input)
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        logits = self.classifier(combined_features)

        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        probs = F.softmax(logits, dim=1)
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


class DCRN_SSUD_SIFD_TempScaling(nn.Module):
    """Model with Temperature Scaling instead of EDL"""

    def __init__(self, input_channels=102, patch_size=7, num_classes=7):
        super(DCRN_SSUD_SIFD_TempScaling, self).__init__()

        self.sifd = SpectrumInvariantFrequencyDisentanglement(input_channels)
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        self.spectral_size = self.dcrn.spectral_size
        self.spatial_size = self.dcrn.spatial_size
        self.feature_size = self.dcrn.total_output_channels

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Replace EDL with Temperature Scaling
        self.spectral_evidence = TemperatureScalingLayer(self.spectral_size, num_classes)
        self.spatial_evidence = TemperatureScalingLayer(self.spatial_size, num_classes)
        self.combined_evidence = TemperatureScalingLayer(self.feature_size, num_classes)

        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

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
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']
        enhanced_input = x + 0.5 * invariant_features

        features_dict = self.dcrn(enhanced_input)
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        logits = self.classifier(combined_features)

        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        probs = F.softmax(logits, dim=1)
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


# ================================
# SIMPLIFIED EDL LOSS FOR COMPATIBILITY
# ================================

def simplified_edl_loss(target, alpha):
    """Simplified EDL loss that works with all uncertainty methods"""
    # For non-EDL methods, alpha is just (probs + 1), so we can use standard cross-entropy
    # Ensure target is LongTensor for one_hot
    target = target.long()
    y = F.one_hot(target, num_classes=alpha.size(1)).float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    expected_p = alpha / S

    # Simple MSE loss
    mse_loss = torch.sum((y - expected_p) ** 2, dim=1).mean()
    return mse_loss


# ================================
# PARALLEL TRAINING FUNCTIONS
# ================================

class ProgressTracker:
    """Thread-safe progress tracker for parallel training"""

    def __init__(self, model_names):
        self.model_names = model_names
        self.current_epochs = {name: 0 for name in model_names}
        self.val_accuracies = {name: [] for name in model_names}
        self.train_accuracies = {name: [] for name in model_names}
        self.best_val_accs = {name: 0.0 for name in model_names}
        self.lock = threading.Lock()

    def update_progress(self, model_name, epoch, train_acc, val_acc):
        with self.lock:
            self.current_epochs[model_name] = epoch
            self.train_accuracies[model_name].append(train_acc)
            self.val_accuracies[model_name].append(val_acc)
            if val_acc > self.best_val_accs[model_name]:
                self.best_val_accs[model_name] = val_acc

    def print_progress(self):
        with self.lock:
            print(f"\n{'=' * 80}")
            print(f"EPOCH PROGRESS - All Models")
            print(f"{'=' * 80}")
            print(f"{'Model':<20} {'Epoch':<8} {'Train Acc':<12} {'Val Acc':<12} {'Best Val':<12}")
            print(f"{'-' * 80}")

            for name in self.model_names:
                epoch = self.current_epochs[name]
                train_acc = self.train_accuracies[name][-1] if self.train_accuracies[name] else 0.0
                val_acc = self.val_accuracies[name][-1] if self.val_accuracies[name] else 0.0
                best_val = self.best_val_accs[name]

                print(f"{name:<20} {epoch:<8} {train_acc:<12.2f} {val_acc:<12.2f} {best_val:<12.2f}")
            print(f"{'=' * 80}")


def train_single_model(model_name, model_class, train_loader, val_loader, test_loader,
                       input_channels, num_classes, epochs, progress_tracker):
    """Train a single model in a separate thread"""
    print(f"🚀 Starting training for {model_name}")

    try:
        # Set seed for this thread
        set_full_determinism(42)

        # Create model
        model = model_class(input_channels=input_channels, patch_size=7, num_classes=num_classes).to(device)

        # Create classifier
        classifier = HybridSSUDClassifierFixed(
            model, num_classes=num_classes,
            uncertainty_threshold=0.4111,
            decoupling_threshold=0.1,
            spectral_weight=0.7
        ).to(device)

        # Optimizer
        optimizer = optim.Adam([
            {'params': model.dcrn.parameters(), 'lr': 0.00001},
            {'params': model.classifier.parameters(), 'lr': 0.00001},
            {'params': model.sifd.parameters(), 'lr': 0.00001},
        ], weight_decay=1e-5)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000001)

        # Best model tracking
        best_val_acc = 0
        best_model_state = None

        print(f"📊 {model_name}: Model created successfully, starting training...")

        # Training loop
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device), target.to(device).long()  # Ensure target is LongTensor

                    # Debug: Check data types on first batch
                    if epoch == 1 and batch_idx == 0:
                        print(f"🔍 {model_name}: Data shape: {data.shape}, Target shape: {target.shape}")
                        print(f"🔍 {model_name}: Data dtype: {data.dtype}, Target dtype: {target.dtype}")
                        print(f"🔍 {model_name}: Target range: {target.min().item()} - {target.max().item()}")

                    optimizer.zero_grad()

                    outputs = model(data)

                    # Classification loss
                    cls_loss = F.cross_entropy(outputs['logits'], target)

                    # Simplified EDL losses (compatible with all methods)
                    edl_combined = simplified_edl_loss(target, outputs['alpha_combined'])
                    edl_spectral = simplified_edl_loss(target, outputs['alpha_spectral'])
                    edl_spatial = simplified_edl_loss(target, outputs['alpha_spatial'])
                    edl_total_loss = (edl_combined + 0.3 * (edl_spectral + edl_spatial)) / 3.0

                    # Domain adversarial loss
                    domain_loss = F.binary_cross_entropy_with_logits(
                        outputs['sifd_outputs']['domain_pred'],
                        torch.ones_like(outputs['sifd_outputs']['domain_pred']) * 0.5
                    )

                    # Reconstruction loss
                    data_flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])
                    recon_loss = F.mse_loss(outputs['sifd_outputs']['reconstructed_spectrum'], data_flat)

                    # Total loss
                    loss = cls_loss + 0.3 * edl_total_loss + 0.5 * domain_loss + 0.3 * recon_loss

                    # Check for NaN/Inf
                    if not torch.isfinite(loss):
                        print(f"⚠️ {model_name}: Non-finite loss detected at epoch {epoch}, batch {batch_idx}")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs['logits'], 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                except Exception as e:
                    print(f"❌ {model_name}: Error in batch {batch_idx} of epoch {epoch}: {e}")
                    continue

            if total == 0:
                print(f"⚠️ {model_name}: No valid batches in epoch {epoch}")
                continue

            train_acc = 100.0 * correct / total

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data, target = data.to(device), target.to(device).long()  # Ensure target is LongTensor
                        outputs = model(data)
                        _, predicted = torch.max(outputs['logits'], 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                    except Exception as e:
                        print(f"⚠️ {model_name}: Error in validation: {e}")
                        continue

            if val_total == 0:
                print(f"⚠️ {model_name}: No valid validation batches in epoch {epoch}")
                continue

            val_acc = 100.0 * val_correct / val_total

            # Update learning rate
            scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())

            # Update progress tracker
            progress_tracker.update_progress(model_name, epoch, train_acc, val_acc)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"🔄 {model_name}: Epoch {epoch}/{epochs} - Train: {train_acc:.2f}% - Val: {val_acc:.2f}%")

        # Load best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            known_acc, unknown_rej, hos_score, cm = test_open_set_hybrid(model, classifier, test_loader)
        else:
            print(f"⚠️ {model_name}: No valid model state saved, using final state")
            known_acc, unknown_rej, hos_score, cm = test_open_set_hybrid(model, classifier, test_loader)

        print(f"✅ {model_name} training completed!")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
        print(f"   Test Known Acc: {known_acc:.2f}%")
        print(f"   Test Unknown Rej: {unknown_rej:.2f}%")
        print(f"   HOS Score: {hos_score:.4f}")

        return {
            'model_name': model_name,
            'best_val_acc': best_val_acc,
            'final_metrics': {
                'known_acc': known_acc,
                'unknown_rej': unknown_rej,
                'hos_score': hos_score
            },
            'train_history': progress_tracker.train_accuracies[model_name],
            'val_history': progress_tracker.val_accuracies[model_name],
            'model_state': best_model_state if best_model_state is not None else model.state_dict()
        }

    except Exception as e:
        print(f"❌ {model_name}: Fatal error during training: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model_name': model_name,
            'error': str(e),
            'best_val_acc': 0.0,
            'final_metrics': {
                'known_acc': 0.0,
                'unknown_rej': 0.0,
                'hos_score': 0.0
            },
            'train_history': [],
            'val_history': [],
            'model_state': None
        }


def run_edl_ablation_parallel():
    """Main function to run EDL ablation study in parallel"""
    print("=" * 100)
    print("EDL ABLATION STUDY - PARALLEL TRAINING ON PAVIA DATASET")
    print("=" * 100)

    # Setup directories
    setup_directories()

    # Set global seed
    set_full_determinism(42)

    # Load Pavia dataset
    print("\n🏛️ Loading Pavia dataset...")
    (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
     hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

    # Create datasets
    source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                      patch_size=7, augment=True)
    source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u,
                                    patch_size=7, augment=False)
    target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped,
                                         patch_size=7)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    input_channels = hs_pavia_u_norm.shape[2]  # 102
    num_classes = 7
    epochs = 50

    # Define models to test
    models_to_test = {
        'SoftmaxConf': DCRN_SSUD_SIFD_SoftmaxConf,
        'Entropy': DCRN_SSUD_SIFD_Entropy,
        'MCDropout': DCRN_SSUD_SIFD_MCDropout,
        'DeepEnsemble': DCRN_SSUD_SIFD_DeepEnsemble,
        'TempScaling': DCRN_SSUD_SIFD_TempScaling
    }

    print(f"\n📊 Models to test: {list(models_to_test.keys())}")
    print(f"🔢 Training parameters: {epochs} epochs, batch size {batch_size}")
    print(f"🎯 Input channels: {input_channels}, Classes: {num_classes}")

    # Create progress tracker
    progress_tracker = ProgressTracker(list(models_to_test.keys()))

    # Progress monitoring thread
    def monitor_progress():
        while True:
            time.sleep(10)  # Print progress every 10 seconds
            progress_tracker.print_progress()

            # Check if all models are done
            with progress_tracker.lock:
                all_done = all(epoch >= epochs for epoch in progress_tracker.current_epochs.values())
            if all_done:
                break

    # Start progress monitoring thread
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()

    print(f"\n🚀 Starting parallel training with {len(models_to_test)} models...")

    # Run parallel training
    with ThreadPoolExecutor(max_workers=len(models_to_test)) as executor:
        futures = []

        for model_name, model_class in models_to_test.items():
            future = executor.submit(
                train_single_model,
                model_name, model_class, train_loader, val_loader, test_loader,
                input_channels, num_classes, epochs, progress_tracker
            )
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                if 'error' not in result:  # Only add successful results
                    results.append(result)
                else:
                    print(f"❌ Model {result['model_name']} failed: {result['error']}")
            except Exception as e:
                print(f"❌ Error collecting result: {e}")

        if not results:
            print("❌ No models completed successfully!")
            return None

    # Final results summary
    print(f"\n" + "=" * 100)
    print("FINAL EDL ABLATION RESULTS")
    print("=" * 100)

    if not results:
        print("❌ No models completed successfully!")
        return None

    print(f"✅ {len(results)} out of {len(models_to_test)} models completed successfully")

    # Sort results by HOS score
    results.sort(key=lambda x: x['final_metrics']['hos_score'], reverse=True)

    print(f"{'Rank':<6} {'Model':<15} {'Val Acc':<10} {'Known Acc':<12} {'Unknown Rej':<12} {'HOS Score':<10}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        metrics = result['final_metrics']
        print(f"{i:<6} {result['model_name']:<15} {result['best_val_acc']:<10.2f} "
              f"{metrics['known_acc']:<12.2f} {metrics['unknown_rej']:<12.2f} "
              f"{metrics['hos_score']:<10.4f}")

    # Save results
    save_results = {
        'experiment_type': 'EDL Ablation Study - Pavia Dataset',
        'models_tested': list(models_to_test.keys()),
        'results': results,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'input_channels': input_channels,
            'num_classes': num_classes,
            'uncertainty_threshold': 0.4111,
            'decoupling_threshold': 0.1,
            'spectral_weight': 0.7
        }
    }

    # Save comprehensive results
    torch.save(save_results, 'outputs/models/edl_ablation_parallel_results.pth')

    # Save best models
    for result in results:
        model_name = result['model_name']
        torch.save({
            'model_state_dict': result['model_state'],
            'results': result['final_metrics'],
            'model_type': model_name
        }, f'outputs/models/edl_ablation_{model_name.lower()}_best_model.pth')

    # Save results as JSON for easy reading
    json_results = {
        'best_model': results[0]['model_name'],
        'best_hos_score': results[0]['final_metrics']['hos_score'],
        'all_results': [
            {
                'model_name': r['model_name'],
                'best_val_acc': float(r['best_val_acc']),
                'known_acc': float(r['final_metrics']['known_acc']),
                'unknown_rej': float(r['final_metrics']['unknown_rej']),
                'hos_score': float(r['final_metrics']['hos_score'])
            }
            for r in results
        ]
    }

    with open('outputs/models/edl_ablation_results_summary.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✅ EDL ablation study completed!")
    print(f"🏆 Best model: {results[0]['model_name']} (HOS: {results[0]['final_metrics']['hos_score']:.4f})")
    print(f"📁 Results saved to outputs/models/")
    print(f"📊 Summary: edl_ablation_results_summary.json")
    print(f"📈 Detailed: edl_ablation_parallel_results.pth")
    print("=" * 100)

    return save_results


if __name__ == "__main__":
    # Run the EDL ablation study
    results = run_edl_ablation_parallel()