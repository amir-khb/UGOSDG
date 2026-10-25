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

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
project_root = current_dir
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from src.config.config import device, worker_init_fn, g
from src.data.loaders import load_and_preprocess_pavia_data
from src.data.datasets import HSIDataset, HSITestDataset
from src.models.dcrn import DCRN
from src.models.evidential import EvidentialLayer
from src.models.sifd import GradientReversalLayer
from src.models.classifier import HybridSSUDClassifierFixed
from src.training.losses import edl_mse_loss
from src.training.evaluator import test_open_set_hybrid
from src.utils.helpers import setup_directories

# Import our SIFD ablation variants
from SIFDAblationHelper import SIFD_ABLATION_VARIANTS


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DCRN_SSUD_SIFD_Ablation(nn.Module):
    """
    DCRN-SSUD model with different SIFD ablation variants
    """

    def __init__(self, input_channels=102, patch_size=7, num_classes=7, sifd_variant='Original'):
        super(DCRN_SSUD_SIFD_Ablation, self).__init__()

        # Select SIFD variant
        sifd_class = SIFD_ABLATION_VARIANTS[sifd_variant]
        self.sifd = sifd_class(input_channels)
        self.sifd_variant = sifd_variant

        # Base DCRN model
        self.dcrn = DCRN(input_channels=input_channels, patch_size=patch_size)

        # Output dimensions from DCRN
        self.spectral_size = self.dcrn.spectral_size  # 1024
        self.spatial_size = self.dcrn.spatial_size  # 2048
        self.feature_size = self.dcrn.total_output_channels  # 3072

        # Main classifier for known classes
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Evidential uncertainty estimation
        self.spectral_evidence = EvidentialLayer(self.spectral_size, num_classes)
        self.spatial_evidence = EvidentialLayer(self.spatial_size, num_classes)
        self.combined_evidence = EvidentialLayer(self.feature_size, num_classes)

        # Adaptive weighting between spectral and spatial pathways
        self.pathway_weight = nn.Sequential(
            nn.Linear(self.spectral_size + self.spatial_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Domain classifier on combined features
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
        # Apply SIFD variant to extract domain-invariant features
        sifd_outputs = self.sifd(x)
        invariant_features = sifd_outputs['invariant_features']

        # Fuse with original input
        enhanced_input = x + 0.5 * invariant_features

        # Get features from base DCRN
        features_dict = self.dcrn(enhanced_input)

        # Extract features
        combined_features = features_dict['features']
        spectral_features = features_dict['spectral_features']
        spatial_features = features_dict['spatial_features']

        # Main classification logits
        logits = self.classifier(combined_features)

        # Calculate evidence and uncertainty for each pathway
        spectral_out = self.spectral_evidence(spectral_features)
        spatial_out = self.spatial_evidence(spatial_features)
        combined_out = self.combined_evidence(combined_features)

        # Calculate adaptive weights between spectral and spatial pathways
        pathway_input = torch.cat([spectral_features, spatial_features], dim=1)
        weights = self.pathway_weight(pathway_input)

        # Calculate reliability scores
        spectral_reliability = 1.0 - spectral_out['uncertainty']
        spatial_reliability = 1.0 - spatial_out['uncertainty']

        # Weighted uncertainty
        weighted_uncertainty = (
                weights[:, 0:1] * spectral_out['uncertainty'] +
                weights[:, 1:2] * spatial_out['uncertainty']
        )

        # Soft probability output
        probs = F.softmax(logits, dim=1)

        # Domain adversarial classification
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


def train_one_epoch_parallel_ablation(models, classifiers, train_loader, optimizers, epoch,
                                      alpha=0.9, beta=0.5, gamma=0.1, clip_value=1.0):
    """
    Train all ablation models simultaneously on the same batches
    """
    model_names = list(models.keys())

    # Set all models to training mode
    for model in models.values():
        model.train()

    total_losses = {name: 0 for name in model_names}
    component_losses = {name: {'cls': 0, 'edl': 0, 'domain': 0, 'recon': 0} for name in model_names}
    total_accuracies = {name: 0 for name in model_names}
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Train each model on the same batch
        for name in model_names:
            model = models[name]
            optimizer = optimizers[name]

            optimizer.zero_grad()

            outputs = model(data)

            # Standard classification loss (ensure target is long type)
            target_long = target.long()
            cls_loss = F.cross_entropy(outputs['logits'], target_long)

            # EDL losses
            edl_combined, _, _ = edl_mse_loss(target_long, outputs['alpha_combined'], epoch)
            edl_spectral, _, _ = edl_mse_loss(target_long, outputs['alpha_spectral'], epoch)
            edl_spatial, _, _ = edl_mse_loss(target_long, outputs['alpha_spatial'], epoch)
            edl_total_loss = (edl_combined + alpha * (edl_spectral + edl_spatial)) / 3.0

            # Domain adversarial loss (conditional based on variant)
            if name == 'NoDomainReg':
                domain_loss = torch.tensor(0.0, device=device)
            else:
                domain_loss = F.binary_cross_entropy_with_logits(
                    outputs['sifd_outputs']['domain_pred'],
                    torch.ones_like(outputs['sifd_outputs']['domain_pred']) * 0.5
                )

            # Reconstruction loss (conditional based on variant)
            if name == 'NoRecon':
                recon_loss = torch.tensor(0.0, device=device)
            else:
                data_flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])
                recon_loss = F.mse_loss(outputs['sifd_outputs']['reconstructed_spectrum'], data_flat)

            # Total loss
            loss = cls_loss + alpha * edl_total_loss + beta * domain_loss + gamma * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            # Update metrics
            total_losses[name] += loss.item()
            component_losses[name]['cls'] += cls_loss.item()
            component_losses[name]['edl'] += edl_total_loss.item()
            component_losses[name]['domain'] += domain_loss.item()
            component_losses[name]['recon'] += recon_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], 1)
            correct = (predicted == target_long).sum().item()
            total_accuracies[name] += correct

        total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in model_names:
        avg_loss = total_losses[name] / len(train_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        avg_components = {k: v / len(train_loader) for k, v in component_losses[name].items()}

        results[name] = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'components': avg_components
        }

    return results


def validate_parallel_ablation(models, classifiers, val_loader, epoch):
    """
    Validate all ablation models simultaneously
    """
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

            for name in model_names:
                model = models[name]
                outputs = model(data)

                # Calculate validation loss (ensure target is long type)
                target_long = target.long()
                loss = F.cross_entropy(outputs['logits'], target_long)
                total_losses[name] += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs['logits'], 1)
                correct = (predicted == target_long).sum().item()
                total_accuracies[name] += correct

            total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in model_names:
        avg_loss = total_losses[name] / len(val_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        results[name] = {'loss': avg_loss, 'accuracy': avg_acc}

    return results


def create_ablation_models_pavia(input_channels, patch_size, num_classes):
    """Create all SIFD ablation models for Pavia dataset"""
    models = {}
    ablation_names = list(SIFD_ABLATION_VARIANTS.keys())

    for variant_name in ablation_names:
        model = DCRN_SSUD_SIFD_Ablation(
            input_channels=input_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            sifd_variant=variant_name
        ).to(device)
        models[variant_name] = model

    return models


def create_optimizers_ablation(models, lr=0.00001):
    """Create optimizers for all ablation models"""
    optimizers = {}
    for name, model in models.items():
        optimizer = optim.Adam([
            {'params': model.dcrn.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr},
            {'params': model.spectral_evidence.parameters(), 'lr': lr},
            {'params': model.spatial_evidence.parameters(), 'lr': lr},
            {'params': model.combined_evidence.parameters(), 'lr': lr},
            {'params': model.sifd.parameters(), 'lr': lr},
        ], weight_decay=1e-5)
        optimizers[name] = optimizer

    return optimizers


def run_sifd_ablation_study_pavia(epochs=50, batch_size=32, patch_size=7, lr=0.00001,
                                  uncertainty_threshold=0.4111, decoupling_threshold=0.1, spectral_weight=0.7,
                                  seed=42):
    """
    Run comprehensive SIFD ablation study on Pavia dataset
    """
    print("=" * 100)
    print("SIFD ABLATION STUDY - PAVIA UNIVERSITY → PAVIA CENTER")
    print("=" * 100)

    # Set seed for reproducibility
    set_seed(seed)

    # Load Pavia dataset
    print("\n🏛️ Loading Pavia dataset...")
    (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
     hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

    # Create datasets
    source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                      patch_size=patch_size, augment=True)
    source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u,
                                    patch_size=patch_size, augment=False)
    target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped,
                                         patch_size=patch_size)

    input_channels = hs_pavia_u_norm.shape[2]  # 102

    # Create data loaders
    train_loader = DataLoader(source_train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(target_test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    print(f"✅ Dataset loaded successfully")
    print(f"  Input channels: {input_channels}")
    print(f"  Training samples: {len(source_train_dataset)}")
    print(f"  Validation samples: {len(source_val_dataset)}")
    print(f"  Test samples: {len(target_test_dataset)}")

    # Create all ablation models
    print(f"\n🔧 Creating {len(SIFD_ABLATION_VARIANTS)} ablation models...")
    models = create_ablation_models_pavia(input_channels, patch_size, 7)

    # Create classifiers for each model
    classifiers = {}
    for name, model in models.items():
        classifiers[name] = HybridSSUDClassifierFixed(
            model, num_classes=7,
            uncertainty_threshold=uncertainty_threshold,
            decoupling_threshold=decoupling_threshold,
            spectral_weight=spectral_weight
        ).to(device)

    # Create optimizers
    optimizers = create_optimizers_ablation(models, lr)

    print(f"✅ Models created:")
    for name in models.keys():
        print(f"  - {name}")

    # Training history
    history = {name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
               for name in models.keys()}

    best_models = {name: None for name in models.keys()}
    best_val_accs = {name: 0 for name in models.keys()}

    print(f"\n🚀 Starting parallel training for {epochs} epochs...")

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train all models on the same batches
        train_results = train_one_epoch_parallel_ablation(
            models, classifiers, train_loader, optimizers, epoch,
            alpha=0.9, beta=0.5, gamma=0.1  # Pavia hyperparameters
        )

        # Validate all models
        val_results = validate_parallel_ablation(models, classifiers, val_loader, epoch)

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

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print("Validation accuracies:")
            for name in models.keys():
                print(f"  {name}: {val_results[name]['accuracy']:.2f}%")

    print(f"\n🎉 Training completed!")

    # Load best models and evaluate on test set
    print(f"\n🧪 Evaluating all models on test set...")
    final_results = {}

    for name in models.keys():
        # Load best model
        models[name].load_state_dict(best_models[name])

        # Evaluate on test set
        known_acc, unknown_rej, hos_score, cm = test_open_set_hybrid(models[name], classifiers[name], test_loader)

        final_results[name] = {
            'sifd_variant': name,
            'known_accuracy': known_acc,
            'unknown_rejection': unknown_rej,
            'hos_score': hos_score,
            'best_val_acc': best_val_accs[name],
            'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
        }

        print(f"\n📊 {name} Results:")
        print(f"  Best Val Acc: {best_val_accs[name]:.2f}%")
        print(f"  Known Acc: {known_acc:.2f}%")
        print(f"  Unknown Rej: {unknown_rej:.2f}%")
        print(f"  HOS Score: {hos_score:.2f}")

    # Create comprehensive results summary
    print(f"\n{'=' * 80}")
    print("FINAL SIFD ABLATION RESULTS SUMMARY")
    print(f"{'=' * 80}")

    # Sort by HOS score
    sorted_results = sorted(final_results.items(), key=lambda x: x[1]['hos_score'], reverse=True)

    print(f"{'Rank':<4} {'Variant':<15} {'Val Acc':<8} {'Known Acc':<10} {'Unknown Rej':<12} {'HOS Score':<10}")
    print("-" * 80)

    for rank, (name, result) in enumerate(sorted_results, 1):
        print(f"{rank:<4} {name:<15} {result['best_val_acc']:<8.2f} {result['known_accuracy']:<10.2f} "
              f"{result['unknown_rejection']:<12.2f} {result['hos_score']:<10.2f}")

    # Save comprehensive results
    comprehensive_results = {
        'experiment_type': 'SIFD Ablation Study - Pavia Dataset',
        'dataset': 'Pavia University → Pavia Center',
        'seed': seed,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'alpha': 0.9,
            'beta': 0.5,
            'gamma': 0.1,
            'uncertainty_threshold': uncertainty_threshold,
            'decoupling_threshold': decoupling_threshold,
            'spectral_weight': spectral_weight
        },
        'final_results': final_results,
        'training_history': history,
        'best_models_info': {name: {'best_val_acc': acc} for name, acc in best_val_accs.items()},
        'rankings': [(name, result['hos_score']) for name, result in sorted_results]
    }

    # Save results
    results_path = '../outputs/models/sifd_ablation_pavia_comprehensive.json'
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    # Save best models
    for name, model_state in best_models.items():
        model_path = f'outputs/models/sifd_ablation_pavia_{name.lower()}_best.pth'
        torch.save({
            'model_state_dict': model_state,
            'results': final_results[name],
            'sifd_variant': name,
            'seed': seed
        }, model_path)

    print(f"\n💾 Results saved:")
    print(f"  📊 Comprehensive results: {results_path}")
    print(f"  🏆 Best models: outputs/models/sifd_ablation_pavia_*_best.pth")

    # Print key insights
    best_variant = sorted_results[0][0]
    best_hos = sorted_results[0][1]['hos_score']
    original_hos = final_results['Original']['hos_score']

    print(f"\n🔍 Key Insights:")
    print(f"  🥇 Best variant: {best_variant} (HOS: {best_hos:.2f})")
    print(f"  📈 Original SIFD: HOS {original_hos:.2f}")
    print(f"  📊 Improvement over original: {best_hos - original_hos:+.2f}")

    # Component analysis
    print(f"\n🧩 Component Analysis:")
    freq_variants = ['Original', 'NoFreqTransform', 'DCT', 'Wavelet', 'RealOnly', 'ImagOnly']
    other_variants = ['NoAttention', 'NoDomainReg', 'NoRecon']

    print("  Frequency Transform Variants:")
    for name in freq_variants:
        if name in final_results:
            hos = final_results[name]['hos_score']
            print(f"    {name}: {hos:.2f}")

    print("  Other Component Variants:")
    for name in other_variants:
        if name in final_results:
            hos = final_results[name]['hos_score']
            print(f"    {name}: {hos:.2f}")

    return comprehensive_results


if __name__ == "__main__":
    # Setup directories
    setup_directories()

    print("🚀 Starting SIFD Ablation Study for Pavia Dataset")

    # Install required packages if not available
    try:
        import pywt

        print("✅ PyWavelets available")
    except ImportError:
        print("❌ PyWavelets not found. Please install with: pip install PyWavelets")
        sys.exit(1)

    # Run the ablation study
    results = run_sifd_ablation_study_pavia(
        epochs=50,
        batch_size=32,
        patch_size=7,
        lr=0.00001,
        uncertainty_threshold=0.4111,
        decoupling_threshold=0.1,
        spectral_weight=0.7,
        seed=42
    )

    print("\n" + "=" * 100)
    print("🎉 SIFD ABLATION STUDY COMPLETED!")
    print("All variants trained in parallel with same data batches")
    print("Results saved in outputs/models/")
    print("=" * 100)