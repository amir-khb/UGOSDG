#!/usr/bin/env python3
"""
Merged Uncertainty Analysis Script
Generates exactly 5 key figures for paper
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up one level to src directory
project_root = os.path.dirname(src_dir)  # Go up one more level to project root

# Add both src and project root to path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your existing modules
from config.config import device, worker_init_fn, g
from data.loaders import load_and_preprocess_pavia_data
from data.datasets import HSITestDataset
from models.classifier import DCRN_SSUD_SIFD, HybridSSUDClassifierFixed
from utils.helpers import setup_directories

# Set style
plt.style.use('default')
sns.set_palette("husl")


def run_analysis():
    """Main function to generate 5 key figures"""
    print("🎯 GENERATING 5 KEY UNCERTAINTY FIGURES")
    print("=" * 80)

    setup_directories()

    # Calibrated parameters
    calibrated_params = {
        'uncertainty': 0.4776875674724579,
        'decoupling': 0.1,
        'spectral_weight': 0.7
    }

    print(f"Parameters: uncertainty={calibrated_params['uncertainty']:.6f}, "
          f"decoupling={calibrated_params['decoupling']}, "
          f"spectral_weight={calibrated_params['spectral_weight']}")

    # Model path
    model_path = '/home/amirreza/OSDG/BestModels/PaviaHOS7738_run066.pth'

    # Load data
    print("\n🏛️ Loading Pavia data...")
    test_loader, input_channels = load_pavia_data()

    # Load model and collect data
    print("\n📂 Loading model and collecting data...")
    data = load_model_and_collect_data(model_path, test_loader, input_channels, 7, calibrated_params)

    # Generate 5 figures
    print("\n🎨 Generating 5 key figures...")
    generate_five_figures(data, calibrated_params)

    print("\n✅ All 5 figures generated successfully!")
    print("📁 Saved to: outputs/figures/")


def load_pavia_data():
    """Load Pavia test data"""
    (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
     hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

    target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped, patch_size=7)
    test_loader = DataLoader(target_test_dataset, batch_size=32, shuffle=False,
                             num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    return test_loader, hs_pavia_u_norm.shape[2]


def load_model_and_collect_data(model_path, test_loader, input_channels, num_classes, calibrated_params):
    """Load model and collect all necessary data"""

    # Create and load model
    model = DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=7, num_classes=num_classes).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create classifier
    classifier = HybridSSUDClassifierFixed(
        model, num_classes=num_classes,
        uncertainty_threshold=calibrated_params['uncertainty'],
        decoupling_threshold=calibrated_params['decoupling'],
        spectral_weight=calibrated_params['spectral_weight']
    ).to(device)

    # Collect data
    model.eval()
    data = {
        'spectral_uncertainties': [],
        'spatial_uncertainties': [],
        'ssud_decisions': [],
        'sample_types': [],
        'alphas': [],
        'targets': [],
        'class_uncertainties': {i: [] for i in range(num_classes + 1)},
    }

    with torch.no_grad():
        for batch_idx, (batch_data, target) in enumerate(test_loader):
            batch_data, target = batch_data.to(device), target.to(device)

            # Get outputs
            outputs = model(batch_data)
            results = classifier.predict(batch_data, return_uncertainties=True)

            # Calculate SSUD decisions
            spectral_uncertainty = outputs['uncertainty_spectral'].view(-1)
            spatial_uncertainty = outputs['uncertainty_spatial'].view(-1)
            spectral_reliability = 1.0 - spectral_uncertainty
            spatial_reliability = 1.0 - spatial_uncertainty
            reliability_diff = torch.abs(spectral_reliability - spatial_reliability)
            decoupling_mask = reliability_diff > calibrated_params['decoupling']
            use_spectral = spectral_reliability > spatial_reliability

            ssud_decisions = torch.zeros_like(reliability_diff, dtype=torch.long)
            spectral_indices = decoupling_mask & use_spectral
            spatial_indices = decoupling_mask & ~use_spectral
            ssud_decisions[spectral_indices] = 1
            ssud_decisions[spatial_indices] = 2

            # Store data
            data['spectral_uncertainties'].extend(spectral_uncertainty.cpu().numpy())
            data['spatial_uncertainties'].extend(spatial_uncertainty.cpu().numpy())
            data['ssud_decisions'].extend(ssud_decisions.cpu().numpy())
            data['alphas'].extend(outputs['alpha_combined'].cpu().numpy())
            data['targets'].extend(target.cpu().numpy())

            # Sample types
            known_mask = target < num_classes
            unknown_mask = target >= num_classes
            sample_types = ['known' if k else 'unknown' for k in known_mask.cpu().numpy()]
            data['sample_types'].extend(sample_types)

            # Class-wise uncertainties
            for i, cls in enumerate(target.cpu().numpy()):
                if cls < num_classes:
                    data['class_uncertainties'][cls].append(results['uncertainty'][i].cpu().numpy())
                else:
                    data['class_uncertainties'][num_classes].append(results['uncertainty'][i].cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Convert to numpy
    for key in ['spectral_uncertainties', 'spatial_uncertainties', 'ssud_decisions', 'alphas', 'targets']:
        data[key] = np.array(data[key])

    print(f"✅ Data collected: {len(data['sample_types'])} samples")
    return data


def generate_five_figures(data, calibrated_params):
    """Generate exactly 5 figures"""

    # Figure 1: SSUD Decision by Sample Type
    create_figure1_ssud_decision_by_type(data)

    # Figure 2: Class-wise Mean Uncertainty
    create_figure2_classwise_uncertainty(data)

    # Figure 3: Evidence vs Uncertainty
    create_figure3_evidence_vs_uncertainty(data)

    # Figure 4: Evidence Entropy
    create_figure4_evidence_entropy(data)

    # Figure 5: Spectral vs Spatial with SSUD
    create_figure5_spectral_spatial_ssud(data)


def create_figure1_ssud_decision_by_type(data):
    """Figure 1: SSUD Decision Distribution by Sample Type"""
    print("  Creating Figure 1: SSUD Decision by Sample Type...")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ssud_decisions = data['ssud_decisions']
    sample_types = data['sample_types']
    known_mask = np.array(sample_types) == 'known'
    unknown_mask = np.array(sample_types) == 'unknown'

    decision_labels = ['Combined', 'Spectral', 'Spatial']
    known_decisions = [np.sum((ssud_decisions == i) & known_mask) for i in range(3)]
    unknown_decisions = [np.sum((ssud_decisions == i) & unknown_mask) for i in range(3)]

    x = np.arange(len(decision_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, known_decisions, width, label='Known Classes', color='blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, unknown_decisions, width, label='Unknown Classes', color='red', alpha=0.8)

    ax.set_xlabel('SSUD Decision', fontsize=20)
    ax.set_ylabel('Number of Samples', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(decision_labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/figure1_ssud_decision_by_type.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_figure2_classwise_uncertainty(data):
    """Figure 2: Class-wise Mean Uncertainty with Error Bars"""
    print("  Creating Figure 2: Class-wise Mean Uncertainty...")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    class_uncertainties = data['class_uncertainties']
    valid_classes = {k: v for k, v in class_uncertainties.items() if len(v) > 0}

    if not valid_classes:
        print("    ⚠️  No valid class data")
        return

    class_ids = sorted(valid_classes.keys())
    class_labels = ['Trees', 'Asphalt', 'Bricks', 'Bitumen', 'Meadows', 'Shadow', 'Bare Soil', 'Unknown']
    means = [np.mean(valid_classes[k]) for k in class_ids]
    stds = [np.std(valid_classes[k]) for k in class_ids]

    colors = ['lightblue'] * (len(class_ids) - 1) + ['lightcoral']
    bars = ax.bar(range(len(class_ids)), means, yerr=stds, capsize=5, alpha=0.8, color=colors)

    ax.set_xlabel('Class', fontsize=20)
    ax.set_ylabel('Mean Uncertainty', fontsize=20)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(class_labels[:len(class_ids)], rotation=0, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=16)

    # Set y-axis limit to accommodate text labels
    max_height = max([means[i] + stds[i] for i in range(len(means))])
    ax.set_ylim(0, max_height * 1.15)  # Add 15% padding at the top

    plt.tight_layout()
    plt.savefig('outputs/figures/figure2_classwise_uncertainty.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_figure3_evidence_vs_uncertainty(data):
    """Figure 3: Evidence Strength vs Uncertainty"""
    print("  Creating Figure 3: Evidence vs Uncertainty...")

    fig, ax = plt.subplots(figsize=(7, 6))

    alphas = data['alphas']
    sample_types = data['sample_types']
    num_classes = alphas.shape[1]

    known_mask = np.array(sample_types) == 'known'
    unknown_mask = np.array(sample_types) == 'unknown'

    # Calculate evidence strength and uncertainty
    known_strength = np.sum(alphas[known_mask], axis=1)
    unknown_strength = np.sum(alphas[unknown_mask], axis=1)
    known_uncertainty = num_classes / known_strength
    unknown_uncertainty = num_classes / unknown_strength

    ax.scatter(known_strength, known_uncertainty, alpha=0.6, label='Known', s=10, color='blue')
    ax.scatter(unknown_strength, unknown_uncertainty, alpha=0.6, label='Unknown', s=10, color='red')

    ax.set_xlabel('Evidence Strength', fontsize=20)
    ax.set_ylabel('Uncertainty (K/∑α)', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/figure3_evidence_vs_uncertainty.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_figure4_evidence_entropy(data):
    """Figure 4: Evidence Entropy Distribution"""
    print("  Creating Figure 4: Evidence Entropy...")

    fig, ax = plt.subplots(figsize=(7, 6))

    alphas = data['alphas']
    sample_types = data['sample_types']

    known_mask = np.array(sample_types) == 'known'
    unknown_mask = np.array(sample_types) == 'unknown'

    # Calculate entropy
    def calculate_entropy(alphas_subset):
        probs = alphas_subset / np.sum(alphas_subset, axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        return entropy

    known_entropy = calculate_entropy(alphas[known_mask])
    unknown_entropy = calculate_entropy(alphas[unknown_mask])

    ax.hist(known_entropy, bins=50, alpha=0.7, label='Known', density=True, color='blue')
    ax.hist(unknown_entropy, bins=50, alpha=0.7, label='Unknown', density=True, color='red')

    ax.set_xlabel('Evidence Entropy', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/figure4_evidence_entropy.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_figure5_spectral_spatial_ssud(data):
    """Figure 5: Spectral vs Spatial Uncertainty with SSUD Decisions"""
    print("  Creating Figure 5: Spectral vs Spatial with SSUD...")

    fig, ax = plt.subplots(figsize=(7, 7))

    spec_unc = data['spectral_uncertainties']
    spat_unc = data['spatial_uncertainties']
    ssud_decisions = data['ssud_decisions']

    colors = ['blue', 'red', 'green']
    labels = ['Combined', 'Spectral', 'Spatial']

    for decision in [0, 1, 2]:
        mask = ssud_decisions == decision
        if np.sum(mask) > 0:
            ax.scatter(spec_unc[mask], spat_unc[mask],
                      c=colors[decision], alpha=0.6, s=15,
                      label=f'{labels[decision]} ({np.sum(mask)})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Spectral Uncertainty', fontsize=20)
    ax.set_ylabel('Spatial Uncertainty', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=16, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('outputs/figures/figure5_spectral_spatial_ssud.pdf', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_analysis()