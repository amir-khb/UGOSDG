import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = current_dir
project_root = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from config.config import device, worker_init_fn, g
from data.loaders import (
    load_and_preprocess_data,
    load_and_preprocess_pavia_data,
    load_and_preprocess_dioni_loukia_data
)
from data.datasets import HSIDataset, HSIDatasetDioni
from models.classifier import DCRN_SSUD_SIFD, HybridSSUDClassifierFixed
from training.losses import edl_mse_loss
from utils.helpers import setup_directories


def train_one_epoch_hyperparameter_tuning(models, optimizers, train_loader, epoch, hyperparameter_configs):
    """
    Train all models simultaneously with different hyperparameter configurations
    """
    # Set all models to training mode
    for model in models.values():
        model.train()

    total_losses = {name: 0 for name in models.keys()}
    component_losses = {name: {'cls': 0, 'edl': 0, 'domain': 0, 'recon': 0} for name in models.keys()}
    total_accuracies = {name: 0 for name in models.keys()}
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Train each model with its specific hyperparameters
        for name in models.keys():
            model = models[name]
            optimizer = optimizers[name]
            config = hyperparameter_configs[name]

            optimizer.zero_grad()

            outputs = model(data)

            # Standard classification loss
            cls_loss = F.cross_entropy(outputs['logits'], target)

            # EDL losses
            edl_combined, _, _ = edl_mse_loss(target, outputs['alpha_combined'], epoch)
            edl_spectral, _, _ = edl_mse_loss(target, outputs['alpha_spectral'], epoch)
            edl_spatial, _, _ = edl_mse_loss(target, outputs['alpha_spatial'], epoch)
            edl_total_loss = (edl_combined + config['alpha'] * (edl_spectral + edl_spatial)) / 3.0

            # Domain adversarial loss
            domain_loss = F.binary_cross_entropy_with_logits(
                outputs['sifd_outputs']['domain_pred'],
                torch.ones_like(outputs['sifd_outputs']['domain_pred']) * 0.5
            )

            # Reconstruction loss
            data_flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1])
            recon_loss = F.mse_loss(outputs['sifd_outputs']['reconstructed_spectrum'], data_flat)

            # Total loss with specific hyperparameters
            loss = cls_loss + config['alpha'] * edl_total_loss + config['beta'] * domain_loss + config[
                'gamma'] * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update metrics
            total_losses[name] += loss.item()
            component_losses[name]['cls'] += cls_loss.item()
            component_losses[name]['edl'] += edl_total_loss.item()
            component_losses[name]['domain'] += domain_loss.item()
            component_losses[name]['recon'] += recon_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], 1)
            correct = (predicted == target).sum().item()
            total_accuracies[name] += correct

        total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in models.keys():
        avg_loss = total_losses[name] / len(train_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        avg_components = {k: v / len(train_loader) for k, v in component_losses[name].items()}

        results[name] = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'components': avg_components
        }

    return results


def validate_hyperparameter_tuning(models, classifiers, val_loader, epoch):
    """
    Validate all models simultaneously
    """
    # Set all models to eval mode
    for model in models.values():
        model.eval()

    total_losses = {name: 0 for name in models.keys()}
    total_accuracies = {name: 0 for name in models.keys()}
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            for name in models.keys():
                model = models[name]
                outputs = model(data)

                # Calculate validation loss
                loss = F.cross_entropy(outputs['logits'], target)
                total_losses[name] += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs['logits'], 1)
                correct = (predicted == target).sum().item()
                total_accuracies[name] += correct

            total_samples += target.size(0)

    # Calculate average metrics
    results = {}
    for name in models.keys():
        avg_loss = total_losses[name] / len(val_loader)
        avg_acc = 100.0 * total_accuracies[name] / total_samples
        results[name] = {'loss': avg_loss, 'accuracy': avg_acc}

    return results


def create_hyperparameter_models(input_channels, patch_size, num_classes, num_models=5):
    """Create multiple identical models for hyperparameter testing"""
    models = {}
    for i in range(num_models):
        model_name = f'Model_{i + 1}'
        models[model_name] = DCRN_SSUD_SIFD(
            input_channels=input_channels,
            patch_size=patch_size,
            num_classes=num_classes
        ).to(device)
    return models


def create_hyperparameter_optimizers(models, lr=0.00001):
    """Create optimizers for all models"""
    optimizers = {}
    for name, model in models.items():
        optimizer = optim.Adam([
            {'params': model.dcrn.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr},
            {'params': model.sifd.parameters(), 'lr': lr},
        ], weight_decay=1e-5)
        optimizers[name] = optimizer
    return optimizers


def run_hyperparameter_experiment(experiment_name, hyperparameter_configs, dataset_loaders,
                                  input_channels, num_classes, epochs=5):
    """
    Run a single hyperparameter experiment
    """
    print(f"\n{'=' * 80}")
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"{'=' * 80}")

    for name, config in hyperparameter_configs.items():
        print(f"{name}: alpha={config['alpha']:.3f}, beta={config['beta']:.3f}, gamma={config['gamma']:.3f}")
    print()

    train_loader, val_loader = dataset_loaders

    # Create models
    models = create_hyperparameter_models(input_channels, 7, num_classes, len(hyperparameter_configs))

    # Rename models to match config names
    model_names = list(hyperparameter_configs.keys())
    renamed_models = {}
    for i, (old_name, model) in enumerate(models.items()):
        new_name = model_names[i]
        renamed_models[new_name] = model
    models = renamed_models

    # Create classifiers
    classifiers = {}
    for name, model in models.items():
        classifiers[name] = HybridSSUDClassifierFixed(
            model, num_classes=num_classes,
            uncertainty_threshold=0.4111,
            decoupling_threshold=0.1,
            spectral_weight=0.7
        ).to(device)

    # Create optimizers
    optimizers = create_hyperparameter_optimizers(models)

    # Training history
    history = {name: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
               for name in models.keys()}

    best_models = {name: None for name in models.keys()}
    best_val_accs = {name: 0 for name in models.keys()}

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train all models
        train_results = train_one_epoch_hyperparameter_tuning(
            models, optimizers, train_loader, epoch, hyperparameter_configs
        )

        # Validate all models
        val_results = validate_hyperparameter_tuning(models, classifiers, val_loader, epoch)

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

        # Print progress
        print("Validation accuracies:")
        for name in models.keys():
            config = hyperparameter_configs[name]
            print(
                f"  {name} (α={config['alpha']:.3f}, β={config['beta']:.3f}, γ={config['gamma']:.3f}): {val_results[name]['accuracy']:.2f}%")

    # Final results based on validation only
    final_results = {}
    for name in models.keys():
        final_results[name] = {
            'hyperparameters': hyperparameter_configs[name],
            'best_val_acc': best_val_accs[name],
            'final_val_loss': history[name]['val_loss'][-1],
            'final_train_acc': history[name]['train_acc'][-1]
        }

    return final_results, history


def plot_hyperparameter_results(results, experiment_name, varied_param, fixed_params, dataset_name):
    """
    Create comprehensive plots for hyperparameter experiment results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{experiment_name} - Varying {varied_param.upper()} - {dataset_name}', fontsize=16, fontweight='bold')

    # Extract data for plotting
    param_values = []
    val_accs = []
    val_losses = []
    train_accs = []

    for name, result in results.items():
        param_values.append(result['hyperparameters'][varied_param])
        val_accs.append(result['best_val_acc'])
        val_losses.append(result['final_val_loss'])
        train_accs.append(result['final_train_acc'])

    # Sort by parameter value for cleaner plots
    sorted_indices = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_indices]
    val_accs = [val_accs[i] for i in sorted_indices]
    val_losses = [val_losses[i] for i in sorted_indices]
    train_accs = [train_accs[i] for i in sorted_indices]

    # Plot 1: Validation Accuracy
    axes[0, 0].plot(param_values, val_accs, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel(f'{varied_param.upper()}')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Validation Accuracy vs ' + varied_param.upper())
    axes[0, 0].grid(True, alpha=0.3)

    # Highlight best validation accuracy
    best_val_idx = np.argmax(val_accs)
    axes[0, 0].scatter(param_values[best_val_idx], val_accs[best_val_idx],
                       color='red', s=100, zorder=5, label=f'Best: {param_values[best_val_idx]:.3f}')
    axes[0, 0].legend()

    # Plot 2: Validation Loss
    axes[0, 1].plot(param_values, val_losses, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel(f'{varied_param.upper()}')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss vs ' + varied_param.upper())
    axes[0, 1].grid(True, alpha=0.3)

    # Highlight best validation loss
    best_loss_idx = np.argmin(val_losses)
    axes[0, 1].scatter(param_values[best_loss_idx], val_losses[best_loss_idx],
                       color='red', s=100, zorder=5, label=f'Best: {param_values[best_loss_idx]:.3f}')
    axes[0, 1].legend()

    # Plot 3: Training vs Validation Accuracy
    axes[1, 0].plot(param_values, train_accs, 'go-', linewidth=2, markersize=8, label='Training Accuracy')
    axes[1, 0].plot(param_values, val_accs, 'bo-', linewidth=2, markersize=8, label='Validation Accuracy')
    axes[1, 0].set_xlabel(f'{varied_param.upper()}')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training vs Validation Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')

    table_data = []
    headers = [varied_param.upper(), 'Val Acc', 'Val Loss', 'Train Acc']

    for i in range(len(param_values)):
        row = [
            f'{param_values[i]:.3f}',
            f'{val_accs[i]:.2f}%',
            f'{val_losses[i]:.4f}',
            f'{train_accs[i]:.2f}%'
        ]
        table_data.append(row)

    table = axes[1, 1].table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight best row
    best_val_row = best_val_idx + 1

    for col in range(len(headers)):
        table[(best_val_row, col)].set_facecolor('#ccffcc')  # Light green for best val acc

    axes[1, 1].set_title('Results Summary\n(Green: Best Validation Accuracy)')

    # Add fixed parameters info
    fixed_str = ', '.join([f'{k}={v:.3f}' for k, v in fixed_params.items()])
    fig.text(0.5, 0.02, f'Fixed parameters: {fixed_str}', ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)

    return fig


def load_dataset(dataset_name):
    """Load and return dataset loaders and metadata"""
    if dataset_name.lower() == "houston":
        print(f"\n🏙️ Loading Houston dataset...")
        (hs_image_2013_norm, gt_2013,
         hs_image_2018_norm, labels_2018_downsampled_new,
         train_labels_2013_new, val_labels_2013_new) = load_and_preprocess_data()

        source_train_dataset = HSIDataset(hs_image_2013_norm, train_labels_2013_new,
                                          patch_size=7, augment=True)
        source_val_dataset = HSIDataset(hs_image_2013_norm, val_labels_2013_new,
                                        patch_size=7, augment=False)

        input_channels = hs_image_2013_norm.shape[2]
        num_classes = 7

    elif dataset_name.lower() == "pavia":
        print(f"\n🏛️ Loading Pavia dataset...")
        (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
         hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

        source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u,
                                          patch_size=7, augment=True)
        source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u,
                                        patch_size=7, augment=False)

        input_channels = hs_pavia_u_norm.shape[2]
        num_classes = 7

    elif dataset_name.lower() == "dioni":
        print(f"\n🌊 Loading Dioni-Loukia dataset...")
        (hs_dioni_norm, train_gt_dioni, val_gt_dioni,
         hs_loukia_norm, gt_loukia_remapped) = load_and_preprocess_dioni_loukia_data()

        source_train_dataset = HSIDatasetDioni(hs_dioni_norm, train_gt_dioni,
                                               patch_size=7, augment=True)
        source_val_dataset = HSIDatasetDioni(hs_dioni_norm, val_gt_dioni,
                                             patch_size=7, augment=False)

        input_channels = hs_dioni_norm.shape[2]
        num_classes = 9
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create data loaders
    train_loader = DataLoader(source_train_dataset, batch_size=32, shuffle=True,
                              num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(source_val_dataset, batch_size=32, shuffle=False,
                            num_workers=0, worker_init_fn=worker_init_fn, generator=g)

    return (train_loader, val_loader), input_channels, num_classes


def run_all_hyperparameter_experiments():
    """
    Run hyperparameter tuning experiments for alpha, beta, and gamma on all three datasets
    """
    print("=" * 100)
    print("HYPERPARAMETER TUNING FOR ALL DATASETS (VALIDATION-BASED)")
    print("Loss function: cls_loss + alpha*edl_loss + beta*domain_loss + gamma*recon_loss")
    print("=" * 100)

    # Define datasets to test
    datasets = ["Houston", "Pavia", "Dioni"]

    # Define base hyperparameters
    base_alpha = 0.3
    base_beta = 0.5
    base_gamma = 0.3

    # Define experiment configurations
    experiments = [
        {
            'name': 'Alpha_Tuning',
            'varied_param': 'alpha',
            'values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fixed_params': {'beta': base_beta, 'gamma': base_gamma}
        },
        {
            'name': 'Beta_Tuning',
            'varied_param': 'beta',
            'values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fixed_params': {'alpha': base_alpha, 'gamma': base_gamma}
        },
        {
            'name': 'Gamma_Tuning',
            'varied_param': 'gamma',
            'values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fixed_params': {'alpha': base_alpha, 'beta': base_beta}
        }
    ]

    all_dataset_results = {}
    dataset_best_params = {}

    # Run experiments for each dataset
    for dataset_name in datasets:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'=' * 80}")

        # Load dataset
        dataset_loaders, input_channels, num_classes = load_dataset(dataset_name)

        dataset_results = {}
        dataset_best_params[dataset_name] = {}

        # Run each experiment on this dataset
        for exp_config in experiments:
            exp_name = exp_config['name']
            varied_param = exp_config['varied_param']
            values = exp_config['values']
            fixed_params = exp_config['fixed_params']

            print(f"\n{'=' * 60}")
            print(f"DATASET: {dataset_name} - EXPERIMENT: {exp_name}")
            print(f"Varying {varied_param} in {values}")
            print(f"Fixed: {fixed_params}")
            print(f"{'=' * 60}")

            # Create hyperparameter configurations
            hyperparameter_configs = {}
            for i, value in enumerate(values):
                config_name = f'{varied_param}_{value:.1f}'
                config = fixed_params.copy()
                config[varied_param] = value
                hyperparameter_configs[config_name] = config

            # Run experiment
            results, history = run_hyperparameter_experiment(
                f"{dataset_name}_{exp_name}", hyperparameter_configs, dataset_loaders,
                input_channels, num_classes, epochs=5
            )

            # Store results
            dataset_results[exp_name] = {
                'results': results,
                'history': history,
                'config': exp_config
            }

            # Find best parameter for this experiment
            best_val_acc = 0
            best_config = None
            best_param_value = None

            for config_name, result in results.items():
                if result['best_val_acc'] > best_val_acc:
                    best_val_acc = result['best_val_acc']
                    best_config = result['hyperparameters']
                    best_param_value = result['hyperparameters'][varied_param]

            dataset_best_params[dataset_name][varied_param] = {
                'value': best_param_value,
                'val_acc': best_val_acc,
                'config': best_config
            }

            # Create and save plots
            fig = plot_hyperparameter_results(results, exp_name, varied_param, fixed_params, dataset_name)

            # Save plot
            plot_path = f'outputs/figures/hyperparameter_tuning_{dataset_name.lower()}_{exp_name.lower()}.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"📊 Plot saved: {plot_path}")

            # Print experiment summary
            print(f"\n📊 {dataset_name} - {exp_name} RESULTS SUMMARY:")
            print("-" * 50)

            for config_name, result in results.items():
                param_val = result['hyperparameters'][varied_param]
                val_acc = result['best_val_acc']
                print(f"{varied_param}={param_val:.1f}: Val Acc={val_acc:.2f}%")

            print(f"\n🏆 Best {varied_param}: {best_param_value:.1f} (Val Acc: {best_val_acc:.2f}%)")

        all_dataset_results[dataset_name] = dataset_results

    # Create comprehensive summary
    create_comprehensive_summary(all_dataset_results, dataset_best_params)

    # Save all results
    save_comprehensive_results(all_dataset_results, dataset_best_params)

    print(f"\n{'=' * 100}")
    print("🎉 ALL HYPERPARAMETER TUNING EXPERIMENTS COMPLETED!")
    print("📁 Results saved in outputs/figures/ and outputs/models/")
    print("📊 Check the comprehensive summary for optimal hyperparameter values")
    print("=" * 100)

    return all_dataset_results, dataset_best_params


def create_comprehensive_summary(all_dataset_results, dataset_best_params):
    """Create a comprehensive summary plot comparing all datasets and parameters"""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Hyperparameter Tuning Summary - All Datasets', fontsize=20, fontweight='bold')

    datasets = list(all_dataset_results.keys())
    parameters = ['alpha', 'beta', 'gamma']

    # Colors for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # Plot 1: Best validation accuracy for each parameter across datasets
    ax1 = axes[0, 0]
    x_pos = np.arange(len(parameters))
    width = 0.25

    for i, dataset in enumerate(datasets):
        val_accs = [dataset_best_params[dataset][param]['val_acc'] for param in parameters]
        ax1.bar(x_pos + i * width, val_accs, width, label=dataset, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Hyperparameters')
    ax1.set_ylabel('Best Validation Accuracy (%)')
    ax1.set_title('Best Validation Accuracy by Parameter and Dataset')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels([p.upper() for p in parameters])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best parameter values across datasets
    ax2 = axes[0, 1]
    for i, dataset in enumerate(datasets):
        param_values = [dataset_best_params[dataset][param]['value'] for param in parameters]
        ax2.plot(parameters, param_values, marker=markers[i], linewidth=2, markersize=8,
                 label=dataset, color=colors[i])

    ax2.set_xlabel('Hyperparameters')
    ax2.set_ylabel('Optimal Parameter Value')
    ax2.set_title('Optimal Parameter Values Across Datasets')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary table for alpha parameter
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')

    alpha_data = []
    headers = ['Dataset', 'Best α', 'Val Acc (%)', 'β (fixed)', 'γ (fixed)']

    for dataset in datasets:
        best_alpha = dataset_best_params[dataset]['alpha']
        row = [
            dataset,
            f"{best_alpha['value']:.3f}",
            f"{best_alpha['val_acc']:.2f}",
            f"{best_alpha['config']['beta']:.3f}",
            f"{best_alpha['config']['gamma']:.3f}"
        ]
        alpha_data.append(row)

    table = ax3.table(cellText=alpha_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    ax3.set_title('Alpha Parameter Optimization Results', fontsize=14, pad=20)

    # Plot 4: Summary table for beta and gamma
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    beta_gamma_data = []
    headers = ['Dataset', 'Best β', 'β Val Acc', 'Best γ', 'γ Val Acc']

    for dataset in datasets:
        best_beta = dataset_best_params[dataset]['beta']
        best_gamma = dataset_best_params[dataset]['gamma']
        row = [
            dataset,
            f"{best_beta['value']:.3f}",
            f"{best_beta['val_acc']:.2f}%",
            f"{best_gamma['value']:.3f}",
            f"{best_gamma['val_acc']:.2f}%"
        ]
        beta_gamma_data.append(row)

    table = ax4.table(cellText=beta_gamma_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    ax4.set_title('Beta and Gamma Parameter Optimization Results', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('outputs/figures/comprehensive_hyperparameter_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Comprehensive summary plot saved: outputs/figures/comprehensive_hyperparameter_summary.png")


def save_comprehensive_results(all_dataset_results, dataset_best_params):
    """Save all experiment results to files"""

    # Save best parameters summary
    best_params_summary = {}
    for dataset_name, params in dataset_best_params.items():
        best_params_summary[dataset_name] = {}
        for param_name, param_info in params.items():
            best_params_summary[dataset_name][param_name] = {
                'optimal_value': float(param_info['value']),
                'validation_accuracy': float(param_info['val_acc']),
                'complete_config': {k: float(v) for k, v in param_info['config'].items()}
            }

    with open('outputs/models/best_hyperparameters_summary.json', 'w') as f:
        json.dump(best_params_summary, f, indent=2)

    # Save detailed results
    detailed_results = {}
    for dataset_name, dataset_data in all_dataset_results.items():
        detailed_results[dataset_name] = {}
        for exp_name, exp_data in dataset_data.items():
            detailed_results[dataset_name][exp_name] = {}
            for config_name, result in exp_data['results'].items():
                detailed_results[dataset_name][exp_name][config_name] = {
                    'hyperparameters': result['hyperparameters'],
                    'best_val_acc': float(result['best_val_acc']),
                    'final_val_loss': float(result['final_val_loss']),
                    'final_train_acc': float(result['final_train_acc'])
                }

    with open('outputs/models/detailed_hyperparameter_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Print final summary to console
    print(f"\n{'=' * 80}")
    print("FINAL HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'=' * 80}")

    for dataset_name, params in dataset_best_params.items():
        print(f"\n📊 {dataset_name.upper()} DATASET:")
        print("-" * 40)
        optimal_alpha = params['alpha']['value']
        optimal_beta = params['beta']['value']
        optimal_gamma = params['gamma']['value']

        print(f"🎯 Optimal α (EDL weight): {optimal_alpha:.3f} (Val Acc: {params['alpha']['val_acc']:.2f}%)")
        print(f"🎯 Optimal β (Domain weight): {optimal_beta:.3f} (Val Acc: {params['beta']['val_acc']:.2f}%)")
        print(f"🎯 Optimal γ (Recon weight): {optimal_gamma:.3f} (Val Acc: {params['gamma']['val_acc']:.2f}%)")
        print(f"📝 Recommended config: α={optimal_alpha:.3f}, β={optimal_beta:.3f}, γ={optimal_gamma:.3f}")

    print(f"\n💾 Best parameters summary saved: outputs/models/best_hyperparameters_summary.json")
    print(f"💾 Detailed results saved: outputs/models/detailed_hyperparameter_results.json")


if __name__ == "__main__":
    setup_directories()

    # Run all hyperparameter tuning experiments
    all_results, best_params = run_all_hyperparameter_experiments()