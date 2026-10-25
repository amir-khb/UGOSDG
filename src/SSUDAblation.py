"""
SSUD Ablation Study - Parallel Experiments

Requirements:
- All existing project dependencies
- tqdm: pip install tqdm

This script runs all SSUD ablation experiments in parallel with progress bars.
"""

import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import time
import json
import random
from collections import defaultdict
from tqdm import tqdm

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
project_root = current_dir
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Import project modules
from src.config.config import device, worker_init_fn, g, new_classesPavia
from src.data.loaders import load_and_preprocess_pavia_data
from src.data.datasets import HSIDataset, HSITestDataset
from src.models.classifier import DCRN_SSUD_SIFD, HybridSSUDClassifierFixed
from src.training.losses import edl_mse_loss
from src.training.evaluator import test_open_set_hybrid
from src.utils.helpers import setup_directories


class SSUDAblationClassifier1_FixedWeights(torch.nn.Module):
    """Ablation 1: Fixed Pathway Weights (w1=w2=0.5)"""

    def __init__(self, model, num_classes=7, uncertainty_threshold=0.4111,
                 decoupling_threshold=0.1, spectral_weight=0.7):
        super(SSUDAblationClassifier1_FixedWeights, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.decoupling_threshold = decoupling_threshold
        self.spectral_weight = spectral_weight
        self.probability_weight = 1.0

    def predict(self, x, return_uncertainties=False):
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # ABLATION: Use FIXED pathway weights (0.5, 0.5) instead of adaptive
            fixed_weights = torch.tensor([[0.5, 0.5]], device=x.device).repeat(x.size(0), 1)

            # Get uncertainty measures
            combined_uncertainty = outputs['uncertainty_combined'].view(-1)
            spectral_reliability = outputs['spectral_reliability'].view(-1)
            spatial_reliability = outputs['spatial_reliability'].view(-1)

            # Apply decoupling logic (same as original)
            reliability_diff = torch.abs(spectral_reliability - spatial_reliability)
            decoupling_mask = reliability_diff > self.decoupling_threshold

            final_uncertainty = combined_uncertainty.clone()
            use_spectral = spectral_reliability > spatial_reliability

            # Use fixed weights in decoupling decisions
            spectral_indices = torch.where(decoupling_mask & use_spectral)[0]
            if len(spectral_indices) > 0:
                spectral_uncertainty = 1.0 - spectral_reliability[spectral_indices]
                scaled_spatial_uncertainty = 0.7 * (1.0 - spatial_reliability[spectral_indices])
                final_uncertainty[spectral_indices] = torch.maximum(spectral_uncertainty, scaled_spatial_uncertainty)

            spatial_indices = torch.where(decoupling_mask & ~use_spectral)[0]
            if len(spatial_indices) > 0:
                spatial_uncertainty = 1.0 - spatial_reliability[spatial_indices]
                scaled_spectral_uncertainty = 0.7 * (1.0 - spectral_reliability[spatial_indices])
                final_uncertainty[spatial_indices] = torch.maximum(spatial_uncertainty, scaled_spectral_uncertainty)

            # Calculate rejection score
            rejection_score = self.spectral_weight * final_uncertainty + self.probability_weight * (1.0 - max_probs)
            unknown_mask = rejection_score > self.uncertainty_threshold

            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'uncertainty': final_uncertainty,
                    'rejection_score': rejection_score,
                    'unknown_mask': unknown_mask
                }
            return final_predictions


class SSUDAblationClassifier2_NoDecoupling(torch.nn.Module):
    """Ablation 2: Without Reliability Decoupling (always use combined pathway)"""

    def __init__(self, model, num_classes=7, uncertainty_threshold=0.4111,
                 decoupling_threshold=0.1, spectral_weight=0.7):
        super(SSUDAblationClassifier2_NoDecoupling, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.spectral_weight = spectral_weight
        self.probability_weight = 1.0

    def predict(self, x, return_uncertainties=False):
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # ABLATION: Always use combined uncertainty (no decoupling)
            final_uncertainty = outputs['uncertainty_combined'].view(-1)

            # Calculate rejection score
            rejection_score = self.spectral_weight * final_uncertainty + self.probability_weight * (1.0 - max_probs)
            unknown_mask = rejection_score > self.uncertainty_threshold

            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'uncertainty': final_uncertainty,
                    'rejection_score': rejection_score,
                    'unknown_mask': unknown_mask
                }
            return final_predictions


class SSUDAblationClassifier3_SimpleAverage(torch.nn.Module):
    """Ablation 3: Simple Average Fusion of spectral and spatial uncertainties"""

    def __init__(self, model, num_classes=7, uncertainty_threshold=0.4111,
                 decoupling_threshold=0.1, spectral_weight=0.7):
        super(SSUDAblationClassifier3_SimpleAverage, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.spectral_weight = spectral_weight
        self.probability_weight = 1.0

    def predict(self, x, return_uncertainties=False):
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # ABLATION: Simple average of spectral and spatial uncertainties
            spectral_uncertainty = outputs['uncertainty_spectral'].view(-1)
            spatial_uncertainty = outputs['uncertainty_spatial'].view(-1)
            final_uncertainty = (spectral_uncertainty + spatial_uncertainty) / 2.0

            # Calculate rejection score
            rejection_score = self.spectral_weight * final_uncertainty + self.probability_weight * (1.0 - max_probs)
            unknown_mask = rejection_score > self.uncertainty_threshold

            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'uncertainty': final_uncertainty,
                    'rejection_score': rejection_score,
                    'unknown_mask': unknown_mask
                }
            return final_predictions


class SSUDAblationClassifier4_MaxUncertainty(torch.nn.Module):
    """Ablation 4: Max Uncertainty from pathways"""

    def __init__(self, model, num_classes=7, uncertainty_threshold=0.4111,
                 decoupling_threshold=0.1, spectral_weight=0.7):
        super(SSUDAblationClassifier4_MaxUncertainty, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.spectral_weight = spectral_weight
        self.probability_weight = 1.0

    def predict(self, x, return_uncertainties=False):
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # ABLATION: Take maximum of pathway uncertainties
            spectral_uncertainty = outputs['uncertainty_spectral'].view(-1)
            spatial_uncertainty = outputs['uncertainty_spatial'].view(-1)
            final_uncertainty = torch.maximum(spectral_uncertainty, spatial_uncertainty)

            # Calculate rejection score
            rejection_score = self.spectral_weight * final_uncertainty + self.probability_weight * (1.0 - max_probs)
            unknown_mask = rejection_score > self.uncertainty_threshold

            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'uncertainty': final_uncertainty,
                    'rejection_score': rejection_score,
                    'unknown_mask': unknown_mask
                }
            return final_predictions


class SSUDAblationClassifier5_NoConfidence(torch.nn.Module):
    """Ablation 5: Without Confidence Term (only uncertainty in rejection score)"""

    def __init__(self, model, num_classes=7, uncertainty_threshold=0.4111,
                 decoupling_threshold=0.1, spectral_weight=0.7):
        super(SSUDAblationClassifier5_NoConfidence, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.decoupling_threshold = decoupling_threshold
        self.spectral_weight = spectral_weight
        self.probability_weight = 0.0  # ABLATION: No confidence term

    def predict(self, x, return_uncertainties=False):
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # Use original SSUD logic but with no confidence term
            combined_uncertainty = outputs['uncertainty_combined'].view(-1)
            spectral_reliability = outputs['spectral_reliability'].view(-1)
            spatial_reliability = outputs['spatial_reliability'].view(-1)

            reliability_diff = torch.abs(spectral_reliability - spatial_reliability)
            decoupling_mask = reliability_diff > self.decoupling_threshold

            final_uncertainty = combined_uncertainty.clone()
            use_spectral = spectral_reliability > spatial_reliability

            spectral_indices = torch.where(decoupling_mask & use_spectral)[0]
            if len(spectral_indices) > 0:
                spectral_uncertainty = 1.0 - spectral_reliability[spectral_indices]
                scaled_spatial_uncertainty = 0.7 * (1.0 - spatial_reliability[spectral_indices])
                final_uncertainty[spectral_indices] = torch.maximum(spectral_uncertainty, scaled_spatial_uncertainty)

            spatial_indices = torch.where(decoupling_mask & ~use_spectral)[0]
            if len(spatial_indices) > 0:
                spatial_uncertainty = 1.0 - spatial_reliability[spatial_indices]
                scaled_spectral_uncertainty = 0.7 * (1.0 - spectral_reliability[spatial_indices])
                final_uncertainty[spatial_indices] = torch.maximum(spatial_uncertainty, scaled_spectral_uncertainty)

            # ABLATION: Only uncertainty term in rejection score
            rejection_score = self.spectral_weight * final_uncertainty  # No confidence term
            unknown_mask = rejection_score > self.uncertainty_threshold

            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'uncertainty': final_uncertainty,
                    'rejection_score': rejection_score,
                    'unknown_mask': unknown_mask
                }
            return final_predictions


class SSUDAblationClassifier6_NoUncertainty(torch.nn.Module):
    """Ablation 6: Without Uncertainty Term (only confidence in rejection score)"""

    def __init__(self, model, num_classes=7, uncertainty_threshold=0.4111,
                 decoupling_threshold=0.1, spectral_weight=0.7):
        super(SSUDAblationClassifier6_NoUncertainty, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold
        self.decoupling_threshold = decoupling_threshold
        self.spectral_weight = 0.0  # ABLATION: No uncertainty term
        self.probability_weight = 1.0

    def predict(self, x, return_uncertainties=False):
        with torch.no_grad():
            outputs = self.model(x)
            probs = outputs['probs']
            max_probs, predicted_class = torch.max(probs, 1)

            # Still compute uncertainty for analysis but don't use in rejection
            combined_uncertainty = outputs['uncertainty_combined'].view(-1)

            # ABLATION: Only confidence term in rejection score
            rejection_score = self.probability_weight * (1.0 - max_probs)  # No uncertainty term
            unknown_mask = rejection_score > self.uncertainty_threshold

            final_predictions = predicted_class.clone()
            final_predictions[unknown_mask] = self.num_classes

            if return_uncertainties:
                return {
                    'predictions': final_predictions,
                    'uncertainty': combined_uncertainty,
                    'rejection_score': rejection_score,
                    'unknown_mask': unknown_mask
                }
            return final_predictions


def create_original_classifier(model, num_classes):
    """Create original SSUD classifier - defined as function to avoid lambda pickling issues"""
    return HybridSSUDClassifierFixed(
        model,
        num_classes=num_classes,
        uncertainty_threshold=0.4111,
        decoupling_threshold=0.1,
        spectral_weight=0.7
    )


def set_seed_for_worker(seed):
    """Set seeds for a worker process"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch_ablation(model, classifier, train_loader, optimizer, epoch, ablation_name):
    """Training function for ablation study"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Create progress bar for training batches
    pbar = tqdm(train_loader, desc=f"{ablation_name} - Epoch {epoch}/50 [Train]",
                leave=False, ncols=100, position=1)

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        # Ensure target is LongTensor for EDL loss
        target = target.long()

        optimizer.zero_grad()

        outputs = model(data)

        # Classification loss
        cls_loss = F.cross_entropy(outputs['logits'], target)

        # EDL losses
        edl_combined, _, _ = edl_mse_loss(target, outputs['alpha_combined'], epoch)
        edl_spectral, _, _ = edl_mse_loss(target, outputs['alpha_spectral'], epoch)
        edl_spatial, _, _ = edl_mse_loss(target, outputs['alpha_spatial'], epoch)
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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs['logits'], 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Update progress bar
        current_acc = 100.0 * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    pbar.close()
    return total_loss / len(train_loader), 100.0 * correct / total


def validate_ablation(model, classifier, val_loader, epoch, ablation_name):
    """Validation function for ablation study"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # Create progress bar for validation batches
    pbar = tqdm(val_loader, desc=f"{ablation_name} - Epoch {epoch}/50 [Val]",
                leave=False, ncols=100, position=1)

    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # Ensure target is LongTensor for consistency
            target = target.long()

            outputs = model(data)
            loss = F.cross_entropy(outputs['logits'], target)
            val_loss += loss.item()

            _, predicted = torch.max(outputs['logits'], 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    pbar.close()
    return val_loss / len(val_loader), 100.0 * correct / total


def run_single_ablation_experiment(experiment_config):
    """Run a single ablation experiment"""
    ablation_name = experiment_config['name']
    classifier_class = experiment_config['classifier_class']
    seed = experiment_config['seed']

    print(f"🚀 Starting {ablation_name} experiment with seed {seed}")

    # Set seed for this worker
    set_seed_for_worker(seed)

    try:
        # Load Pavia data
        (hs_pavia_u_norm, train_gt_pavia_u, val_gt_pavia_u,
         hs_pavia_c_norm, gt_pavia_c_remapped) = load_and_preprocess_pavia_data()

        # Create datasets
        source_train_dataset = HSIDataset(hs_pavia_u_norm, train_gt_pavia_u, patch_size=7, augment=True)
        source_val_dataset = HSIDataset(hs_pavia_u_norm, val_gt_pavia_u, patch_size=7, augment=False)
        target_test_dataset = HSITestDataset(hs_pavia_c_norm, gt_pavia_c_remapped, patch_size=7)

        # Create data loaders
        train_loader = DataLoader(source_train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(source_val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(target_test_dataset, batch_size=32, shuffle=False, num_workers=0)

        input_channels = hs_pavia_u_norm.shape[2]
        num_classes = 7

        # Create model
        model = DCRN_SSUD_SIFD(input_channels=input_channels, patch_size=7, num_classes=num_classes).to(device)

        # Create ablation classifier
        if classifier_class == 'original':
            classifier = create_original_classifier(model, num_classes).to(device)
        else:
            classifier = classifier_class(model, num_classes=num_classes).to(device)

        # Create optimizer
        optimizer = optim.Adam([
            {'params': model.dcrn.parameters(), 'lr': 0.00001},
            {'params': model.classifier.parameters(), 'lr': 0.00001},
            {'params': model.spectral_evidence.parameters(), 'lr': 0.00001},
            {'params': model.spatial_evidence.parameters(), 'lr': 0.00001},
            {'params': model.combined_evidence.parameters(), 'lr': 0.00001},
            {'params': model.sifd.parameters(), 'lr': 0.00001},
        ], weight_decay=1e-5)

        # Training loop with progress bar
        best_val_acc = 0
        best_model_state = None
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Create progress bar for epochs
        epoch_pbar = tqdm(range(1, 51), desc=f"{ablation_name} Progress",
                          position=0, ncols=120, leave=True)

        for epoch in epoch_pbar:
            train_loss, train_acc = train_one_epoch_ablation(model, classifier, train_loader, optimizer, epoch,
                                                             ablation_name)
            val_loss, val_acc = validate_ablation(model, classifier, val_loader, epoch, ablation_name)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Acc': f'{train_acc:.2f}%',
                'Val Acc': f'{val_acc:.2f}%',
                'Best Val': f'{best_val_acc:.2f}%'
            })

        epoch_pbar.close()

        # Load best model and test with progress bar
        print(f"\n📊 {ablation_name}: Testing best model...")
        model.load_state_dict(best_model_state)

        # Add progress bar for test evaluation
        with tqdm(desc=f"{ablation_name} - Testing", ncols=100, leave=False) as test_pbar:
            known_acc, unknown_rej, hos_score, cm = test_open_set_hybrid(model, classifier, test_loader)
            test_pbar.update(1)

        result = {
            'ablation_name': ablation_name,
            'seed': seed,
            'best_val_acc': best_val_acc,
            'known_accuracy': known_acc,
            'unknown_rejection': unknown_rej,
            'hos_score': hos_score,
            'history': history,
            'confusion_matrix': cm.tolist()
        }

        print(
            f"✅ {ablation_name} completed: HOS = {hos_score:.2f}, Known Acc = {known_acc:.2f}%, Unknown Rej = {unknown_rej:.2f}%")
        return result

    except Exception as e:
        print(f"❌ Error in {ablation_name}: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {
            'ablation_name': ablation_name,
            'seed': seed,
            'error': str(e),
            'known_accuracy': 0,
            'unknown_rejection': 0,
            'hos_score': 0
        }


def run_parallel_ssud_ablation_study():
    """Run all SSUD ablation experiments in parallel"""
    print("=" * 80)
    print("SSUD MODULE ABLATION STUDY - PARALLEL EXECUTION")
    print("Dataset: Pavia University → Pavia Center")
    print("Epochs: 50, Seed: 42, Parallel Processing: Enabled")
    print("=" * 80)

    # Define ablation experiments - NO LAMBDA FUNCTIONS
    ablation_experiments = [
        {
            'name': 'Original_SSUD',
            'classifier_class': 'original',  # Special marker for original
            'seed': 42
        },
        {
            'name': 'Fixed_Pathway_Weights',
            'classifier_class': SSUDAblationClassifier1_FixedWeights,
            'seed': 42
        },
        {
            'name': 'No_Reliability_Decoupling',
            'classifier_class': SSUDAblationClassifier2_NoDecoupling,
            'seed': 42
        },
        {
            'name': 'Simple_Average_Fusion',
            'classifier_class': SSUDAblationClassifier3_SimpleAverage,
            'seed': 42
        },
        {
            'name': 'Max_Uncertainty',
            'classifier_class': SSUDAblationClassifier4_MaxUncertainty,
            'seed': 42
        },
        {
            'name': 'No_Confidence_Term',
            'classifier_class': SSUDAblationClassifier5_NoConfidence,
            'seed': 42
        },
        {
            'name': 'No_Uncertainty_Term',
            'classifier_class': SSUDAblationClassifier6_NoUncertainty,
            'seed': 42
        }
    ]

    # Set multiprocessing start method to 'spawn' to fix CUDA issues
    mp.set_start_method('spawn', force=True)

    # Use multiprocessing to run experiments in parallel
    start_time = time.time()

    print(f"\n🚀 Starting {len(ablation_experiments)} parallel experiments...")
    print("📊 Progress will be shown for each experiment individually")
    print("⏱️  Estimated time: ~15-20 minutes total\n")

    with mp.Pool(processes=min(len(ablation_experiments), mp.cpu_count())) as pool:
        # Use pool.map to run experiments in parallel
        results = pool.map(run_single_ablation_experiment, ablation_experiments)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'=' * 80}")
    print("SSUD ABLATION STUDY RESULTS")
    print(f"{'=' * 80}")
    print(f"✅ Completed experiments: {len(valid_results)}/{len(ablation_experiments)}")
    print(f"⏱️  Total execution time: {total_time / 60:.1f} minutes")
    print(f"🔄 Parallel processes used: {min(len(ablation_experiments), mp.cpu_count())}")
    print(f"{'=' * 80}")

    # Sort results by HOS score
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: x['hos_score'], reverse=True)

    if not valid_results:
        print("❌ No experiments completed successfully!")
        return None

    # Print results table with enhanced formatting
    print(f"\n🏆 RANKING BY HOS SCORE:")
    print("-" * 90)
    print(
        f"{'Rank':<4} {'Ablation':<25} {'HOS Score':<10} {'Known Acc':<11} {'Unknown Rej':<12} {'Val Acc':<9} {'Status':<6}")
    print("-" * 90)

    for i, result in enumerate(valid_results, 1):
        status = "✅" if result['hos_score'] > 0 else "⚠️"
        print(f"{i:<4} {result['ablation_name']:<25} {result['hos_score']:<10.2f} "
              f"{result['known_accuracy']:<11.2f} {result['unknown_rejection']:<12.2f} "
              f"{result['best_val_acc']:<9.2f} {status:<6}")

    # Show any failed experiments
    failed_results = [r for r in results if 'error' in r]
    if failed_results:
        print(f"\n❌ FAILED EXPERIMENTS ({len(failed_results)}):")
        print("-" * 50)
        for result in failed_results:
            print(f"   {result['ablation_name']}: {result['error'][:60]}...")

    print("-" * 90)

    # Print detailed analysis
    print(f"\n{'=' * 80}")
    print("DETAILED ANALYSIS:")
    print(f"{'=' * 80}")

    if valid_results:
        best_result = valid_results[0]
        worst_result = valid_results[-1]

        print(f"🏆 BEST PERFORMING ABLATION: {best_result['ablation_name']}")
        print(f"   HOS Score: {best_result['hos_score']:.2f}")
        print(f"   Known Class Accuracy: {best_result['known_accuracy']:.2f}%")
        print(f"   Unknown Rejection Rate: {best_result['unknown_rejection']:.2f}%")
        print(f"   Best Validation Accuracy: {best_result['best_val_acc']:.2f}%")

        print(f"\n📉 WORST PERFORMING ABLATION: {worst_result['ablation_name']}")
        print(f"   HOS Score: {worst_result['hos_score']:.2f}")
        print(f"   Performance drop: {best_result['hos_score'] - worst_result['hos_score']:.2f} points")

        # Component analysis
        original_result = next((r for r in valid_results if r['ablation_name'] == 'Original_SSUD'), None)
        if original_result:
            print(f"\n📊 COMPARISON WITH ORIGINAL SSUD:")
            print(f"   Original HOS Score: {original_result['hos_score']:.2f}")

            for result in valid_results:
                if result['ablation_name'] != 'Original_SSUD':
                    diff = result['hos_score'] - original_result['hos_score']
                    print(f"   {result['ablation_name']:<25}: {diff:+.2f} points")

    # Save results
    comprehensive_results = {
        'experiment_info': {
            'dataset': 'Pavia University → Pavia Center',
            'epochs': 50,
            'seed': 42,
            'execution_time_minutes': total_time / 60,
            'parallel_processing': True
        },
        'results': valid_results,
        'summary': {
            'best_ablation': valid_results[0]['ablation_name'] if valid_results else None,
            'best_hos_score': valid_results[0]['hos_score'] if valid_results else 0,
            'worst_ablation': valid_results[-1]['ablation_name'] if valid_results else None,
            'worst_hos_score': valid_results[-1]['hos_score'] if valid_results else 0
        }
    }

    # Save to JSON
    with open('outputs/models/ssud_ablation_parallel_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n💾 Results saved to: outputs/models/ssud_ablation_parallel_results.json")
    print(f"🎉 SSUD Ablation Study completed successfully with progress tracking!")
    print(f"📊 Used progress bars for: epoch training, batch processing, validation, and testing")

    return comprehensive_results


if __name__ == '__main__':
    # Check if tqdm is available
    try:
        from tqdm import tqdm
    except ImportError:
        print("❌ tqdm is required for progress bars. Install with: pip install tqdm")
        sys.exit(1)

    # Ensure directories exist
    setup_directories()

    # Run the parallel ablation study
    results = run_parallel_ssud_ablation_study()