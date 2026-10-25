import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# Original SIFD for reference
class OriginalSIFD(nn.Module):
    """Original SIFD module"""

    def __init__(self, input_channels, feature_dim=64):
        super(OriginalSIFD, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.freq_size = input_channels // 2 + 1

        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_real = x_freq_complex.real
        x_freq_imag = x_freq_complex.imag
        x_freq = torch.cat([x_freq_real, x_freq_imag], dim=1)
        x_freq_mag = torch.sqrt(x_freq_real ** 2 + x_freq_imag ** 2)

        freq_features = self.freq_encoder(x_freq)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 1: w/o Frequency Transform - Use spatial domain features directly
class SIFD_NoFreqTransform(nn.Module):
    """SIFD without frequency transform - processes spatial domain directly"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_NoFreqTransform, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # Direct spatial processing instead of frequency
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        # Process in spatial domain directly (no frequency transform)
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        # Direct spatial encoding
        spatial_features = self.spatial_encoder(x_flat)
        invariant_features = self.invariant_extractor(spatial_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        # Dummy freq_info for compatibility
        freq_info = {'magnitude': torch.zeros(batch_size * num_pixels, channels // 2 + 1, device=x.device)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 2: w/o Attention Mechanism - Remove attention weighting
class SIFD_NoAttention(nn.Module):
    """SIFD without attention mechanism"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_NoAttention, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # No attention mechanism

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_real = x_freq_complex.real
        x_freq_imag = x_freq_complex.imag
        x_freq = torch.cat([x_freq_real, x_freq_imag], dim=1)
        x_freq_mag = torch.sqrt(x_freq_real ** 2 + x_freq_imag ** 2)

        freq_features = self.freq_encoder(x_freq)
        invariant_features = self.invariant_extractor(freq_features)

        # No attention - use features directly
        weighted_features = invariant_features

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 3: w/o Domain Regularization - Remove domain adversarial loss
class SIFD_NoDomainReg(nn.Module):
    """SIFD without domain regularization"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_NoDomainReg, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # No domain classifier - removed domain regularization

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_real = x_freq_complex.real
        x_freq_imag = x_freq_complex.imag
        x_freq = torch.cat([x_freq_real, x_freq_imag], dim=1)
        x_freq_mag = torch.sqrt(x_freq_real ** 2 + x_freq_imag ** 2)

        freq_features = self.freq_encoder(x_freq)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        # No domain prediction - return dummy tensor
        domain_pred = torch.zeros(batch_size * num_pixels, 1, device=x.device)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 4: w/o Reconstruction Loss - Remove spectrum reconstruction
class SIFD_NoRecon(nn.Module):
    """SIFD without reconstruction loss"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_NoRecon, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # No spectrum decoder - removed reconstruction

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_real = x_freq_complex.real
        x_freq_imag = x_freq_complex.imag
        x_freq = torch.cat([x_freq_real, x_freq_imag], dim=1)
        x_freq_mag = torch.sqrt(x_freq_real ** 2 + x_freq_imag ** 2)

        freq_features = self.freq_encoder(x_freq)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)

        # No reconstruction - return dummy tensor
        reconstructed_spectrum = torch.zeros(batch_size * num_pixels, channels, device=x.device)

        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 5: FFT → DCT - Replace FFT with Discrete Cosine Transform
class SIFD_DCT(nn.Module):
    """SIFD using DCT instead of FFT"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_DCT, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # DCT produces only real values, so single channel input
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def simple_dct(self, x):
        """Simple DCT implementation using matrix multiplication"""
        N = x.size(-1)
        k = torch.arange(N, device=x.device, dtype=torch.float32)
        n = torch.arange(N, device=x.device, dtype=torch.float32)

        # DCT matrix
        dct_matrix = torch.cos(torch.pi * k.unsqueeze(1) * (2 * n + 1) / (2 * N))
        dct_matrix[0] *= np.sqrt(1 / N)
        dct_matrix[1:] *= np.sqrt(2 / N)

        # Apply DCT
        return torch.matmul(x, dct_matrix.T)

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        # Use simple DCT instead of FFT
        x_dct = self.simple_dct(x_flat)

        # For magnitude compatibility
        x_freq_mag = torch.abs(x_dct)

        freq_features = self.freq_encoder(x_dct)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 6: FFT → Wavelet - Replace FFT with simple wavelet-like transform
class SIFD_Wavelet(nn.Module):
    """SIFD using simple wavelet-like transform instead of FFT"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_Wavelet, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # Simple wavelet-like transform using convolution
        self.wavelet_approx = nn.Conv1d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.wavelet_detail = nn.Conv1d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)

        # Initialize with simple wavelet-like filters
        with torch.no_grad():
            self.wavelet_approx.weight[0, 0] = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Low-pass
            self.wavelet_detail.weight[0, 0] = torch.tensor([0.5, -0.5, 0.5, -0.5])  # High-pass

        # Wavelet produces approximation + detail coefficients
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),  # approx + detail
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        # Apply simple wavelet-like transform
        approx = self.wavelet_approx(x_flat)
        detail = self.wavelet_detail(x_flat)

        # Ensure same length for both
        min_len = min(approx.size(2), detail.size(2))
        approx = approx[:, :, :min_len]
        detail = detail[:, :, :min_len]

        x_wavelet = torch.cat([approx, detail], dim=1)

        # For magnitude compatibility
        x_freq_mag = torch.sqrt(approx ** 2 + detail ** 2)

        freq_features = self.freq_encoder(x_wavelet)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 7: Real Part Only - Use only real components of FFT
class SIFD_RealOnly(nn.Module):
    """SIFD using only real part of FFT"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_RealOnly, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # Only real part, so single channel input
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_real = x_freq_complex.real  # Only use real part

        # For magnitude compatibility
        x_freq_mag = torch.abs(x_freq_real)

        freq_features = self.freq_encoder(x_freq_real)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Ablation 8: Imaginary Part Only - Use only imaginary components of FFT
class SIFD_ImagOnly(nn.Module):
    """SIFD using only imaginary part of FFT"""

    def __init__(self, input_channels, feature_dim=64):
        super(SIFD_ImagOnly, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # Only imaginary part, so single channel input
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.invariant_extractor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.spectrum_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_channels),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        num_pixels = h * w

        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_pixels, 1, channels)

        x_freq_complex = torch.fft.rfft(x_flat, dim=2)
        x_freq_imag = x_freq_complex.imag  # Only use imaginary part

        # For magnitude compatibility
        x_freq_mag = torch.abs(x_freq_imag)

        freq_features = self.freq_encoder(x_freq_imag)
        invariant_features = self.invariant_extractor(freq_features)
        att_weights = self.attention(invariant_features)
        weighted_features = invariant_features * att_weights

        domain_features = weighted_features.mean(dim=2)
        domain_pred = self.domain_classifier(domain_features)

        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(2)
        reconstructed_spectrum = self.spectrum_decoder(pooled_features)
        modulation = self.projector(pooled_features)
        modulation = modulation.view(batch_size, h, w, channels).permute(0, 3, 1, 2)

        freq_info = {'magnitude': x_freq_mag.reshape(batch_size * num_pixels, -1)}

        return {
            'invariant_features': modulation,
            'domain_pred': domain_pred,
            'reconstructed_spectrum': reconstructed_spectrum,
            'original_shape': (batch_size, h, w),
            'freq_info': freq_info
        }


# Dictionary to easily access all ablation variants
SIFD_ABLATION_VARIANTS = {
    'Original': OriginalSIFD,
    'NoFreqTransform': SIFD_NoFreqTransform,
    'NoAttention': SIFD_NoAttention,
    'NoDomainReg': SIFD_NoDomainReg,
    'NoRecon': SIFD_NoRecon,
    'DCT': SIFD_DCT,
    'Wavelet': SIFD_Wavelet,
    'RealOnly': SIFD_RealOnly,
    'ImagOnly': SIFD_ImagOnly
}