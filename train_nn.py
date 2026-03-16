#!/usr/bin/env python3
"""
Residual Neural Network: ARIC Visit 5 -> Visit 7 Prediction
=============================================================
Paper Reference: Section 3.8 -- Residual Neural Network, Eq. 14

This module implements a small residual MLP that predicts Visit 7 clinical
variables from Visit 5 clinical variables, learning the disease progression
mapping:

    V7_hat = W_skip * V5 + g_phi(V5)       (Eq. 14)

where:
    - W_skip is a learnable linear skip connection (n_features x n_features)
      that captures the dominant "identity plus linear drift" component of
      disease progression. For most variables, V7 ~ V5 (things don't change
      dramatically over 6 years), so the skip connection provides a strong
      baseline that the nonlinear branch only needs to refine.
    - g_phi(V5) is a nonlinear residual branch (MLP with residual blocks)
      that captures the nonlinear, interaction-driven components of disease
      progression (e.g., the accelerating decline when heart failure and
      kidney disease co-exist).

Architecture overview (Section 3.8):
    Input (D features) -> BatchNorm -> Linear(D, H) -> ReLU
    -> [ResidualBlock(H)] x N_blocks -> Linear(H, D)   [this is g_phi]
    + Linear(D, D) applied to raw V5                     [this is W_skip]
    = V7_hat (D features)

The composite loss (Section 3.8, "Training Objective") combines:
    1. Variance-normalized, clinically-weighted MSE
    2. Direction-of-change penalty (does the prediction correctly capture
       whether each variable increased or decreased from V5 to V7?)

Usage:
    python train_nn.py                          # train on cohort_data.npz
    python train_nn.py --data cohort_data.npz --epochs 200
    python train_nn.py --evaluate models/v5_to_v7_best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Ensure project root is importable for config.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# config.py provides:
#   ARIC_VARIABLES   -- dict of variable metadata including 'weight' (clinical importance)
#   NUMERIC_VAR_NAMES -- ordered list of ~100 numeric ARIC variable names
#   N_FEATURES        -- len(NUMERIC_VAR_NAMES), the input/output dimensionality
#   NN_DEFAULTS       -- default hyperparameters (hidden_dim, n_blocks, dropout, lr, etc.)
from config import ARIC_VARIABLES, NUMERIC_VAR_NAMES, N_FEATURES, NN_DEFAULTS


# ===========================================================================
# Model Architecture
# ===========================================================================
#
# Paper Reference: Section 3.8, "Residual Neural Network Architecture"
#
# The model uses residual connections at two levels:
#   1. MACRO-LEVEL: V7 = W_skip(V5) + g_phi(V5)
#      The skip connection from input to output ensures the network only
#      needs to learn the DELTA (disease progression) rather than the
#      full V7 values. This is critical because V5 and V7 are highly
#      correlated (most clinical values don't change dramatically over
#      6 years), so learning the full mapping from scratch would waste
#      most capacity on the trivial "identity" component.
#
#   2. MICRO-LEVEL: Within g_phi, each ResidualBlock uses x + f(x)
#      skip connections, enabling gradient flow through deep networks
#      and preventing degradation (He et al., 2016).
# ===========================================================================

class ResidualBlock(nn.Module):
    """
    Two fully-connected layers with BatchNorm, ReLU, Dropout, and skip connection.

    Paper Reference: Section 3.8, "Residual Blocks"

    Architecture:
        x -> Linear(H, H) -> BatchNorm -> ReLU -> Dropout -> Linear(H, H) -> BatchNorm
        |                                                                              |
        +-----------------------------------(skip)-------------------------------------+
        -> ReLU(sum)

    Design choices:
    - BatchNorm after each Linear (before activation): stabilizes training by
      normalizing internal activations, reducing sensitivity to initialization
      and learning rate. Particularly important here because input features
      span vastly different scales (LVEF in [20,70]% vs NT-proBNP in [50,5000] pg/mL).
    - Dropout (default 0.1): regularization to prevent overfitting on the
      synthetic training data. Moderate rate because the synthetic data has
      limited noise (it's generated from a deterministic model, so the main
      source of variation is the sampling distributions, not measurement noise).
    - Pre-activation ordering (BN before ReLU) follows the original ResNet
      convention for fully-connected residual blocks.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        dim : int
            Hidden dimension (both input and output, since this is a
            same-dimension residual block).
        dropout : float
            Dropout probability. Default 0.1 = 10% of neurons dropped
            during training.
        """
        super().__init__()

        # The two-layer "residual function" f(x).
        # Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN
        self.net = nn.Sequential(
            nn.Linear(dim, dim),       # First FC layer (H -> H)
            nn.BatchNorm1d(dim),       # Normalize activations
            nn.ReLU(),                 # Nonlinearity
            nn.Dropout(dropout),       # Regularization (only active during training)
            nn.Linear(dim, dim),       # Second FC layer (H -> H)
            nn.BatchNorm1d(dim),       # Normalize before skip addition
        )

        # Final ReLU applied AFTER adding the skip connection:
        # output = ReLU(x + f(x))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass: x + f(x), followed by ReLU.

        The skip connection (x +) ensures gradients flow directly through
        the block even if f(x) has vanishing gradients, enabling training
        of deeper networks.
        """
        return self.relu(x + self.net(x))


class V5toV7Net(nn.Module):
    """
    Residual MLP implementing Eq. 14: V7_hat = W_skip * V5 + g_phi(V5).

    Paper Reference: Section 3.8, "Network Architecture"

    The architecture separates the prediction into two branches:
      1. SKIP BRANCH: A single linear layer (n_features -> n_features) that
         learns the best linear approximation to V5 -> V7 mapping. This
         captures the dominant behavior: most variables at V7 are close to
         their V5 values plus a small linear drift.
      2. NONLINEAR BRANCH: An MLP with residual blocks that captures the
         nonlinear interactions (e.g., how worsening GFR + elevated E/e'
         together predict worse NT-proBNP than either alone).

    The nonlinear branch processes V5 through:
      Input BatchNorm -> Linear projection (D -> H) -> ReLU
      -> [ResidualBlock(H)] x N_blocks
      -> Linear projection (H -> D)

    Then the final output is: nonlinear_branch(V5) + skip_branch(V5).
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of input/output features (D). Default from config.N_FEATURES (~100).
        hidden_dim : int
            Width of hidden layers (H). Default 256 provides sufficient capacity
            to model nonlinear interactions among ~100 features without being
            so large that it overfits the synthetic data.
        n_blocks : int
            Number of residual blocks in the nonlinear branch. Default 3 gives
            6 total hidden layers (2 per block), which empirically balances
            expressiveness and training stability.
        dropout : float
            Dropout probability in residual blocks. Default 0.1.
        """
        super().__init__()

        # Input BatchNorm: normalize raw V5 features to zero mean, unit variance.
        # This is essential because input features have wildly different scales:
        #   LVEF ~ [20, 70] (%)
        #   NT-proBNP ~ [50, 5000] (pg/mL)
        #   E/e' ~ [4, 20] (ratio)
        # Without normalization, features with larger magnitudes would dominate
        # the gradients, causing the network to effectively ignore small-scale
        # but clinically important features.
        self.input_norm = nn.BatchNorm1d(n_features)

        # Project from feature space (D) to hidden space (H).
        # H > D allows the network to create an overcomplete representation
        # where nonlinear feature interactions can be more easily separated.
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # Stack of residual blocks forming the nonlinear branch g_phi.
        # Each block has 2 FC layers + skip connection, for a total of
        # 2 * n_blocks hidden layers in the nonlinear branch.
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Project from hidden space (H) back to feature space (D).
        # This is the output of the nonlinear branch g_phi(V5).
        self.output_proj = nn.Linear(hidden_dim, n_features)

        # Learnable skip connection: W_skip * V5 (Eq. 14).
        # This is a FULL linear layer (not just identity) because:
        #   a) Some variables scale systematically from V5 to V7 (e.g., age
        #      increases by ~6 years, GFR tends to decline by ~10%).
        #   b) Linear cross-variable effects exist (e.g., V5 diabetes
        #      linearly predicts V7 eGFR decline).
        #   c) The identity is a special case (W_skip = I) that the optimizer
        #      can learn if the data warrants it.
        self.skip = nn.Linear(n_features, n_features)

    def forward(self, v5):
        """
        Forward pass implementing Eq. 14: V7_hat = g_phi(V5) + W_skip * V5.

        Parameters
        ----------
        v5 : Tensor of shape (batch_size, n_features)
            Visit 5 clinical variable vectors.

        Returns
        -------
        Tensor of shape (batch_size, n_features)
            Predicted Visit 7 clinical variable vectors.
        """
        # Nonlinear branch: normalize -> project up -> residual blocks -> project down
        h = torch.relu(self.input_proj(self.input_norm(v5)))  # (B, D) -> (B, H)
        for block in self.blocks:
            h = block(h)                                       # (B, H) -> (B, H)
        # Eq. 14: g_phi(V5) + W_skip * V5
        return self.output_proj(h) + self.skip(v5)             # (B, D) + (B, D) -> (B, D)


# ===========================================================================
# Loss Function
# ===========================================================================
#
# Paper Reference: Section 3.8, "Training Objective"
#
# The composite loss has two components:
#   1. Variance-normalized, clinically-weighted MSE: ensures all variables
#      contribute proportionally to the loss regardless of their scale,
#      with clinical importance weights giving higher priority to key
#      variables (e.g., LVEF, eGFR, NT-proBNP get weight 2.0 vs 0.3
#      for less critical variables like RA diameter).
#   2. Direction-of-change penalty: penalizes predictions that get the
#      SIGN of the V5->V7 change wrong. This is clinically important
#      because a model that predicts LVEF increases when it actually
#      decreases is making a qualitatively wrong prediction, even if
#      the absolute error is small.
# ===========================================================================

class CompositeLoss(nn.Module):
    """
    Weighted MSE with direction-of-change penalty.

    Paper Reference: Section 3.8, "Training Objective"

    Loss = weighted_normalized_MSE + direction_weight * direction_penalty

    where:
        weighted_normalized_MSE = mean( ((V7_hat - V7) / std_train)^2 * w )
        direction_penalty = mean( I[sign(V7_hat - V5) != sign(V7 - V5)] * w )

    The normalized MSE divides by the training set standard deviation of each
    variable, ensuring that variables with large absolute scales (e.g.,
    NT-proBNP in pg/mL) don't dominate the loss over variables with small
    scales (e.g., E/e' ratio). The clinical weights (w) then prioritize
    variables by medical importance rather than statistical variance.

    The direction penalty is a differentiable approximation: it uses the
    indicator function I[true_delta * pred_delta < 0] (sign disagrees),
    converted to float and weighted. While not differentiable at exactly
    zero, the MSE component provides smooth gradients everywhere, and the
    direction penalty acts as a regularizer that nudges predictions toward
    correct sign behavior.
    """

    def __init__(self, var_weights: torch.Tensor, training_std: torch.Tensor,
                 direction_weight: float = 0.1):
        """
        Parameters
        ----------
        var_weights : Tensor of shape (n_features,)
            Per-variable clinical importance weights from config.ARIC_VARIABLES.
            Higher weight = variable contributes more to the loss.
            Typical range: 0.3 (e.g., RA diameter) to 2.0 (e.g., LVEF, eGFR).
        training_std : Tensor of shape (n_features,)
            Standard deviation of each variable in the training set.
            Used to normalize MSE so all variables contribute equally
            (before clinical weighting is applied).
        direction_weight : float
            Weight of the direction-of-change penalty relative to the MSE.
            Default 0.1 means direction penalty contributes ~10% of the loss.
            Higher values prioritize getting the sign right over exact magnitude.
        """
        super().__init__()

        # register_buffer stores tensors that should move with the model to
        # GPU/CPU but are NOT learnable parameters (no gradients).
        self.register_buffer('var_weights', var_weights)

        # Clamp std to avoid division by zero for constant-valued variables
        # (e.g., a variable that doesn't vary in the training set).
        self.register_buffer('training_std', training_std.clamp(min=1e-6))

        # Relative weight of direction penalty vs MSE
        self.direction_weight = direction_weight

    def forward(self, pred_v7, true_v7, v5):
        """
        Compute the composite loss.

        Parameters
        ----------
        pred_v7 : Tensor (batch_size, n_features)
            Model's predicted V7 variables.
        true_v7 : Tensor (batch_size, n_features)
            Ground-truth V7 variables from the synthetic cohort.
        v5 : Tensor (batch_size, n_features)
            V5 variables (needed for computing direction of change).

        Returns
        -------
        Tensor (scalar)
            Combined loss value.
        """
        # ----- Component 1: Variance-normalized, weighted MSE -----
        # Divide residuals by training std to normalize each variable's
        # contribution. A 1-std error on any variable contributes equally
        # (before weighting). Then multiply by clinical weights.
        diff = (pred_v7 - true_v7) / self.training_std
        weighted_mse = (diff ** 2 * self.var_weights).mean()

        # ----- Component 2: Direction-of-change penalty -----
        # Compute true and predicted deltas from V5 to V7.
        true_delta = true_v7 - v5     # What actually changed
        pred_delta = pred_v7 - v5     # What the model predicts changed

        # Identify where the predicted sign differs from the true sign.
        # true_delta * pred_delta < 0 means they have opposite signs
        # (one positive, one negative). Convert boolean to float for
        # arithmetic. Note: this is 0 when both are positive, both
        # negative, or either is exactly zero.
        direction_mismatch = (true_delta * pred_delta < 0).float()

        # Weight mismatches by clinical importance and average over
        # all variables and batch samples.
        direction_penalty = (direction_mismatch * self.var_weights).mean()

        # ----- Combined loss -----
        # MSE handles magnitude accuracy, direction penalty handles
        # qualitative correctness (did the variable go up or down?).
        return weighted_mse + self.direction_weight * direction_penalty


# ===========================================================================
# Data Loading & Preprocessing
# ===========================================================================
#
# Paper Reference: Section 3.8, "Data Preparation"
#
# The cohort_data.npz file (produced by synthetic_cohort.py) contains:
#   v5: (N, D) array of V5 clinical variables
#   v7: (N, D) array of V7 clinical variables
#   var_names: (D,) array of variable name strings
#
# We split into train (70%) / val (15%) / test (15%) with a fixed shuffle
# seed for reproducibility.
# ===========================================================================

def load_data(data_path: str):
    """
    Load cohort_data.npz and split into train/val/test sets.

    Parameters
    ----------
    data_path : str
        Path to the .npz file produced by synthetic_cohort.py.

    Returns
    -------
    dict with keys:
        'train': (train_v5, train_v7) -- training arrays
        'val':   (val_v5, val_v7)     -- validation arrays
        'test':  (test_v5, test_v7)   -- test arrays
        'var_names': list of str      -- variable names
        'training_std': ndarray       -- per-variable std from training set
        'weights': ndarray            -- per-variable clinical importance weights
    """
    # Load the compressed NumPy archive
    data = np.load(data_path, allow_pickle=True)
    v5 = data['v5'].astype(np.float32)   # Ensure float32 for PyTorch compatibility
    v7 = data['v7'].astype(np.float32)
    var_names = list(data['var_names'])

    # Replace NaN/Inf with 0. This handles any edge cases that slipped
    # through the finite-check in synthetic_cohort.py (e.g., from loading
    # corrupted data or different generation runs).
    v5 = np.nan_to_num(v5, nan=0.0, posinf=0.0, neginf=0.0)
    v7 = np.nan_to_num(v7, nan=0.0, posinf=0.0, neginf=0.0)

    # Split sizes based on NN_DEFAULTS (70/15/15 by default)
    n = len(v5)
    n_train = int(n * NN_DEFAULTS['train_frac'])   # 70% for training
    n_val = int(n * NN_DEFAULTS['val_frac'])       # 15% for validation (early stopping)
    # Remaining ~15% for test (final evaluation)

    # Deterministic shuffle with fixed seed=42 so that train/val/test splits
    # are identical across runs, even if the cohort generation seed differs.
    # This is important for reproducible comparisons of hyperparameter tuning.
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    v5, v7 = v5[idx], v7[idx]

    # Split into three sets
    train_v5, train_v7 = v5[:n_train], v7[:n_train]
    val_v5, val_v7 = v5[n_train:n_train+n_val], v7[n_train:n_train+n_val]
    test_v5, test_v7 = v5[n_train+n_val:], v7[n_train+n_val:]

    # Compute per-variable standard deviation from the TRAINING set only.
    # This is used to normalize the MSE loss (see CompositeLoss).
    # Using only training stats prevents information leakage from val/test.
    training_std = train_v7.std(axis=0)

    # Load clinical importance weights from config.ARIC_VARIABLES.
    # Each variable has a 'weight' field (0.3 to 2.0) reflecting its
    # clinical importance for cardiorenal disease assessment. Variables
    # with weight 2.0 (LVEF, eGFR, NT-proBNP, GLS, E/e') are the most
    # important for diagnosis and prognosis. Default weight is 0.5 for
    # any variable not explicitly listed.
    weights = np.array([
        ARIC_VARIABLES.get(vn, {}).get('weight', 0.5) for vn in var_names
    ], dtype=np.float32)

    return {
        'train': (train_v5, train_v7),
        'val': (val_v5, val_v7),
        'test': (test_v5, test_v7),
        'var_names': var_names,
        'training_std': training_std,
        'weights': weights,
    }


def make_loaders(data_dict, batch_size: int = 256):
    """
    Create PyTorch DataLoaders from the data dictionary.

    Parameters
    ----------
    data_dict : dict
        Output of load_data() containing 'train', 'val', 'test' splits.
    batch_size : int
        Mini-batch size. Default 256 balances GPU utilization (larger batches
        = better GPU efficiency) with gradient noise (smaller batches =
        more stochastic gradient steps per epoch = faster convergence).

    Returns
    -------
    dict of {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    loaders = {}
    for split in ['train', 'val', 'test']:
        v5, v7 = data_dict[split]

        # Wrap in TensorDataset: each __getitem__ returns (v5_row, v7_row)
        ds = TensorDataset(torch.from_numpy(v5), torch.from_numpy(v7))

        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            # Shuffle training data each epoch for SGD (prevents the model
            # from learning order-dependent artifacts). Val/test are not
            # shuffled (deterministic evaluation).
            shuffle=(split == 'train'),
            # drop_last=True for training: ensures all batches are the same
            # size (the last partial batch is dropped). This prevents the
            # final small batch from having disproportionate influence on
            # BatchNorm statistics and gradient estimates.
            drop_last=(split == 'train'),
        )
    return loaders


# ===========================================================================
# Training Loop
# ===========================================================================
#
# Paper Reference: Section 3.8, "Training Procedure"
#
# Training uses:
#   - AdamW optimizer: Adam with decoupled weight decay (Loshchilov & Hutter,
#     2019). Weight decay regularizes the model by penalizing large weights,
#     reducing overfitting. Decoupled (vs. L2 in standard Adam) provides
#     more consistent regularization across parameters with different
#     gradient magnitudes.
#   - Cosine annealing learning rate schedule: smoothly decays the learning
#     rate from lr to ~0 over the total number of epochs. This allows large
#     steps early (fast initial convergence) and small steps late (fine-tuning
#     near the optimum).
#   - Gradient clipping (max norm 1.0): prevents gradient explosions that
#     can occur when a batch contains outlier patients with extreme clinical
#     values. Clips the global gradient norm to 1.0, preserving gradient
#     direction but limiting step size.
#   - Early stopping (patience=20): monitors validation loss and stops
#     training if it hasn't improved for 20 consecutive epochs. This
#     prevents overfitting by stopping before the model memorizes training
#     noise. The best model (lowest val loss) is saved to disk.
# ===========================================================================

def train(
    data_path: str = 'cohort_data.npz',
    hidden_dim: int = NN_DEFAULTS['hidden_dim'],
    n_blocks: int = NN_DEFAULTS['n_blocks'],
    dropout: float = NN_DEFAULTS['dropout'],
    lr: float = NN_DEFAULTS['lr'],
    weight_decay: float = NN_DEFAULTS['weight_decay'],
    epochs: int = NN_DEFAULTS['epochs'],
    batch_size: int = NN_DEFAULTS['batch_size'],
    patience: int = NN_DEFAULTS['patience'],
    save_dir: str = 'models',
):
    """
    Train the V5 -> V7 residual network.

    Paper Reference: Section 3.8

    Parameters
    ----------
    data_path : str
        Path to cohort_data.npz (from synthetic_cohort.py).
    hidden_dim : int
        Width of hidden layers in the nonlinear branch. Default 256.
    n_blocks : int
        Number of residual blocks. Default 3.
    dropout : float
        Dropout probability. Default 0.1.
    lr : float
        Initial learning rate for AdamW. Default 1e-3.
    weight_decay : float
        Weight decay (L2 penalty) for AdamW. Default 1e-4.
        Regularizes against overfitting to synthetic training data.
    epochs : int
        Maximum number of training epochs. Default 200.
        Actual training often stops earlier due to early stopping.
    batch_size : int
        Mini-batch size. Default 256.
    patience : int
        Early stopping patience: number of epochs without validation
        improvement before stopping. Default 20.
    save_dir : str
        Directory to save the best model checkpoint.

    Returns
    -------
    str
        Path to the saved best model checkpoint.
    """
    # Select device: use GPU if available, otherwise CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load and prepare data
    # -----------------------------------------------------------------------
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, data_path) if not os.path.isabs(data_path) else data_path
    data = load_data(data_path)
    loaders = make_loaders(data, batch_size)

    n_features = len(data['var_names'])
    print(f"Features: {n_features}, Train: {len(data['train'][0])}, "
          f"Val: {len(data['val'][0])}, Test: {len(data['test'][0])}")

    # -----------------------------------------------------------------------
    # Initialize model
    # -----------------------------------------------------------------------
    model = V5toV7Net(n_features, hidden_dim, n_blocks, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    # Typical model size: ~256K params for hidden_dim=256, n_blocks=3, D=100.
    # This is deliberately small to prevent overfitting to the synthetic data
    # and to enable fast inference for the agentic loop.

    # -----------------------------------------------------------------------
    # Initialize loss function
    # -----------------------------------------------------------------------
    # Move clinical weights and training std to the same device as the model.
    var_weights = torch.from_numpy(data['weights']).to(device)
    training_std = torch.from_numpy(data['training_std'].astype(np.float32)).to(device)
    criterion = CompositeLoss(var_weights, training_std).to(device)

    # -----------------------------------------------------------------------
    # Optimizer and learning rate scheduler
    # -----------------------------------------------------------------------
    # AdamW: Adam with decoupled weight decay. Better than Adam + L2 because
    # it applies weight decay directly to the parameters (not through the
    # gradient), which interacts more predictably with the adaptive learning
    # rates in Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing: lr decays smoothly from `lr` to ~0 over `epochs`.
    # This schedule works well with early stopping: if we stop at epoch 80/200,
    # the LR has only decayed to ~cos(80/200 * pi) * lr/2, not to zero.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # -----------------------------------------------------------------------
    # Training loop setup
    # -----------------------------------------------------------------------
    best_val_loss = float('inf')    # Track best validation loss for early stopping
    epochs_no_improve = 0            # Counter for early stopping patience
    save_path = os.path.join(base, save_dir, 'v5_to_v7_best.pt')
    os.makedirs(os.path.join(base, save_dir), exist_ok=True)

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------
    for epoch in range(1, epochs + 1):

        # === Training phase ===
        model.train()   # Enable dropout and BatchNorm training mode
        train_loss = 0.0

        for v5_batch, v7_batch in loaders['train']:
            # Move batch to device (GPU/CPU)
            v5_batch, v7_batch = v5_batch.to(device), v7_batch.to(device)

            # Forward pass: predict V7 from V5 using Eq. 14
            pred = model(v5_batch)

            # Compute composite loss (normalized MSE + direction penalty)
            loss = criterion(pred, v7_batch, v5_batch)

            # Backward pass: compute gradients
            optimizer.zero_grad()   # Clear accumulated gradients from previous step
            loss.backward()         # Backpropagate through the network

            # Gradient clipping: prevent gradient explosions.
            # Max norm of 1.0 clips the GLOBAL gradient norm (L2 norm of
            # all parameter gradients concatenated). If the norm exceeds 1.0,
            # all gradients are scaled down proportionally to have norm exactly 1.0.
            # This is important because outlier patients (e.g., very low Sf_act
            # + very low Kf) can produce extreme clinical values that cause
            # large loss gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Parameter update step
            optimizer.step()

            # Accumulate loss (weighted by batch size for correct averaging)
            train_loss += loss.item() * len(v5_batch)

        # Average training loss over all training samples
        train_loss /= len(data['train'][0])

        # === Validation phase ===
        model.eval()    # Disable dropout, use running BatchNorm statistics
        val_loss = 0.0

        with torch.no_grad():  # No gradient computation needed for validation
            for v5_batch, v7_batch in loaders['val']:
                v5_batch, v7_batch = v5_batch.to(device), v7_batch.to(device)
                pred = model(v5_batch)
                loss = criterion(pred, v7_batch, v5_batch)
                val_loss += loss.item() * len(v5_batch)

        # Average validation loss
        val_loss /= len(data['val'][0])

        # Step the learning rate scheduler (cosine annealing)
        scheduler.step()

        # Print progress every 10 epochs (and at epoch 1 for sanity check)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        # === Early stopping check ===
        if val_loss < best_val_loss:
            # New best validation loss: save model checkpoint
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save full checkpoint: model weights + hyperparameters + metadata.
            # This allows loading the model later without knowing the original
            # hyperparameters (they're stored in the checkpoint).
            torch.save({
                'model_state': model.state_dict(),     # Learned weights
                'n_features': n_features,               # For reconstructing model architecture
                'hidden_dim': hidden_dim,
                'n_blocks': n_blocks,
                'dropout': dropout,
                'var_names': data['var_names'],          # Variable ordering (critical for inference)
                'training_std': data['training_std'],    # For loss function reconstruction
                'weights': data['weights'],              # Clinical importance weights
                'best_val_loss': best_val_loss,          # For tracking training quality
                'epoch': epoch,                          # Which epoch was best
            }, save_path)
        else:
            # No improvement: increment patience counter
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # Patience exhausted: stop training to prevent overfitting.
                # The model has been getting worse on validation data for
                # `patience` consecutive epochs, suggesting it's starting
                # to memorize training data rather than learning generalizable
                # patterns.
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}, saved to {save_path}")

    # -----------------------------------------------------------------------
    # Final evaluation on held-out test set
    # -----------------------------------------------------------------------
    # Reload the BEST model (not the final model, which may have overfit)
    # from the checkpoint saved during training.
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    evaluate_model(model, data, device)

    return save_path


# ===========================================================================
# Evaluation
# ===========================================================================
#
# Paper Reference: Section 3.8, "Model Evaluation"
#
# The evaluation computes:
#   1. Per-variable R^2 and MAE on the held-out test set
#   2. Direction-of-change accuracy (what fraction of V5->V7 changes are
#      predicted with the correct sign?)
#   3. Summary statistics across all variables
# ===========================================================================

def load_trained_model(model_path: str, device=None):
    """
    Load a trained V5toV7Net from a saved checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the .pt checkpoint file saved during training.
    device : torch.device or None
        Device to load the model onto. Defaults to CUDA if available.

    Returns
    -------
    model : V5toV7Net
        Loaded model in eval mode.
    checkpoint : dict
        Full checkpoint dictionary (contains var_names, training_std, etc.).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint (weights_only=False needed because checkpoint contains
    # non-tensor data like var_names list)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Reconstruct model architecture from saved hyperparameters
    model = V5toV7Net(
        n_features=checkpoint['n_features'],
        hidden_dim=checkpoint['hidden_dim'],
        n_blocks=checkpoint['n_blocks'],
        dropout=checkpoint['dropout'],
    ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state'])

    # Set to evaluation mode (disables dropout, uses running BN stats)
    model.eval()

    return model, checkpoint


def predict(model, v5_array: np.ndarray, device=None) -> np.ndarray:
    """
    Run V5 -> V7 prediction on new data.

    This is the inference entry point used by the agentic framework
    (Section 3.9) to predict V7 from V5 for individual patients or batches.

    Parameters
    ----------
    model : V5toV7Net
        Trained model (should be in eval mode).
    v5_array : ndarray of shape (N, n_features) or (n_features,)
        V5 clinical variables. Single patient (1D) or batch (2D).
    device : torch.device or None
        Device for computation. Defaults to model's current device.

    Returns
    -------
    ndarray of shape (N, n_features) or (n_features,)
        Predicted V7 variables. Same shape convention as input.
    """
    if device is None:
        device = next(model.parameters()).device

    # Handle single-patient (1D) input by adding batch dimension
    single = v5_array.ndim == 1
    if single:
        v5_array = v5_array[np.newaxis, :]   # (D,) -> (1, D)

    # Run inference with no gradient tracking (faster, less memory)
    with torch.no_grad():
        v5_t = torch.from_numpy(v5_array.astype(np.float32)).to(device)
        pred = model(v5_t).cpu().numpy()

    # Return single-patient output as 1D array
    return pred[0] if single else pred


def evaluate_model(model, data_dict, device):
    """
    Compute per-variable metrics on the held-out test set.

    Paper Reference: Section 3.8, "Model Evaluation Metrics"

    Computes and prints:
      1. Per-variable R^2 and MAE for key clinical variables
      2. Direction-of-change accuracy (across all variables where V5 != V7)
      3. Summary statistics (mean/median R^2, count of variables above thresholds)

    Parameters
    ----------
    model : V5toV7Net
        Trained model (best checkpoint).
    data_dict : dict
        Output of load_data() containing 'test' split and 'var_names'.
    device : torch.device
        Device for computation.
    """
    model.eval()   # Ensure evaluation mode
    test_v5, test_v7 = data_dict['test']
    var_names = data_dict['var_names']

    # Generate predictions for the entire test set
    pred_v7 = predict(model, test_v5, device)

    print(f"\n{'='*70}")
    print(f"  Test Set Evaluation ({len(test_v5)} patients, {len(var_names)} variables)")
    print(f"{'='*70}")

    # -----------------------------------------------------------------------
    # Per-variable R^2 and MAE
    # -----------------------------------------------------------------------
    # R^2 (coefficient of determination): measures how much variance in V7
    # is explained by the model. R^2 = 1 - SS_res/SS_tot.
    #   R^2 = 1.0: perfect prediction
    #   R^2 = 0.0: model is no better than predicting the mean
    #   R^2 < 0.0: model is WORSE than predicting the mean (bad)
    #
    # MAE (mean absolute error): average magnitude of prediction errors
    # in the variable's native units.
    r2_list, mae_list = [], []

    # Key clinical variables to display (the most important for
    # cardiorenal disease assessment and clinical decision-making)
    key_vars = ['LVEF_pct', 'MAP_mmHg', 'GFR_mL_min', 'eGFR_mL_min_173m2',
                'E_e_prime_avg', 'CO_Lmin', 'SBP_mmHg', 'NTproBNP_pg_mL',
                'serum_creatinine_mg_dL', 'GLS_pct', 'LVEDV_mL', 'LVESV_mL']

    for i, vn in enumerate(var_names):
        true = test_v7[:, i]
        pred = pred_v7[:, i]

        # R^2: 1 - (sum of squared residuals) / (total sum of squares)
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)   # max prevents div-by-zero

        # MAE: mean absolute error in native units
        mae = np.mean(np.abs(true - pred))

        r2_list.append(r2)
        mae_list.append(mae)

        # Print metrics for key variables only (to keep output readable)
        if vn in key_vars:
            print(f"  {vn:35s}  R^2={r2:.3f}  MAE={mae:.2f}")

    # -----------------------------------------------------------------------
    # Direction-of-change accuracy
    # -----------------------------------------------------------------------
    # For each variable and patient, did the model correctly predict whether
    # the value went up or down from V5 to V7? This is a clinically
    # important metric because a model that predicts "eGFR improved" when
    # it actually declined is making a qualitatively wrong prediction.
    true_delta = test_v7 - test_v5    # Shape: (N_test, D)
    pred_delta = pred_v7 - test_v5    # Shape: (N_test, D)

    # Only evaluate direction accuracy where the true change is non-trivial
    # (|delta| > 1e-6). Tiny changes could be numerical noise.
    mask = np.abs(true_delta) > 1e-6
    if mask.any():
        # Correct direction: true_delta and pred_delta have the same sign
        # (both positive or both negative), OR the variable didn't change
        # meaningfully (masked out).
        correct_dir = ((true_delta * pred_delta) > 0) | (~mask)
        dir_acc = correct_dir[mask].mean()
        print(f"\n  Direction-of-change accuracy: {dir_acc:.1%}")

    # -----------------------------------------------------------------------
    # Summary statistics across all variables
    # -----------------------------------------------------------------------
    r2_arr = np.array(r2_list)
    print(f"\n  Overall R^2 -- mean: {r2_arr.mean():.3f}, median: {np.median(r2_arr):.3f}")
    print(f"  Variables with R^2 > 0.8: {(r2_arr > 0.8).sum()}/{len(r2_arr)}")
    print(f"  Variables with R^2 > 0.5: {(r2_arr > 0.5).sum()}/{len(r2_arr)}")


# ===========================================================================
# CLI (Command-Line Interface)
# ===========================================================================
#
# Supports two modes:
#   1. Training mode (default): loads data, trains model, evaluates on test set.
#   2. Evaluation mode (--evaluate): loads a trained model and evaluates only.
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train V5 -> V7 residual NN')
    parser.add_argument('--data', type=str, default='cohort_data.npz',
                        help='Path to cohort_data.npz')
    parser.add_argument('--epochs', type=int, default=NN_DEFAULTS['epochs'],
                        help='Max training epochs (default: 200)')
    parser.add_argument('--hidden_dim', type=int, default=NN_DEFAULTS['hidden_dim'],
                        help='Hidden layer width (default: 256)')
    parser.add_argument('--n_blocks', type=int, default=NN_DEFAULTS['n_blocks'],
                        help='Number of residual blocks (default: 3)')
    parser.add_argument('--dropout', type=float, default=NN_DEFAULTS['dropout'],
                        help='Dropout probability (default: 0.1)')
    parser.add_argument('--lr', type=float, default=NN_DEFAULTS['lr'],
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=NN_DEFAULTS['batch_size'],
                        help='Mini-batch size (default: 256)')
    parser.add_argument('--patience', type=int, default=NN_DEFAULTS['patience'],
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--evaluate', type=str, default=None,
                        help='Path to trained model for evaluation only (skip training)')
    args = parser.parse_args()

    if args.evaluate:
        # Evaluation-only mode: load trained model and evaluate on test set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, ckpt = load_trained_model(args.evaluate, device)
        data = load_data(args.data)
        evaluate_model(model, data, device)
    else:
        # Training mode: train from scratch and evaluate
        train(
            data_path=args.data,
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )


if __name__ == '__main__':
    main()
