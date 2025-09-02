#!/usr/bin/env python3
"""
Test script for the spectral monitoring system
"""

import torch
import torch.nn as nn
from Spectral_monitor import EnhancedMuonMonitor, SpectralAnalyzerWithMSign

def test_spectral_analyzer():
    """Test the core spectral analysis functions"""
    print("Testing SpectralAnalyzerWithMSign...")
    
    # Create test matrices
    torch.manual_seed(42)
    A = torch.randn(64, 32, dtype=torch.float32)
    
    analyzer = SpectralAnalyzerWithMSign()
    
    # Test 1: msign function
    print("  Testing msign function...")
    A_sign = analyzer.msign_newton_schulz(A, steps=5)
    orthogonality_error = torch.norm(A_sign @ A_sign.T - torch.eye(64))
    print(f"    Orthogonality error: {orthogonality_error:.6f}")
    assert orthogonality_error < 1e-3, "msign function not sufficiently orthogonal"
    
    # Test 2: Polar decomposition
    print("  Testing polar decomposition...")
    U, P = analyzer.polar_decomposition_via_msign(A, steps=5)
    reconstruction_error = torch.norm(A - U @ P)
    print(f"    Reconstruction error: {reconstruction_error:.6f}")
    assert reconstruction_error < 1e-3, "Polar decomposition reconstruction failed"
    
    # Test 3: Spectral norm computation
    print("  Testing spectral norm computation...")
    spectral_norm_msign = analyzer.compute_spectral_norm_via_msign(A)
    spectral_norm_svd = torch.linalg.svdvals(A)[0]
    relative_error = abs(spectral_norm_msign - spectral_norm_svd) / spectral_norm_svd
    print(f"    Spectral norm (msign): {spectral_norm_msign:.6f}")
    print(f"    Spectral norm (SVD): {spectral_norm_svd:.6f}")
    print(f"    Relative error: {relative_error * 100:.3f}%")
    assert relative_error < 0.05, "Spectral norm computation too inaccurate"
    
    print("  ✅ All spectral analyzer tests passed!")

def test_muon_monitor():
    """Test the Muon monitoring system"""
    print("Testing EnhancedMuonMonitor...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(256, 256, bias=False)
            self.mlp = nn.Linear(256, 1024, bias=False)
            
        def forward(self, x):
            return self.mlp(self.attn(x))
    
    model = TestModel()
    
    # Initialize monitor (disable wandb for testing)
    monitor = EnhancedMuonMonitor(
        model=model, 
        monitor_freq=1,
        use_wandb=False
    )
    
    # Create fake gradients and momentum states
    torch.manual_seed(42)
    x = torch.randn(10, 256)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Create fake optimizer states
    optimizer_state = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            optimizer_state[name] = {
                'momentum_buffer': torch.randn_like(param.grad) * 0.1
            }
    
    # Test monitoring
    print("  Testing monitor step...")
    metrics = monitor.monitor_training_step(
        loss=loss,
        optimizer_state=optimizer_state,
        learning_rate=0.01
    )
    
    print(f"  Collected {len(metrics)} metrics")
    assert len(metrics) > 0, "No metrics collected"
    
    # Test algorithm comparison
    print("  Testing algorithm comparison...")
    comparison = monitor.compare_newton_schulz_variants(model.attn.weight.grad)
    print(f"    Original error: {comparison['original_orthogonality_error']:.6f}")
    print(f"    MSign error: {comparison['msign_orthogonality_error']:.6f}")
    print(f"    Improvement ratio: {comparison['improvement_ratio']:.2f}x")
    
    print("  ✅ Monitor test passed!")

def test_integration():
    """Test integration with training-like scenario"""
    print("Testing integration scenario...")
    
    # Simple training setup
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 128, bias=False)
    )
    
    # Disable wandb for testing
    monitor = EnhancedMuonMonitor(
        model=model,
        monitor_freq=5,
        use_wandb=False
    )
    
    # Simulate training steps
    data = torch.randn(32, 128)
    target = torch.randn(32, 128)
    
    for step in range(10):
        # Forward pass
        output = model(data)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Create fake momentum states
        optimizer_state = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                optimizer_state[name] = {
                    'momentum_buffer': torch.randn_like(param.grad) * 0.1
                }
        
        # Monitor (will only run every 5 steps due to monitor_freq)
        metrics = monitor.monitor_training_step(
            loss=loss,
            optimizer_state=optimizer_state,
            learning_rate=0.01 * (0.9 ** step)
        )
        
        if step % 5 == 0:
            print(f"  Step {step}: Collected {len(metrics)} metrics")
    
    print("  ✅ Integration test passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Spectral Monitoring System")
    print("=" * 60)
    
    try:
        test_spectral_analyzer()
        print()
        test_muon_monitor()
        print()
        test_integration()
        print()
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise