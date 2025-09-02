import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import wandb
import os

class SpectralAnalyzerWithMSign:
    """
    使用苏剑林博客中的高效算法进行谱分析
    参考: https://kexue.fm/archives/11221
    """
    
    @staticmethod
    @torch.compile
    def msign_newton_schulz(A: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        计算矩阵符号函数 msign(A)，基于苏剑林的优化Newton-Schulz迭代
        
        这个函数满足：如果 A = U @ S @ V^T，则 msign(A) = U @ sign(S) @ V^T
        其中 sign(S) 是对角元素取符号
        """
        # 苏剑林博客中的优化系数（5阶收敛）
        # 这些系数是通过Padé逼近优化得到的
        a0 = 1.0
        a1 = 1.5  # (3/2)
        a2 = 0.375  # (3/8)
        
        X = A.clone()
        
        # 归一化以确保收敛
        scale = X.norm(dim=(-2, -1), keepdim=True)
        X = X / (scale + 1e-8)
        
        for _ in range(steps):
            # Newton-Schulz迭代: X_{k+1} = X_k * (a0*I + a1*X_k^2 + a2*X_k^4)
            X2 = X @ X.mT  # X^2（对于非方阵使用X @ X^T）
            X4 = X2 @ X2   # X^4
            
            # 多项式组合
            poly = a0 * torch.eye(X2.size(-1), device=X.device) + a1 * X2 + a2 * X4
            X = X @ poly
        
        return X
    
    @staticmethod
    @torch.compile
    def compute_spectral_norm_via_msign(A: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        使用msign函数计算谱范数
        
        基于恒等式：||A||_2 = sqrt(λ_max(A^T @ A))
        可以通过幂迭代或msign方法高效计算
        """
        if A.ndim > 2:
            A = A.flatten(0, -2)
        
        # 构造增广矩阵进行谱范数计算
        # 利用 msign([0, A; A^T, 0]) 的特殊性质
        m, n = A.shape
        
        # 使用幂迭代的改进版本（结合msign思想）
        if m >= n:
            # 计算 A^T @ A 的最大特征值
            v = torch.randn(n, 1, device=A.device, dtype=A.dtype)
            v = v / v.norm()
            
            for _ in range(steps):
                # 幂迭代步
                w = A.T @ (A @ v)
                
                # 使用msign思想的归一化技巧
                w_norm = w.norm()
                v = w / (w_norm + 1e-8)
            
            # 最终的谱范数估计
            Av = A @ v
            return Av.norm()
        else:
            # 对于 m < n 的情况
            u = torch.randn(m, 1, device=A.device, dtype=A.dtype)
            u = u / u.norm()
            
            for _ in range(steps):
                w = A @ (A.T @ u)
                w_norm = w.norm()
                u = w / (w_norm + 1e-8)
            
            ATu = A.T @ u
            return ATu.norm()
    
    @staticmethod
    @torch.compile
    def polar_decomposition_via_msign(A: torch.Tensor, steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用msign计算极分解 A = U @ P
        其中 U 是正交矩阵，P 是半正定矩阵
        
        基于苏剑林博客：U = msign(A) 当 A 满秩时
        """
        # 增广矩阵技巧
        m, n = A.shape
        if m >= n:
            # 构造 [0, A; A^T, 0]
            Z = torch.zeros(m, m, device=A.device)
            top = torch.cat([Z, A], dim=1)
            bottom = torch.cat([A.T, torch.zeros(n, n, device=A.device)], dim=1)
            M = torch.cat([top, bottom], dim=0)
            
            # 计算 msign(M)
            M_sign = SpectralAnalyzerWithMSign.msign_newton_schulz(M, steps)
            
            # 提取正交部分
            U = M_sign[:m, m:]
            
            # 计算半正定部分 P = U^T @ A
            P = U.T @ A
        else:
            # 转置处理
            U_T, P_T = SpectralAnalyzerWithMSign.polar_decomposition_via_msign(A.T, steps)
            U = U_T.T
            P = P_T.T
        
        return U, P
    
    @staticmethod  
    @torch.compile
    def matrix_square_root_inverse(A: torch.Tensor, steps: int = 5, epsilon: float = 1e-6) -> torch.Tensor:
        """
        计算 A^{-1/2} 使用Newton-Schulz迭代
        基于苏剑林博客的优化版本
        """
        # 确保A是对称半正定的
        A_sym = 0.5 * (A + A.T)
        
        # 归一化
        scale = A_sym.norm()
        A_norm = A_sym / (scale + epsilon)
        
        # 初始化 X_0 = I
        X = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        
        # Newton-Schulz迭代计算 A^{-1/2}
        # X_{k+1} = X_k * (3I - A * X_k^2) / 2
        for _ in range(steps):
            AX2 = A_norm @ (X @ X)
            X = 0.5 * X @ (3.0 * torch.eye(A.size(0), device=A.device) - AX2)
        
        # 反归一化
        return X / (scale.sqrt() + epsilon)


class EnhancedMuonMonitor:
    """
    增强的Muon监控器，利用苏剑林博客的高效算法
    """
    
    def __init__(self, model: nn.Module, monitor_freq: int = 10, use_wandb: bool = True, project_name: str = "speedrun-spectral"):
        self.model = model
        self.monitor_freq = monitor_freq
        self.analyzer = SpectralAnalyzerWithMSign()
        self.history = {}
        self.step = 0
        self.use_wandb = use_wandb
        
        # Initialize wandb if enabled and on master process
        if self.use_wandb and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            wandb.init(
                project=project_name,
                config={
                    "monitor_freq": monitor_freq,
                    "model_params": sum(p.numel() for p in model.parameters()),
                    "model_layers": len([m for m in model.modules() if isinstance(m, nn.Module)])
                }
            )
        
    @torch.no_grad()
    def analyze_muon_orthogonalization(self, grad: torch.Tensor, momentum: torch.Tensor, 
                                      lr: float, momentum_coef: float) -> Dict:
        """
        分析Muon优化器的正交化过程，使用msign相关技术
        """
        results = {}
        
        # 1. 计算梯度的极分解
        U_grad, P_grad = self.analyzer.polar_decomposition_via_msign(grad, steps=5)
        
        # 2. 计算动量的极分解  
        U_mom, P_mom = self.analyzer.polar_decomposition_via_msign(momentum, steps=5)
        
        # 3. 分析正交部分的相似性
        if U_grad.shape == U_mom.shape:
            # 正交部分的Frobenius内积
            orthogonal_similarity = (U_grad * U_mom).sum().item()
            results['orthogonal_alignment'] = orthogonal_similarity
        
        # 4. 谱范数分析（使用改进的算法）
        grad_spectral = self.analyzer.compute_spectral_norm_via_msign(grad)
        mom_spectral = self.analyzer.compute_spectral_norm_via_msign(momentum)
        
        results['grad_spectral_norm'] = grad_spectral.item()
        results['momentum_spectral_norm'] = mom_spectral.item()
        
        # 5. 条件数估计（使用msign方法）
        # 对于条件数 κ(A) = ||A|| * ||A^{-1}||
        # 可以通过极分解的P矩阵估计
        P_grad_cond = self._estimate_condition_via_polar(P_grad)
        results['grad_condition_number'] = P_grad_cond
        
        # 6. 有效学习率分析
        # Muon的实际更新是正交化后的梯度
        effective_update_norm = self.analyzer.compute_spectral_norm_via_msign(U_grad)
        results['effective_lr_scale'] = (effective_update_norm / (grad_spectral + 1e-8)).item()
        
        return results
    
    def _estimate_condition_via_polar(self, P: torch.Tensor, epsilon: float = 1e-6) -> float:
        """
        通过极分解的P矩阵估计条件数
        """
        # P是半正定的，其条件数等于最大特征值/最小特征值
        try:
            eigenvalues = torch.linalg.eigvalsh(P)
            lambda_max = eigenvalues.max().item()
            lambda_min = eigenvalues[eigenvalues > epsilon].min().item() if (eigenvalues > epsilon).any() else epsilon
            return lambda_max / lambda_min
        except:
            return float('inf')
    
    @torch.no_grad()
    def compare_newton_schulz_variants(self, matrix: torch.Tensor) -> Dict:
        """
        比较不同的Newton-Schulz变体（原始vs苏剑林优化版）
        """
        results = {}
        
        # 1. 原始的Newton-Schulz（从train_gpt.py）
        original = self._original_newton_schulz(matrix.clone())
        
        # 2. 苏剑林的msign版本
        msign_version = self.analyzer.msign_newton_schulz(matrix.clone())
        
        # 3. 比较正交性
        original_error = torch.norm(original @ original.T - torch.eye(original.size(0), device=matrix.device))
        msign_error = torch.norm(msign_version @ msign_version.T - torch.eye(msign_version.size(0), device=matrix.device))
        
        results['original_orthogonality_error'] = original_error.item()
        results['msign_orthogonality_error'] = msign_error.item()
        results['improvement_ratio'] = (original_error / (msign_error + 1e-8)).item()
        
        return results
    
    def _original_newton_schulz(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """原始的Newton-Schulz实现（作为对比）"""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G
        if G.size(0) > G.size(1):
            X = X.T
        X = X / (X.norm() + 1e-7)
        
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
            
        if G.size(0) > G.size(1):
            X = X.T
        return X
    
    def monitor_training_step(self, loss: torch.Tensor, optimizer_state: Dict, learning_rate: float = None) -> Dict:
        """
        监控一个训练步骤，使用苏剑林的高效算法，并记录到wandb
        """
        if self.step % self.monitor_freq != 0:
            self.step += 1
            return {}
        
        step_metrics = {'step': self.step, 'loss': loss.item()}
        wandb_metrics = {'step': self.step, 'train/loss': loss.item()}
        
        if learning_rate is not None:
            wandb_metrics['train/lr'] = learning_rate
        
        # 分析关键层的参数和梯度
        layer_analysis = {}
        for name, param in self.model.named_parameters():
            if param.ndim >= 2 and param.grad is not None:
                # 获取动量缓冲区（如果存在）
                momentum = None
                if name in optimizer_state and 'momentum_buffer' in optimizer_state[name]:
                    momentum = optimizer_state[name]['momentum_buffer']
                
                # 使用增强的分析
                if momentum is not None and ('attn' in name or 'mlp' in name):
                    analysis = self.analyze_muon_orthogonalization(
                        param.grad, momentum, lr=learning_rate or 0.05, momentum_coef=0.95
                    )
                    
                    layer_type = 'attention' if 'attn' in name else 'mlp'
                    layer_analysis[f"{layer_type}/{name}"] = analysis
                    
                    # 记录关键指标到wandb
                    for key, value in analysis.items():
                        metric_name = f"{name}/{key}"
                        wandb_key = f"spectral/{layer_type}/{key}"
                        
                        if metric_name not in self.history:
                            self.history[metric_name] = []
                        self.history[metric_name].append(value)
                        step_metrics[metric_name] = value
                        wandb_metrics[wandb_key] = value
                
                # 每100步比较一次算法变体（仅对注意力层）
                if self.step % 100 == 0 and 'attn' in name:
                    comparison = self.compare_newton_schulz_variants(param.grad)
                    
                    # 记录算法比较结果到wandb
                    wandb_metrics.update({
                        f'algorithm_comparison/{name}/original_error': comparison['original_orthogonality_error'],
                        f'algorithm_comparison/{name}/msign_error': comparison['msign_orthogonality_error'],
                        f'algorithm_comparison/{name}/improvement_ratio': comparison['improvement_ratio']
                    })
                    
                    if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                        print(f"\n{name} - Newton-Schulz comparison:")
                        print(f"  Original error: {comparison['original_orthogonality_error']:.6f}")
                        print(f"  MSign error: {comparison['msign_orthogonality_error']:.6f}")
                        print(f"  Improvement: {comparison['improvement_ratio']:.2f}x")
        
        # 计算全局统计信息
        if layer_analysis:
            global_stats = self._compute_global_stats(layer_analysis)
            wandb_metrics.update(global_stats)
        
        # 记录到wandb（仅在master process）
        if self.use_wandb and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            wandb.log(wandb_metrics, step=self.step)
        
        self.step += 1
        return step_metrics
    
    def _compute_global_stats(self, layer_analysis: Dict) -> Dict:
        """计算全局统计信息"""
        global_stats = {}
        
        # 收集所有层的指标
        all_spectral_norms = []
        all_condition_numbers = []
        all_orthogonal_alignments = []
        all_effective_lr_scales = []
        
        for layer_name, analysis in layer_analysis.items():
            if 'grad_spectral_norm' in analysis:
                all_spectral_norms.append(analysis['grad_spectral_norm'])
            if 'grad_condition_number' in analysis:
                all_condition_numbers.append(analysis['grad_condition_number'])
            if 'orthogonal_alignment' in analysis:
                all_orthogonal_alignments.append(analysis['orthogonal_alignment'])
            if 'effective_lr_scale' in analysis:
                all_effective_lr_scales.append(analysis['effective_lr_scale'])
        
        # 计算全局统计
        if all_spectral_norms:
            global_stats.update({
                'global/spectral_norm_mean': np.mean(all_spectral_norms),
                'global/spectral_norm_std': np.std(all_spectral_norms),
                'global/spectral_norm_max': np.max(all_spectral_norms),
                'global/spectral_norm_min': np.min(all_spectral_norms)
            })
        
        if all_condition_numbers:
            valid_conds = [c for c in all_condition_numbers if not np.isinf(c) and not np.isnan(c)]
            if valid_conds:
                global_stats.update({
                    'global/condition_number_mean': np.mean(valid_conds),
                    'global/condition_number_std': np.std(valid_conds),
                    'global/condition_number_max': np.max(valid_conds)
                })
        
        if all_orthogonal_alignments:
            global_stats.update({
                'global/orthogonal_alignment_mean': np.mean(all_orthogonal_alignments),
                'global/orthogonal_alignment_std': np.std(all_orthogonal_alignments)
            })
        
        if all_effective_lr_scales:
            global_stats.update({
                'global/effective_lr_scale_mean': np.mean(all_effective_lr_scales),
                'global/effective_lr_scale_std': np.std(all_effective_lr_scales)
            })
        
        return global_stats
    
    def finish_monitoring(self):
        """完成监控，关闭wandb"""
        if self.use_wandb and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            wandb.finish()


# 实际使用示例
def demo_enhanced_monitoring():
    """
    演示如何使用增强的监控系统
    """
    # 创建一个简单的测试矩阵
    torch.manual_seed(42)
    A = torch.randn(100, 50, dtype=torch.float32)
    
    analyzer = SpectralAnalyzerWithMSign()
    
    print("=== 测试苏剑林算法 ===\n")
    
    # 1. 测试msign函数
    A_sign = analyzer.msign_newton_schulz(A, steps=5)
    print(f"MSign正交性检查: ||A_sign @ A_sign^T - I|| = {torch.norm(A_sign @ A_sign.T - torch.eye(100)):.6f}")
    
    # 2. 测试极分解
    U, P = analyzer.polar_decomposition_via_msign(A, steps=5)
    reconstruction_error = torch.norm(A - U @ P)
    print(f"极分解重构误差: ||A - U@P|| = {reconstruction_error:.6f}")
    
    # 3. 测试谱范数计算
    spectral_norm = analyzer.compute_spectral_norm_via_msign(A)
    spectral_norm_svd = torch.linalg.svdvals(A)[0]  # 对比标准SVD
    print(f"谱范数 (msign方法): {spectral_norm:.6f}")
    print(f"谱范数 (SVD): {spectral_norm_svd:.6f}")
    print(f"相对误差: {abs(spectral_norm - spectral_norm_svd) / spectral_norm_svd * 100:.2f}%")
    
    # 4. 测试矩阵平方根逆
    ATA = A.T @ A
    ATA_sqrt_inv = analyzer.matrix_square_root_inverse(ATA, steps=5)
    identity_check = ATA_sqrt_inv @ ATA @ ATA_sqrt_inv
    print(f"\n平方根逆验证: ||A^(-1/2) @ A @ A^(-1/2) - I|| = {torch.norm(identity_check - torch.eye(50)):.6f}")
    
    return analyzer


if __name__ == "__main__":
    # 运行演示
    analyzer = demo_enhanced_monitoring()