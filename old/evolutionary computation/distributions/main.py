import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the range of μ values
mu_values = np.arange(-5, 5.1, 0.1)  # From -5 to 5 with step 0.1

# Function to calculate KL divergence from P = N(0,1) to Q = N(mu,1)
def kl_p_to_q(mu):
    """
    Calculate KL(P||Q) where:
    P = N(0,1)
    Q = N(mu,1)
    
    For normal distributions with same variance σ²=1:
    KL(P||Q) = (μ₁ - μ₂)²/(2σ²) = μ²/2 when μ₁=0, μ₂=mu, σ²=1
    """
    return mu**2 / 2

# Function to calculate KL divergence from Q = N(mu,1) to P = N(0,1)
def kl_q_to_p(mu):
    """
    Calculate KL(Q||P) where:
    Q = N(mu,1)
    P = N(0,1)
    
    For normal distributions with same variance σ²=1:
    KL(Q||P) = (μ₂ - μ₁)²/(2σ²) = μ²/2 when μ₂=mu, μ₁=0, σ²=1
    """
    return mu**2 / 2

# Calculate KL divergences for each μ value
kl_p_to_q_values = [kl_p_to_q(mu) for mu in mu_values]
kl_q_to_p_values = [kl_q_to_p(mu) for mu in mu_values]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot both KL divergences
plt.plot(mu_values, kl_p_to_q_values, 'b-', label='KL(P||Q): P=N(0,1), Q=N(μ,1)')
plt.plot(mu_values, kl_q_to_p_values, 'r--', label='KL(Q||P): Q=N(μ,1), P=N(0,1)')

# Add vertical lines to show symmetry around μ=0
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Add horizontal line at KL=0
plt.axhline(y=0, color='k', alpha=0.3)

# Customize the plot
plt.title('Kullback-Leibler Divergence Between Normal Distributions', fontsize=14)
plt.xlabel('μ (mean of Q)', fontsize=12)
plt.ylabel('KL Divergence', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Add text explaining the special case
plt.text(3, 10, 'Note: KL(P||Q) = KL(Q||P) in this case\nbecause the variances are equal (σ²=1)', 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Optional: Annotate some specific points
plt.annotate(f'KL = {kl_p_to_q(2):.1f}', xy=(2, kl_p_to_q(2)), xytext=(2.2, kl_p_to_q(2)+0.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Show grid and set axis limits
plt.xlim(-5.2, 5.2)
plt.ylim(-0.5, 13)

# Display the plot
plt.tight_layout()
plt.show()

# Print some sample values for verification
print("Sample KL divergence values:")
print("μ\tKL(P||Q)\tKL(Q||P)")
for mu in [-3, -2, -1, 0, 1, 2, 3]:
    print(f"{mu}\t{kl_p_to_q(mu):.4f}\t\t{kl_q_to_p(mu):.4f}")

# Optional: Verify with a more general formula
# The general formula for KL divergence between two normal distributions
# KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

def general_kl(mu1, sigma1_sq, mu2, sigma2_sq):
    """Calculate KL divergence between N(μ₁,σ₁²) and N(μ₂,σ₂²)"""
    return (np.log(np.sqrt(sigma2_sq)/np.sqrt(sigma1_sq)) + 
            (sigma1_sq + (mu1-mu2)**2)/(2*sigma2_sq) - 0.5)

# Testing with our specific case
test_mu = 2.5
kl_specific = kl_p_to_q(test_mu)
kl_general = general_kl(0, 1, test_mu, 1)

print(f"\nVerification for μ = {test_mu}:")
print(f"Using simplified formula: KL(P||Q) = {kl_specific:.4f}")
print(f"Using general formula: KL(P||Q) = {kl_general:.4f}")

# Mathematical explanation of the simplification
print("\nMathematical explanation:")
print("For normal distributions with equal variances (σ²=1):")
print("KL(P||Q) = KL(N(0,1)||N(μ,1)) = (0-μ)²/(2×1) = μ²/2")
print("KL(Q||P) = KL(N(μ,1)||N(0,1)) = (μ-0)²/(2×1) = μ²/2")