import numpy as np
import matplotlib.pyplot as plt

# ========================
# 🔧 Parameters (unchanged)
# ========================
sigma = 4
alpha = 4.5
h = 4.75
L_fixed = 50.0

R_fixed = sigma * np.sqrt(-2 * np.log(alpha / h))
N4_theoretical = 1 + np.sqrt(2) * L_fixed / (2 * R_fixed)

# -------------------------
# Define beta3 function
# -------------------------
def beta3(N):
    if N <= 1:
        return np.nan
    x = L_fixed / (2 * R_fixed * (N - 1))
    x = np.clip(x, -1.0 + 1e-8, 1.0 - 1e-8)
    sqrt_inside = 4 * R_fixed ** 2 - (L_fixed ** 2) / ((N - 1) ** 2)
    sqrt_inside = max(sqrt_inside, 0)
    sqrt_term = np.sqrt(sqrt_inside)
    term_in_bracket = (
        np.pi * R_fixed ** 2
        - 4 * R_fixed ** 2 * np.arccos(x)
        + (L_fixed / (N - 1)) * sqrt_term
    )
    return ((N - 1) ** 2 * term_in_bracket) / L_fixed ** 2

# ========================
# 📊 Compute H(N)
# ========================
N_start, N_end = 2.0, 28.0
N_values = np.linspace(N_start, N_end, 5000)
H_values = np.full_like(N_values, np.nan)

for i, N in enumerate(N_values):
    if N <= N4_theoretical:
        beta_val = beta3(N)
    else:
        beta_val = beta3(N4_theoretical)
    if not (0 < beta_val < 1):
        continue
    H_values[i] = -beta_val * np.log2(beta_val) - (1 - beta_val) * np.log2(1 - beta_val)

# ========================
# 🔍 Find critical points: N1, N2, N3, N4
# ========================
valid = ~np.isnan(H_values)
N_valid = N_values[valid]
H_valid = H_values[valid]

# Compute derivatives
dH = np.gradient(H_valid, N_valid)
d2H = np.gradient(dH, N_valid)

# Find maximum (N2)
sign_change = np.where((dH[:-1] > 0) & (dH[1:] <= 0))[0]
N2 = None
if len(sign_change) > 0:
    i0 = sign_change[0]
    n1, n2 = N_valid[i0], N_valid[i0 + 1]
    d1, d2 = dH[i0], dH[i0 + 1]
    N2 = n1 - d1 * (n2 - n1) / (d2 - d1) if d2 != d1 else n1

# Find inflection points (N1, N3)
d2_sign = np.sign(d2H)
sign_change2 = np.where(d2_sign[:-1] * d2_sign[1:] < 0)[0]
N_inflection_list = []
for idx in sign_change2:
    n1, n2 = N_valid[idx], N_valid[idx + 1]
    dd1, dd2 = d2H[idx], d2H[idx + 1]
    N_inf = n1 - dd1 * (n2 - n1) / (dd2 - dd1) if dd2 != dd1 else n1
    N_inflection_list.append(N_inf)

N1 = N_inflection_list[0] if len(N_inflection_list) >= 1 else None
N3 = N_inflection_list[1] if len(N_inflection_list) >= 2 else None
N4 = N4_theoretical

# Print results
print(f"\n🔍 Critical N values:")
print(f"   N4_theoretical (cutoff) = {N4:.6f}")
if N1 is not None: print(f"   N1 (first inflection) = {N1:.6f}")
if N2 is not None: print(f"   N2 (maximum) = {N2:.6f}")
if N3 is not None: print(f"   N3 (second inflection) = {N3:.6f}")

# ========================
# 🖼️ Plot: Final figure
# ========================
fig, ax = plt.subplots(figsize=(10, 6))

# Plot main curve
ax.plot(N_values, H_values, color='blue', linewidth=2, label='Resolution Task Entropy')

# Deeper blue shades (from light to dark)
blue_shades = ['#b3d9ff', '#80bfff', '#4da6ff', '#1a8cff']

# Define segment boundaries
boundaries = [N_start, N1, N2, N3, N4, N_end]

# Handle possible None values gracefully
for i in range(1, len(boundaries)):
    if boundaries[i] is None:
        boundaries[i] = boundaries[i - 1]

# Draw vertical segments
for i in range(len(boundaries) - 1):
    left = boundaries[i]
    right = boundaries[i + 1]
    if left < right:
        color = blue_shades[min(i, len(blue_shades) - 1)]
        ax.axvspan(left, right, color=color, alpha=0.6, zorder=-1)

# Helper: get H(N) from precomputed array
def get_H_at_N(N_point):
    if N_point is None:
        return None
    idx = np.argmin(np.abs(N_values - N_point))
    return H_values[idx]

# 🔸 CONFIGURE LABEL STYLE HERE
label_fontsize = 14  # ← 可在此调整所有标注的字体大小

# Manual label offsets (dx, dy) for each point
label_offsets = {
    r'$N_1$': (-0.6, 0.03),
    r'$N_2$': (0.2, -0.05),
    r'$N_3$': (0.3, 0.02),
    r'$N_4$': (-0.8, 0.03),
}

# Annotate points ON the curve using H(N) with custom offsets and italic N
points = [
    (N1, get_H_at_N(N1), r'$N_1$'),
    (N2, get_H_at_N(N2), r'$N_2$'),
    (N3, get_H_at_N(N3), r'$N_3$'),
    (N4, get_H_at_N(N4), r'$N_4$')
]

for N_point, H_point, label in points:
    if N_point is not None and H_point is not None and not np.isnan(H_point):
        ax.plot(N_point, H_point, 'ko', markersize=8, zorder=5)
        dx, dy = label_offsets[label]
        ax.text(N_point + dx, H_point + dy, label, fontsize=label_fontsize, fontweight='bold', zorder=5)

# Finalize plot
ax.set_xlabel('Sampling Resolution', fontsize=14, fontweight='bold')
ax.set_ylabel('Resolution Task Entropy', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left')
ax.set_xlim(N_start, N_end)
ax.set_ylim(0, 1.05)
ax.set_xticks(np.arange(N_start, N_end + 1, 1))

# Save outputs
plt.savefig('sampling_confidence_index_segmented_blue.png', dpi=300, bbox_inches='tight')
plt.savefig('sampling_confidence_index_segmented_blue.svg', format='svg', bbox_inches='tight')
plt.savefig('sampling_confidence_index_segmented_blue.pdf', format='pdf', bbox_inches='tight')

plt.show()

print("✅ Saved to 'sampling_confidence_index_segmented_blue.png' and '.svg'")