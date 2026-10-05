import json
import glob
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob('logs/training_logs/*.json') + glob.glob('training_logs/*.json')
log_file = files[-1] if files else 'training_logs/training_20260527_014141.json'

with open(log_file, 'r') as f:
    data = json.load(f)

epochs_data = data['epochs']
epochs = [e['epoch'] for e in epochs_data]
train_loss = [e['train_loss'] for e in epochs_data]
val_loss = [e['val_loss'] for e in epochs_data]
train_ppl = [e['train_ppl'] for e in epochs_data]
val_ppl = [e['val_ppl'] for e in epochs_data]
overfit_gap = [e['overfit_gap'] for e in epochs_data]

plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
colors = {'train': '#2E86AB', 'val': '#A23B72', 'gap': '#F18F01'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Результаты обучения модели генерации молекул', fontsize=16, fontweight='bold', y=1.02)

ax1 = axes[0]
ax1.plot(epochs, train_loss, 'o-', color=colors['train'], linewidth=2.5, markersize=8, label='Train Loss', alpha=0.8)
ax1.plot(epochs, val_loss, 's-', color=colors['val'], linewidth=2.5, markersize=8, label='Validation Loss', alpha=0.8)
ax1.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
ax1.set_title('Динамика функции потерь', fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(epochs)

final_val = val_loss[-1]
ax1.annotate(f'Финальный\nVal Loss: {final_val:.4f}',
             xy=(epochs[-1], final_val),
             xytext=(epochs[-3], final_val + 0.05),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

ax2 = axes[1]
ax2.plot(epochs, train_ppl, 'o-', color=colors['train'], linewidth=2.5, markersize=8, label='Train PPL', alpha=0.8)
ax2.plot(epochs, val_ppl, 's-', color=colors['val'], linewidth=2.5, markersize=8, label='Validation PPL', alpha=0.8)
ax2.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
ax2.set_title('Динамика Perplexity', fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(epochs)

final_ppl = val_ppl[-1]
ax2.annotate(f'Финальный\nVal PPL: {final_ppl:.2f}',
             xy=(epochs[-1], final_ppl),
             xytext=(epochs[-3], final_ppl + 0.15),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

ax3 = axes[2]
ax3.plot(epochs, overfit_gap, 'D-', color=colors['gap'], linewidth=2.5, markersize=8, label='Overfit Gap', alpha=0.8)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Идеальная сходимость')
ax3.fill_between(epochs, 0, overfit_gap, where=(np.array(overfit_gap) < 0), alpha=0.2, color='green', label='Underfitting')
ax3.fill_between(epochs, 0, overfit_gap, where=(np.array(overfit_gap) >= 0), alpha=0.2, color='red', label='Overfitting')
ax3.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
ax3.set_ylabel('Val Loss - Train Loss', fontsize=12, fontweight='bold')
ax3.set_title('Анализ переобучения', fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(epochs)

final_gap = overfit_gap[-1]
ax3.annotate(f'Финальный gap:\n{final_gap:.4f}\n(почти идеально!)',
             xy=(epochs[-1], final_gap),
             xytext=(epochs[-3], final_gap + 0.05),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('assets/training_results_beautiful.png' if glob.glob('assets') else 'training_results_beautiful.png', dpi=300)
