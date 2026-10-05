import json
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
with open('training_logs/training_20260527_014141.json', 'r') as f:
    data = json.load(f)

epochs_data = data['epochs']
epochs = [e['epoch'] for e in epochs_data]
train_loss = [e['train_loss'] for e in epochs_data]
val_loss = [e['val_loss'] for e in epochs_data]
train_ppl = [e['train_ppl'] for e in epochs_data]
val_ppl = [e['val_ppl'] for e in epochs_data]
overfit_gap = [e['overfit_gap'] for e in epochs_data]

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'train': '#2E86AB',  # синий
    'val': '#A23B72',    # фиолетовый
    'gap': '#F18F01'     # оранжевый
}

# Создание фигуры с 3 графиками
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Результаты обучения модели генерации молекул', fontsize=16, fontweight='bold', y=1.02)

# График 1: Loss
ax1 = axes[0]
ax1.plot(epochs, train_loss, 'o-', color=colors['train'], linewidth=2.5, markersize=8, label='Train Loss', alpha=0.8)
ax1.plot(epochs, val_loss, 's-', color=colors['val'], linewidth=2.5, markersize=8, label='Validation Loss', alpha=0.8)
ax1.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
ax1.set_title('Динамика функции потерь', fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(epochs)

# Добавление аннотации финального значения
final_val = val_loss[-1]
ax1.annotate(f'Финальный\nVal Loss: {final_val:.4f}',
             xy=(epochs[-1], final_val),
             xytext=(epochs[-3], final_val + 0.05),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

# График 2: Perplexity
ax2 = axes[1]
ax2.plot(epochs, train_ppl, 'o-', color=colors['train'], linewidth=2.5, markersize=8, label='Train PPL', alpha=0.8)
ax2.plot(epochs, val_ppl, 's-', color=colors['val'], linewidth=2.5, markersize=8, label='Validation PPL', alpha=0.8)
ax2.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
ax2.set_title('Динамика Perplexity', fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(epochs)

# Добавление аннотации
final_ppl = val_ppl[-1]
ax2.annotate(f'Финальный\nVal PPL: {final_ppl:.2f}',
             xy=(epochs[-1], final_ppl),
             xytext=(epochs[-3], final_ppl + 0.15),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

# График 3: Overfit Gap
ax3 = axes[2]
ax3.plot(epochs, overfit_gap, 'D-', color=colors['gap'], linewidth=2.5, markersize=8, label='Overfit Gap', alpha=0.8)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Идеальная сходимость')
ax3.fill_between(epochs, 0, overfit_gap, where=(np.array(overfit_gap) < 0), alpha=0.2, color='green', label='Underfitting')
ax3.fill_between(epochs, 0, overfit_gap, where=(np.array(overfit_gap) >= 0), alpha=0.2, color='red', label='Overfitting')
ax3.set_xlabel('Эпоха', fontsize=12, fontweight='bold')
ax3.set_ylabel('Train Loss - Val Loss', fontsize=12, fontweight='bold')
ax3.set_title('Анализ переобучения', fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10, frameon=True, shadow=True)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(epochs)

# Добавление аннотации
final_gap = overfit_gap[-1]
ax3.annotate(f'Финальный gap:\n{final_gap:.4f}\n(почти идеально!)',
             xy=(epochs[-1], final_gap),
             xytext=(epochs[-4], final_gap + 0.03),
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('training_results_beautiful.png', dpi=300, bbox_inches='tight')
print("✓ Сохранён график: training_results_beautiful.png")

# Создание отдельного большого графика Loss для презентации
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, train_loss, 'o-', color=colors['train'], linewidth=3, markersize=10, label='Train Loss', alpha=0.8)
ax.plot(epochs, val_loss, 's-', color=colors['val'], linewidth=3, markersize=10, label='Validation Loss', alpha=0.8)
ax.set_xlabel('Эпоха', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss (Cross-Entropy)', fontsize=14, fontweight='bold')
ax.set_title('Обучение модели: функция потерь по эпохам', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=13, frameon=True, shadow=True, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)

# Добавление текстового блока с информацией
textstr = f'Конфигурация модели:\n• Параметры: 25.3M\n• Датасет: 317K молекул (20%)\n• Batch size: 64\n• Learning rate: 3e-4\n\nРезультат:\n• Val Loss: {val_loss[-1]:.4f}\n• Val PPL: {val_ppl[-1]:.2f}\n• Overfit gap: {overfit_gap[-1]:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('loss_plot_presentation.png', dpi=300, bbox_inches='tight')
print("✓ Сохранён график: loss_plot_presentation.png")

# Создание графика Perplexity для презентации
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, train_ppl, 'o-', color=colors['train'], linewidth=3, markersize=10, label='Train Perplexity', alpha=0.8)
ax.plot(epochs, val_ppl, 's-', color=colors['val'], linewidth=3, markersize=10, label='Validation Perplexity', alpha=0.8)
ax.set_xlabel('Эпоха', fontsize=14, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
ax.set_title('Обучение модели: Perplexity по эпохам', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=13, frameon=True, shadow=True, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)

# Добавление пояснения
textstr = f'Perplexity показывает "уверенность" модели:\n\n• Начальный PPL: {val_ppl[0]:.2f}\n  (модель сомневается между ~2 токенами)\n\n• Финальный PPL: {val_ppl[-1]:.2f}\n  (модель сомневается между ~1.6 токенами)\n\n• Улучшение: {((val_ppl[0] - val_ppl[-1])/val_ppl[0]*100):.1f}%\n\nНизкий PPL = высокая уверенность модели'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('perplexity_plot_presentation.png', dpi=300, bbox_inches='tight')
print("✓ Сохранён график: perplexity_plot_presentation.png")

print("\n" + "="*60)
print("Все графики успешно созданы!")
print("="*60)
print("\nФайлы:")
print("  1. training_results_beautiful.png - 3 графика в одном")
print("  2. loss_plot_presentation.png - большой график Loss")
print("  3. perplexity_plot_presentation.png - большой график Perplexity")
print("\nИспользуйте их в курсовой работе!")
