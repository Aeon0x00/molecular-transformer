import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
import numpy as np
from pathlib import Path
import re

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def parse_generation_log(log_file):
    """Парсинг лога генерации молекул"""
    with open(log_file, 'r') as f:
        content = f.read()

    # Извлечение SMILES и валидности
    pattern = r'\d+\.\s+([^\s]+)\s+-\s+(valid|invalid)'
    matches = re.findall(pattern, content)

    smiles_list = [m[0] for m in matches]
    validity = [m[1] == 'valid' for m in matches]

    # Извлечение статистики
    valid_match = re.search(r'Valid:\s+(\d+)/(\d+)\s+\((\d+\.?\d*)%\)', content)
    if valid_match:
        valid_count = int(valid_match.group(1))
        total_count = int(valid_match.group(2))
        valid_percent = float(valid_match.group(3))
    else:
        valid_count = sum(validity)
        total_count = len(validity)
        valid_percent = 100 * valid_count / total_count if total_count > 0 else 0

    return smiles_list, validity, valid_count, total_count, valid_percent

def calculate_molecular_properties(smiles_list):
    """Расчет молекулярных свойств"""
    properties = {
        'MW': [],           # Молекулярная масса
        'LogP': [],         # Липофильность
        'HBA': [],          # Акцепторы водородных связей
        'HBD': [],          # Доноры водородных связей
        'TPSA': [],         # Топологическая полярная площадь
        'RotBonds': [],     # Вращающиеся связи
        'Rings': [],        # Количество колец
        'Aromatic': []      # Ароматические кольца
    }

    valid_mols = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
            properties['MW'].append(Descriptors.MolWt(mol))
            properties['LogP'].append(Descriptors.MolLogP(mol))
            properties['HBA'].append(Descriptors.NumHAcceptors(mol))
            properties['HBD'].append(Descriptors.NumHDonors(mol))
            properties['TPSA'].append(Descriptors.TPSA(mol))
            properties['RotBonds'].append(Descriptors.NumRotatableBonds(mol))
            properties['Rings'].append(Descriptors.RingCount(mol))
            properties['Aromatic'].append(Descriptors.NumAromaticRings(mol))

    return properties, valid_mols

def plot_training_metrics(checkpoint_path):
    """Визуализация метрик обучения из чекпоинта"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Информация о модели
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')

    axes[0].text(0.5, 0.7, f'Эпоха: {epoch}', ha='center', fontsize=16, weight='bold')
    axes[0].text(0.5, 0.5, f'Validation Loss: {val_loss:.4f}' if isinstance(val_loss, float) else f'Validation Loss: {val_loss}',
                 ha='center', fontsize=14)
    axes[0].text(0.5, 0.3, f'Размер словаря: {len(checkpoint["tokenizer_vocab"])}', ha='center', fontsize=14)
    axes[0].axis('off')
    axes[0].set_title('Информация о модели', fontsize=16, weight='bold', pad=20)

    # Статистика словаря
    vocab = checkpoint['tokenizer_vocab']
    special_tokens = ['[PAD]', '[SOS]', '[EOS]', '[UNK]']
    special_count = sum(1 for token in special_tokens if token in vocab)
    regular_count = len(vocab) - special_count

    axes[1].pie([regular_count, special_count],
                labels=['Обычные токены', 'Специальные токены'],
                autopct='%1.1f%%',
                colors=['#3498db', '#e74c3c'],
                startangle=90,
                textprops={'fontsize': 12})
    axes[1].set_title('Состав словаря', fontsize=16, weight='bold', pad=20)

    plt.tight_layout()
    return fig

def plot_molecular_properties(properties):
    """Визуализация молекулярных свойств"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    prop_names = list(properties.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

    for idx, (prop_name, color) in enumerate(zip(prop_names, colors)):
        data = properties[prop_name]

        # Гистограмма с KDE
        axes[idx].hist(data, bins=30, alpha=0.7, color=color, edgecolor='black', density=True)

        # KDE
        from scipy import stats
        if len(data) > 1 and np.std(data) > 0:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 100)
            axes[idx].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Статистика
        mean_val = np.mean(data)
        median_val = np.median(data)
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Медиана: {median_val:.2f}')

        axes[idx].set_xlabel(prop_name, fontsize=12, weight='bold')
        axes[idx].set_ylabel('Плотность', fontsize=12)
        axes[idx].set_title(f'Распределение {prop_name}', fontsize=13, weight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_lipinski_rule_of_five(properties):
    """Визуализация соответствия правилу Липински"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Правило Липински
    lipinski_violations = []
    for i in range(len(properties['MW'])):
        violations = 0
        if properties['MW'][i] > 500:
            violations += 1
        if properties['LogP'][i] > 5:
            violations += 1
        if properties['HBA'][i] > 10:
            violations += 1
        if properties['HBD'][i] > 5:
            violations += 1
        lipinski_violations.append(violations)

    # Распределение нарушений
    violation_counts = [lipinski_violations.count(i) for i in range(5)]
    colors_viol = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']

    axes[0].bar(range(5), violation_counts, color=colors_viol, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Количество нарушений правила Липински', fontsize=12, weight='bold')
    axes[0].set_ylabel('Количество молекул', fontsize=12, weight='bold')
    axes[0].set_title('Соответствие правилу Липински (Rule of Five)', fontsize=14, weight='bold')
    axes[0].set_xticks(range(5))
    axes[0].grid(True, alpha=0.3, axis='y')

    # Добавление процентов
    total = sum(violation_counts)
    for i, count in enumerate(violation_counts):
        if count > 0:
            axes[0].text(i, count + max(violation_counts)*0.02, f'{count}\n({100*count/total:.1f}%)',
                        ha='center', va='bottom', fontsize=10, weight='bold')

    # Scatter plot: MW vs LogP
    scatter = axes[1].scatter(properties['MW'], properties['LogP'],
                             c=lipinski_violations, cmap='RdYlGn_r',
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Линии правила Липински
    axes[1].axvline(500, color='red', linestyle='--', linewidth=2, alpha=0.7, label='MW = 500')
    axes[1].axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='LogP = 5')

    axes[1].set_xlabel('Молекулярная масса (MW)', fontsize=12, weight='bold')
    axes[1].set_ylabel('Липофильность (LogP)', fontsize=12, weight='bold')
    axes[1].set_title('MW vs LogP (цвет = нарушения Липински)', fontsize=14, weight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Нарушения', fontsize=11, weight='bold')

    plt.tight_layout()
    return fig

def plot_validity_statistics(valid_count, total_count, valid_percent):
    """Визуализация статистики валидности"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    sizes = [valid_count, total_count - valid_count]
    colors = ['#2ecc71', '#e74c3c']
    labels = [f'Валидные\n{valid_count} ({valid_percent:.1f}%)',
              f'Невалидные\n{total_count - valid_count} ({100-valid_percent:.1f}%)']
    explode = (0.05, 0)

    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 13, 'weight': 'bold'})
    axes[0].set_title('Валидность сгенерированных молекул', fontsize=16, weight='bold', pad=20)

    # Bar chart
    categories = ['Валидные', 'Невалидные']
    counts = [valid_count, total_count - valid_count]

    bars = axes[1].bar(categories, counts, color=colors, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Количество молекул', fontsize=12, weight='bold')
    axes[1].set_title('Статистика генерации', fontsize=16, weight='bold', pad=20)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Добавление значений на столбцы
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({100*count/total_count:.1f}%)',
                    ha='center', va='bottom', fontsize=13, weight='bold')

    plt.tight_layout()
    return fig

def draw_molecule_grid(mols, n_mols=20, mols_per_row=5, img_size=(300, 300)):
    """Отрисовка сетки молекул"""
    mols_to_draw = mols[:n_mols]

    # Генерация 2D координат
    for mol in mols_to_draw:
        AllChem.Compute2DCoords(mol)

    img = Draw.MolsToGridImage(mols_to_draw,
                               molsPerRow=mols_per_row,
                               subImgSize=img_size,
                               legends=[f'Молекула {i+1}' for i in range(len(mols_to_draw))],
                               returnPNG=False)
    return img

def main():
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ И ГЕНЕРАЦИИ МОЛЕКУЛ")
    print("=" * 80)

    # Пути к файлам
    checkpoint_path = 'checkpoints/best_model.pt'
    log_file = 'generation_200.log'
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)

    # 1. Метрики обучения
    print("\n[1/6] Визуализация метрик обучения...")
    fig1 = plot_training_metrics(checkpoint_path)
    fig1.savefig(output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Сохранено: {output_dir / 'training_metrics.png'}")

    # 2. Парсинг результатов генерации
    print("\n[2/6] Парсинг результатов генерации...")
    smiles_list, validity, valid_count, total_count, valid_percent = parse_generation_log(log_file)
    print(f"✓ Обработано {total_count} молекул, валидных: {valid_count} ({valid_percent:.1f}%)")

    # 3. Статистика валидности
    print("\n[3/6] Визуализация статистики валидности...")
    fig2 = plot_validity_statistics(valid_count, total_count, valid_percent)
    fig2.savefig(output_dir / 'validity_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Сохранено: {output_dir / 'validity_statistics.png'}")

    # 4. Расчет молекулярных свойств
    print("\n[4/6] Расчет молекулярных свойств...")
    valid_smiles = [s for s, v in zip(smiles_list, validity) if v]
    properties, valid_mols = calculate_molecular_properties(valid_smiles)
    print(f"✓ Рассчитаны свойства для {len(valid_mols)} валидных молекул")

    # 5. Визуализация молекулярных свойств
    print("\n[5/6] Визуализация молекулярных свойств...")
    fig3 = plot_molecular_properties(properties)
    fig3.savefig(output_dir / 'molecular_properties.png', dpi=300, bbox_inches='tight')
    print(f"✓ Сохранено: {output_dir / 'molecular_properties.png'}")

    # 6. Правило Липински
    print("\n[6/6] Визуализация правила Липински...")
    fig4 = plot_lipinski_rule_of_five(properties)
    fig4.savefig(output_dir / 'lipinski_rule.png', dpi=300, bbox_inches='tight')
    print(f"✓ Сохранено: {output_dir / 'lipinski_rule.png'}")

    # 7. Отрисовка структур молекул
    print("\n[7/7] Отрисовка структур молекул...")

    # Топ-20 молекул
    img_top20 = draw_molecule_grid(valid_mols[:20], n_mols=20, mols_per_row=5)
    img_top20.save(output_dir / 'molecules_top20.png')
    print(f"✓ Сохранено: {output_dir / 'molecules_top20.png'}")

    # Случайная выборка
    if len(valid_mols) > 20:
        random_indices = np.random.choice(len(valid_mols), min(20, len(valid_mols)), replace=False)
        random_mols = [valid_mols[i] for i in random_indices]
        img_random = draw_molecule_grid(random_mols, n_mols=20, mols_per_row=5)
        img_random.save(output_dir / 'molecules_random20.png')
        print(f"✓ Сохранено: {output_dir / 'molecules_random20.png'}")

    # Сводная статистика
    print("\n" + "=" * 80)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Всего сгенерировано молекул: {total_count}")
    print(f"Валидных молекул: {valid_count} ({valid_percent:.1f}%)")
    print(f"\nМолекулярные свойства (среднее ± std):")
    print(f"  • Молекулярная масса: {np.mean(properties['MW']):.2f} ± {np.std(properties['MW']):.2f}")
    print(f"  • LogP: {np.mean(properties['LogP']):.2f} ± {np.std(properties['LogP']):.2f}")
    print(f"  • HBA: {np.mean(properties['HBA']):.2f} ± {np.std(properties['HBA']):.2f}")
    print(f"  • HBD: {np.mean(properties['HBD']):.2f} ± {np.std(properties['HBD']):.2f}")
    print(f"  • TPSA: {np.mean(properties['TPSA']):.2f} ± {np.std(properties['TPSA']):.2f}")
    print(f"  • Вращающиеся связи: {np.mean(properties['RotBonds']):.2f} ± {np.std(properties['RotBonds']):.2f}")
    print(f"  • Кольца: {np.mean(properties['Rings']):.2f} ± {np.std(properties['Rings']):.2f}")
    print(f"  • Ароматические кольца: {np.mean(properties['Aromatic']):.2f} ± {np.std(properties['Aromatic']):.2f}")

    print("\n" + "=" * 80)
    print(f"Все результаты сохранены в директории: {output_dir}")
    print("=" * 80)

    plt.close('all')

if __name__ == '__main__':
    main()
