import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mamba_ssm import Mamba
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2

# === 配置参数 ===
N_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
batch_size = 64
epochs = 10  # 用于图表标题，实际测试不依赖此值
alpha = 4.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 路径配置 ===
# 包含所有模型文件夹 (如 model_12_18 ) 的父目录
models_parent_dir = '/root/autodl-tmp/program/model_mamba/progressive_training/'

#/root/autodl-tmp/program/model_vit/data_gaussian/test
#/root/autodl-tmp/program/model_vit/data_test_normal/test
image_dir = '/root/autodl-tmp/program/model_vit/data_gaussian/test'

# 基础输出目录 (所有结果的根目录)
#/root/autodl-tmp/program/model_mamba/sample_results/2
#/root/autodl-tmp/program/model_mamba/sample_results_guassian/2
base_output_dir = '/root/autodl-tmp/program/model_mamba/progressive_training/1'
excel_load = 'mamba_confusion.xlsx'

# 确保基础输出目录存在
os.makedirs(base_output_dir, exist_ok=True)


class MambaModel(nn.Module):
    def __init__(self, num_classes=2, pretrained_weight_path=None, target_image_size=16):
        super().__init__()
        self.target_image_size = target_image_size
        self.mobilenet_input_size = 224

        # 输入预处理
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.mobilenet_input_size, self.mobilenet_input_size))

        self.backbone = mobilenet_v2(pretrained=False)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.backbone.classifier = nn.Identity()
        self.backbone.avgpool = nn.Identity()  # 关键：禁用全局池化

        # 投影层
        self.projection = nn.Sequential(
            nn.Conv2d(1280, 32, kernel_size=1),  # 输入 [B, 1280, 7, 7]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((target_image_size, target_image_size))  # 输出 [B, 32, 16, 16]
        )

        # Mamba 配置
        self.mamba = Mamba(
            d_model=32,  # 输入维度
            d_state=16,  # 状态维度
            d_conv=4,  # 卷积核大小
            expand=2  # 扩展因子
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.adaptive_pool(x)  # [B, 1, H, W] -> [B, 1, 224, 224]
        x = self.backbone.features(x)  # [B, 1, 224, 224] -> [B, 1280, 7, 7]
        x = self.projection(x)  # [B, 1280, 7, 7] -> [B, 32, 16, 16]

        # 将空间维度展平为序列
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 16*16, 32]

        # Mamba处理
        x = self.mamba(x)  # [B, 16*16, 32]

        # 全局平均和最大池化
        x_mean = x.mean(dim=1)  # [B, 32]
        x_max = x.max(dim=1).values  # [B, 32]
        x = x_mean + x_max

        return self.classifier(x)


# === 数据处理 ===
class ImageSampler:
    def __init__(self, image_dir, N=11, alpha=None):
        self.image_dir = image_dir
        self.sampling_N = N
        self.alpha = alpha
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tiff')])

    def sample_single_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                img = np.array(img).astype(np.float32)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return None, None

        h, w = img.shape
        x_coords = np.linspace(0, w - 1, self.sampling_N, dtype=int)
        y_coords = np.linspace(0, h - 1, self.sampling_N, dtype=int)
        X, Y = np.meshgrid(x_coords, y_coords)
        sampled_values = img[Y, X]
        sample_tensor = torch.FloatTensor(sampled_values).unsqueeze(0)

        label = 0
        if self.alpha is not None:
            max_pixel_value = np.max(img)
            label = 1 if max_pixel_value >= self.alpha else 0

        return sample_tensor, label


transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class HeightThresholdDataset(Dataset):
    def __init__(self, image_dir, N, transform=None, alpha=None):
        self.image_dir = image_dir
        self.N = N
        self.transform = transform
        self.alpha = alpha
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tiff')])
        self.sampler = ImageSampler(image_dir, N, alpha)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)

        sample, label = self.sampler.sample_single_image(img_path)

        if sample is None or label is None:
            # 返回默认值或跳过
            sample = torch.zeros(1, self.N, self.N)
            label = 0

        if self.transform:
            sample = self.transform(sample)

        return sample, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    # 过滤掉无效样本
    batch = [(s, l) for s, l in batch if s is not None and l is not None]
    if len(batch) == 0:
        return [], []

    size_groups = {}
    for sample, label in batch:
        size_key = sample.shape[-1]
        if size_key not in size_groups:
            size_groups[size_key] = []
        size_groups[size_key].append((sample, label))

    batched_samples = []
    batched_labels = []
    for size, group in size_groups.items():
        samples = torch.stack([item[0] for item in group])
        labels = torch.stack([item[1] for item in group])

        num_samples = len(samples)
        if num_samples < batch_size:
            repeat_times = batch_size // num_samples + 1
            samples = samples.repeat(repeat_times, *[1] * (samples.dim() - 1))[:batch_size]
            labels = labels.repeat(repeat_times)[:batch_size]

        batched_samples.append(samples)
        batched_labels.append(labels)

    return batched_samples, batched_labels


# === 可视化函数 ===
def generate_confusion_matrix_data(true_labels, predictions, model_name, N, output_dir):
    """生成混淆矩阵数据并保存为JSON"""
    if len(true_labels) == 0:
        print(f"No data available for confusion matrix for {model_name}, N={N}")
        return None

    try:
        # 确保标签是Python原生类型
        true_labels = [int(label) for label in true_labels]
        predictions = [int(pred) for pred in predictions]

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predictions)

        # 计算归一化混淆矩阵（百分比）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 保存混淆矩阵数据为JSON
        cm_data = {
            'model_name': model_name,
            'N': N,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'true_labels': true_labels,
            'predictions': predictions,
            'accuracy': 100 * np.mean(np.array(predictions) == np.array(true_labels))
        }
        cm_json_path = os.path.join(output_dir, f'confusion_matrix_data_{model_name}_N{N}.json')
        with open(cm_json_path, 'w') as f:
            json.dump(cm_data, f, indent=2)

        print(f"Confusion matrix data saved: {cm_json_path}")
        return cm_data

    except Exception as e:
        print(f"Error generating confusion matrix data for {model_name}, N={N}: {e}")
        return None


def generate_average_confusion_matrix(all_cm_data, model_name, output_dir):
    """为每个模型生成一个平均的混淆矩阵图"""
    try:
        if not all_cm_data:
            print(f"No confusion matrix data available for {model_name}")
            return None, None

        # 计算所有N值的平均归一化混淆矩阵
        normalized_matrices = []
        raw_matrices = []

        for cm_data in all_cm_data:
            if cm_data is not None:
                normalized_matrices.append(np.array(cm_data['confusion_matrix_normalized']))
                raw_matrices.append(np.array(cm_data['confusion_matrix']))

        if not normalized_matrices:
            print(f"No valid confusion matrix data for {model_name}")
            return None, None

        # 计算平均混淆矩阵
        avg_normalized_cm = np.mean(normalized_matrices, axis=0)
        avg_raw_cm = np.mean(raw_matrices, axis=0)

        # 计算平均准确率
        avg_accuracy = np.mean([cm_data['accuracy'] for cm_data in all_cm_data if cm_data is not None])

        # 创建图形
        plt.figure(figsize=(8, 6))

        # 绘制平均混淆矩阵热力图
        sns.heatmap(avg_normalized_cm,
                    annot=True,
                    fmt='.3f',
                    cmap='Blues',
                    cbar=True,
                    vmin=0,
                    vmax=1,
                    annot_kws={'size': 14, 'weight': 'bold'})

        # 设置标签和标题
        plt.title(f'Average Confusion Matrix - {model_name}\n'
                  f'(Averaged over all N values, Average Accuracy: {avg_accuracy:.1f}%)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')

        # 设置刻度标签
        plt.xticks([0.5, 1.5], ['Negative', 'Positive'], rotation=0, fontsize=11)
        plt.yticks([0.5, 1.5], ['Negative', 'Positive'], rotation=0, fontsize=11)

        plt.tight_layout()

        # 保存平均混淆矩阵图
        avg_cm_path = os.path.join(output_dir, f'average_confusion_matrix_{model_name}.svg')
        plt.savefig(avg_cm_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        # 保存平均混淆矩阵数据
        avg_cm_data = {
            'model_name': model_name,
            'average_confusion_matrix_normalized': avg_normalized_cm.tolist(),
            'average_confusion_matrix_raw': avg_raw_cm.tolist(),
            'average_accuracy': float(avg_accuracy),
            'number_of_N_values': len(normalized_matrices),
            'N_values_used': [cm_data['N'] for cm_data in all_cm_data if cm_data is not None]
        }

        avg_cm_json_path = os.path.join(output_dir, f'average_confusion_matrix_data_{model_name}.json')
        with open(avg_cm_json_path, 'w') as f:
            json.dump(avg_cm_data, f, indent=2)

        print(f"Average confusion matrix saved: {avg_cm_path}")
        print(f"Average confusion matrix data saved: {avg_cm_json_path}")

        return avg_cm_path, avg_cm_data

    except Exception as e:
        print(f"Error generating average confusion matrix for {model_name}: {e}")
        return None, None


def generate_excel_report(all_avg_cm_data, output_dir):
    """生成Excel报告，包含所有模型的混淆矩阵数据"""
    try:
        if not all_avg_cm_data:
            print("No average confusion matrix data available for Excel report")
            return None

        # 准备Excel数据
        excel_data = []

        for model_name, avg_cm_data in all_avg_cm_data.items():
            if avg_cm_data is None:
                continue

            # 提取混淆矩阵的四个值
            cm_normalized = np.array(avg_cm_data['average_confusion_matrix_normalized'])

            # 混淆矩阵的四个单元格：
            # [0,0]: True Negative (TN)
            # [0,1]: False Positive (FP)
            # [1,0]: False Negative (FN)
            # [1,1]: True Positive (TP)

            row_data = {
                'Model_Name': model_name,
                'TN_Rate': cm_normalized[0, 0],  # True Negative Rate
                'FP_Rate': cm_normalized[0, 1],  # False Positive Rate
                'FN_Rate': cm_normalized[1, 0],  # False Negative Rate
                'TP_Rate': cm_normalized[1, 1],  # True Positive Rate
                'Average_Accuracy': avg_cm_data['average_accuracy'],
                'Number_of_N_Values': avg_cm_data['number_of_N_values']
            }
            excel_data.append(row_data)

        # 创建DataFrame
        df = pd.DataFrame(excel_data)

        # 设置列的顺序
        column_order = [
            'Model_Name',
            'TN_Rate',
            'FP_Rate',
            'FN_Rate',
            'TP_Rate',
            'Average_Accuracy',
            'Number_of_N_Values'
        ]
        df = df[column_order]

        # 保存Excel文件
        excel_path = os.path.join(output_dir, excel_load)

        # 使用ExcelWriter来设置格式
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Confusion Matrix Summary', index=False)

            # 获取工作表对象来设置格式
            worksheet = writer.sheets['Confusion Matrix Summary']

            # 设置列宽
            worksheet.column_dimensions['A'].width = 25  # Model_Name
            worksheet.column_dimensions['B'].width = 12  # TN_Rate
            worksheet.column_dimensions['C'].width = 12  # FP_Rate
            worksheet.column_dimensions['D'].width = 12  # FN_Rate
            worksheet.column_dimensions['E'].width = 12  # TP_Rate
            worksheet.column_dimensions['F'].width = 15  # Average_Accuracy
            worksheet.column_dimensions['G'].width = 15  # Number_of_N_Values

            # 设置标题行格式
            for cell in worksheet[1]:
                cell.font = pd.ExcelWriter._get_default_style()['font'].copy(bold=True)
                cell.alignment = pd.ExcelWriter._get_default_style()['alignment'].copy(horizontal='center')

        print(f"Excel report saved: {excel_path}")
        return excel_path

    except Exception as e:
        print(f"Error generating Excel report: {e}")
        return None


# === 核心测试逻辑 ===
def load_model(model_path, device):
    model = MambaModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    bump_probabilities = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_samples, batch_labels in test_loader:
            if len(batch_samples) == 0:  # 处理空批次
                continue
            for samples, labels in zip(batch_samples, batch_labels):
                samples = samples.to(device)
                labels = labels.to(device)
                outputs = model(samples)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                bump_probabilities.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    if len(all_labels) == 0 or len(all_preds) == 0:
        return {
            'accuracy': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'avg_bump_prob': 0.0,
            'all_bump_probs': [],
            'all_labels': [],
            'all_preds': []
        }

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    recall = 100 * recall_score(all_labels, all_preds, zero_division=0)  # 计算 recall
    f1 = 100 * f1_score(all_labels, all_preds, zero_division=0)  # 计算 F1 score
    avg_bump_prob = np.mean(bump_probabilities) if bump_probabilities else 0.0

    return {
        'accuracy': float(accuracy),
        'recall': float(recall),
        'f1': float(f1),
        'avg_bump_prob': float(avg_bump_prob),
        'all_bump_probs': [float(p) for p in bump_probabilities],
        'all_labels': all_labels,
        'all_preds': all_preds
    }


def find_model_paths(parent_dir):
    """在父目录下查找所有包含 .pth 文件的子目录，并返回模型路径列表"""
    model_paths = []
    if not os.path.isdir(parent_dir):
        print(f"Warning: Parent directory {parent_dir} does not exist or is not a directory.")
        return model_paths

    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            # 查找该目录下的 .pth 文件
            pth_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]
            if pth_files:
                # 假设每个文件夹只有一个模型文件，或取第一个找到的
                model_path = os.path.join(item_path, pth_files[0])
                model_paths.append(model_path)
                print(f"Found model: {model_path}")
            else:
                print(f"No .pth file found in directory: {item_path}")
    return model_paths


def test_different_N_values():
    # 自动查找模型路径
    model_paths = find_model_paths(models_parent_dir)

    if not model_paths:
        print("No model paths found. Exiting.")
        return

    # 收集所有模型的平均混淆矩阵数据
    all_models_avg_cm_data = {}

    for model_idx, model_path in enumerate(model_paths):
        print(f"\n==================== Testing Model {model_idx + 1}/{len(model_paths)} ====================")
        print(f"Model Path: {model_path}")

        # 为当前模型创建独立的结果目录
        # 提取模型文件所在的文件夹名称
        model_folder_name = os.path.basename(os.path.dirname(model_path))
        model_output_dir = os.path.join(base_output_dir, model_folder_name)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"Output directory for this model: {model_output_dir}")

        model = load_model(model_path, device)
        results = {}
        all_cm_data = []  # 收集所有N值的混淆矩阵数据

        # 对预设的 N_values 列表中的每一个 N 值进行测试
        for N in N_values:
            print(f"\nTesting with N={N}")

            # 直接使用原始图像目录和采样器，不保存中间文件
            test_dataset = HeightThresholdDataset(image_dir, N, transform, alpha)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            eval_results = evaluate_model(model, test_loader, device)
            results[N] = eval_results

            # 为每个N值生成混淆矩阵JSON文件并收集数据
            if len(eval_results['all_labels']) > 0:
                cm_data = generate_confusion_matrix_data(
                    eval_results['all_labels'],
                    eval_results['all_preds'],
                    model_folder_name, N, model_output_dir
                )
                if cm_data:
                    all_cm_data.append(cm_data)
            else:
                print(f"No data collected for N={N}, skipping confusion matrix")

        # 为当前模型生成平均混淆矩阵图
        avg_cm_path = None
        avg_cm_data = None
        if all_cm_data:
            avg_cm_path, avg_cm_data = generate_average_confusion_matrix(all_cm_data, model_folder_name, model_output_dir)
            if avg_cm_path:
                print(f"Average confusion matrix saved: {avg_cm_path}")

            # 保存平均混淆矩阵数据用于Excel报告
            all_models_avg_cm_data[model_folder_name] = avg_cm_data

        # 保存结果到JSON
        results_file = os.path.join(model_output_dir, 'all_results.json')
        with open(results_file, 'w') as f:
            serializable_results = {}
            for N_val, result in results.items():
                serializable_result = result.copy()
                # 移除不需要保存到大文件中的数据
                if 'all_bump_probs' in serializable_result:
                    del serializable_result['all_bump_probs']
                if 'all_labels' in serializable_result:
                    del serializable_result['all_labels']
                if 'all_preds' in serializable_result:
                    del serializable_result['all_preds']
                serializable_results[N_val] = serializable_result
            json.dump(serializable_results, f, indent=4)
        print(f"\nResults JSON saved to: {results_file}")

        print(f"\nSummary of Results for {model_folder_name}:")
        print("N\tAccuracy\tRecall\tF1\tAvg Bump Prob")
        for N in sorted(results.keys()):
            r = results[N]
            print(f"{N}\t{r['accuracy']:.2f}%\t{r['recall']:.2f}%\t{r['f1']:.2f}%\t{r['avg_bump_prob']:.4f}")

        # 计算并保存平均值
        avg_accuracy = np.mean([results[N]['accuracy'] for N in N_values])
        avg_recall = np.mean([results[N]['recall'] for N in N_values])
        avg_f1 = np.mean([results[N]['f1'] for N in N_values])

        print(f"\nAverage Accuracy across all N values for {model_folder_name}: {avg_accuracy:.2f}%")
        print(f"Average Recall across all N values for {model_folder_name}: {avg_recall:.2f}%")
        print(f"Average F1 across all N values for {model_folder_name}: {avg_f1:.2f}%")

        with open(results_file, 'r') as f:
            all_results = json.load(f)
        all_results['average_metrics'] = {
            'average_accuracy': float(avg_accuracy),
            'average_recall': float(avg_recall),
            'average_f1': float(avg_f1)
        }
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)

        # 提取数据用于绘图
        N_list = sorted(results.keys())
        accuracies = [results[N]['accuracy'] for N in N_list]
        recalls = [results[N]['recall'] for N in N_list]
        f1s = [results[N]['f1'] for N in N_list]

        # --- 绘制并保存图表 ---
        # Combined Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(N_list, accuracies, marker='o', linestyle='-', color='r', label='Accuracy')
        plt.plot(N_list, recalls, marker='s', linestyle='--', color='g', label='Recall')
        plt.plot(N_list, f1s, marker='^', linestyle='-.', color='b', label='F1 Score')
        plt.xlabel('Sampling Points per Side (N)')
        plt.ylabel('Percentage (%)')
        plt.title(
            f'Mamba: Accuracy, Recall, F1 vs Sampling Points\n(Model: {model_folder_name})')
        plt.legend()
        plt.grid(True)
        plt.xticks(N_list)
        plt.ylim(0, 100)
        plt.tight_layout()
        combined_svg_path = os.path.join(model_output_dir, f'Mamba_combined_metrics_trend.svg')
        plt.savefig(combined_svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Combined Metrics SVG saved to: {combined_svg_path}")
        plt.close()

        # Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(N_list, accuracies, marker='o', linestyle='-', color='r')
        plt.xlabel('Sampling Points per Side (N)')
        plt.ylabel('Accuracy (%)')
        plt.title(
            f'Mamba: Accuracy vs Sampling Points\n(Model: {model_folder_name})')
        plt.grid(True)
        plt.xticks(N_list)
        plt.ylim(0, 100)
        plt.tight_layout()
        accuracy_svg_path = os.path.join(model_output_dir, f'Mamba_accuracy_trend.svg')
        plt.savefig(accuracy_svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Accuracy SVG saved to: {accuracy_svg_path}")
        plt.close()

        # Recall
        plt.figure(figsize=(10, 6))
        plt.plot(N_list, recalls, marker='o', linestyle='-', color='g')
        plt.xlabel('Sampling Points per Side (N)')
        plt.ylabel('Recall (%)')
        plt.title(
            f'Mamba: Recall vs Sampling Points\n(Model: {model_folder_name})')
        plt.grid(True)
        plt.xticks(N_list)
        plt.ylim(0, 100)
        plt.tight_layout()
        recall_svg_path = os.path.join(model_output_dir, f'Mamba_recall_trend.svg')
        plt.savefig(recall_svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Recall SVG saved to: {recall_svg_path}")
        plt.close()

        # F1 Score
        plt.figure(figsize=(10, 6))
        plt.plot(N_list, f1s, marker='o', linestyle='-', color='b')
        plt.xlabel('Sampling Points per Side (N)')
        plt.ylabel('F1 Score (%)')
        plt.title(
            f'Mamba: F1 Score vs Sampling Points\n(Model: {model_folder_name})')
        plt.grid(True)
        plt.xticks(N_list)
        plt.ylim(0, 100)
        plt.tight_layout()
        f1_svg_path = os.path.join(model_output_dir, f'Mamba_f1_trend.svg')
        plt.savefig(f1_svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"F1 Score SVG saved to: {f1_svg_path}")
        plt.close()

    # 在所有模型测试完成后生成Excel报告
    if all_models_avg_cm_data:
        excel_path = generate_excel_report(all_models_avg_cm_data, base_output_dir)
        if excel_path:
            print(f"\nExcel summary report saved: {excel_path}")

    print(f"\nAll models have been tested. Results are saved under: {base_output_dir}")


if __name__ == "__main__":
    test_different_N_values()