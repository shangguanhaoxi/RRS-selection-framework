import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2

# === 配置参数 ===
N_values = [3,9,13,16]
batch_size = 64
epochs = 10  # 用于图表标题，实际测试不依赖此值
alpha = 4.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 包含所有模型文件夹 (如 model_12_18 ) 的父目录
models_parent_dir = '/root/autodl-tmp/program/model_mobilenet/sample_1'
# 测试图像目录
# '/root/autodl-tmp/program/model_vit/data_gaussian/test'
# '/root/autodl-tmp/program/model_vit/data_test_normal/test'
image_dir = '/root/autodl-tmp/program/model_vit/data_test_normal/test'
# 基础输出目录 (所有结果的根目录)
# /root/autodl-tmp/program/model_mobilenet/sample_results/2
# /root/autodl-tmp/program/model_mobilenet/guassian_sample_results/2

base_output_dir = '/root/autodl-tmp/program/model_mobilenet/sample_1/1'
excel_load = 'mobilenet_guassian_confusion.xlsx'

# 确保基础输出目录存在
os.makedirs(base_output_dir, exist_ok=True)


class MobileNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained_weight_path=None):
        super().__init__()
        self.mobilenet_input_size = 224

        # 输入预处理
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.mobilenet_input_size, self.mobilenet_input_size))

        # 使用MobileNetV2作为主干网络
        self.backbone = mobilenet_v2(pretrained=False)
        # 修改第一层卷积以适应单通道输入
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # 替换分类器
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def extract_features(self, x):
        """提取特征向量"""
        x = self.adaptive_pool(x)
        features = self.backbone.features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.backbone(x)
        return x


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


def generate_tsne_plot(features, labels, model_name, N, output_dir):
    """生成t-SNE可视化图"""
    try:
        if len(features) == 0 or len(labels) == 0:
            print(f"No features available for t-SNE plot for {model_name}, N={N}")
            return None

        # 确保特征和标签是numpy数组
        features_array = np.array(features)
        labels_array = np.array(labels)

        # 如果特征数量太少，跳过t-SNE
        if len(features_array) < 5:
            print(f"Not enough samples for t-SNE ({len(features_array)} samples) for {model_name}, N={N}")
            return None

        print(f"Generating t-SNE plot for {model_name}, N={N} with {len(features_array)} samples...")

        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_array) - 1))
        features_2d = tsne.fit_transform(features_array)

        # 创建t-SNE图
        plt.figure(figsize=(10, 8))

        # 根据标签着色
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=labels_array, cmap='viridis', alpha=0.7,
                              s=50, edgecolors='w', linewidth=0.5)

        plt.colorbar(scatter, label='Class Label')
        plt.title(f't-SNE Visualization - {model_name} (N={N})\n'
                  f'Total samples: {len(features_array)}', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)

        # 添加图例
        unique_labels = np.unique(labels_array)
        for label in unique_labels:
            mask = labels_array == label
            plt.scatter([], [], c=[plt.cm.viridis(label / len(unique_labels))],
                        label=f'Class {label}', alpha=0.7)
        plt.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存t-SNE图
        tsne_path = os.path.join(output_dir, f'tsne_plot_{model_name}_N{N}.svg')
        plt.savefig(tsne_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        # 保存t-SNE数据
        tsne_data = {
            'model_name': model_name,
            'N': N,
            'tsne_features': features_2d.tolist(),
            'labels': labels_array.tolist(),
            'num_samples': len(features_array)
        }

        tsne_json_path = os.path.join(output_dir, f'tsne_data_{model_name}_N{N}.json')
        with open(tsne_json_path, 'w') as f:
            json.dump(tsne_data, f, indent=2)

        print(f"t-SNE plot saved: {tsne_path}")
        print(f"t-SNE data saved: {tsne_json_path}")

        return tsne_path

    except Exception as e:
        print(f"Error generating t-SNE plot for {model_name}, N={N}: {e}")
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
    model = MobileNetModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    bump_probabilities = []
    all_labels = []
    all_preds = []
    all_features = []

    with torch.no_grad():
        for batch_samples, batch_labels in test_loader:
            if len(batch_samples) == 0:
                continue
            for samples, labels in zip(batch_samples, batch_labels):
                samples = samples.to(device)
                labels = labels.to(device)

                features = model.extract_features(samples)
                outputs = model(samples)

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                bump_probabilities.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_features.extend(features.cpu().numpy())

    if len(all_labels) == 0 or len(all_preds) == 0:
        return {
            'accuracy': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'avg_bump_prob': 0.0,
            'all_bump_probs': [],
            'all_features': [],
            'all_labels': [],
            'all_preds': []
        }

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    recall = 100 * recall_score(all_labels, all_preds, zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, zero_division=0)
    avg_bump_prob = np.mean(bump_probabilities) if bump_probabilities else 0.0

    return {
        'accuracy': float(accuracy),
        'recall': float(recall),
        'f1': float(f1),
        'avg_bump_prob': float(avg_bump_prob),
        'all_bump_probs': [float(p) for p in bump_probabilities],
        'all_features': all_features,
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
            pth_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]
            if pth_files:
                model_path = os.path.join(item_path, pth_files[0])
                model_paths.append(model_path)
                print(f"Found model: {model_path}")
            else:
                print(f"No .pth file found in directory: {item_path}")
    return model_paths


def test_different_N_values():
    model_paths = find_model_paths(models_parent_dir)

    if not model_paths:
        print("No model paths found. Exiting.")
        return

    # 收集所有模型的平均混淆矩阵数据
    all_models_avg_cm_data = {}

    for model_idx, model_path in enumerate(model_paths):
        print(f"\n==================== Testing Model {model_idx + 1}/{len(model_paths)} ====================")
        print(f"Model Path: {model_path}")

        model_folder_name = os.path.basename(os.path.dirname(model_path))
        model_output_dir = os.path.join(base_output_dir, model_folder_name)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"Output directory for this model: {model_output_dir}")

        model = load_model(model_path, device)
        results = {}
        all_cm_data = []  # 收集所有N值的混淆矩阵数据

        for N in N_values:
            print(f"\nTesting with N={N}")

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

                # 为每个N值生成t-SNE图
                if len(eval_results['all_features']) > 0:
                    tsne_path = generate_tsne_plot(
                        eval_results['all_features'],
                        eval_results['all_labels'],
                        model_folder_name, N, model_output_dir
                    )
                    if tsne_path:
                        print(f"t-SNE plot generated: {tsne_path}")
            else:
                print(f"No data collected for N={N}, skipping confusion matrix and t-SNE")

        # 为当前模型生成平均混淆矩阵图
        avg_cm_path = None
        avg_cm_data = None
        if all_cm_data:
            avg_cm_path, avg_cm_data = generate_average_confusion_matrix(all_cm_data, model_folder_name,
                                                                         model_output_dir)
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
                if 'all_features' in serializable_result:
                    del serializable_result['all_features']
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

        # 绘制性能趋势图
        N_list = sorted(results.keys())
        accuracies = [results[N]['accuracy'] for N in N_list]
        recalls = [results[N]['recall'] for N in N_list]
        f1s = [results[N]['f1'] for N in N_list]

        # Combined Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(N_list, accuracies, marker='o', linestyle='-', color='r', label='Accuracy')
        plt.plot(N_list, recalls, marker='s', linestyle='--', color='g', label='Recall')
        plt.plot(N_list, f1s, marker='^', linestyle='-.', color='b', label='F1 Score')
        plt.xlabel('Sampling Points per Side (N)')
        plt.ylabel('Percentage (%)')
        plt.title(f'mobilenet: Accuracy, Recall, F1 vs Sampling Points\n(Model: {model_folder_name})')
        plt.legend()
        plt.grid(True)
        plt.xticks(N_list)
        plt.ylim(0, 100)
        plt.tight_layout()
        combined_svg_path = os.path.join(model_output_dir, f'mobilenet_combined_metrics_trend.svg')
        plt.savefig(combined_svg_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"Combined Metrics SVG saved to: {combined_svg_path}")
        plt.close()

    # 在所有模型测试完成后生成Excel报告
    if all_models_avg_cm_data:
        excel_path = generate_excel_report(all_models_avg_cm_data, base_output_dir)
        if excel_path:
            print(f"\nExcel summary report saved: {excel_path}")

    print(f"\nAll models have been tested. Results are saved under: {base_output_dir}")


if __name__ == "__main__":
    test_different_N_values()