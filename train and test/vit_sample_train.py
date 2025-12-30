import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import mobilenet_v2
from transformers import ViTModel, ViTConfig
import pandas as pd
import time
from datetime import datetime

# ========== Configuration Parameters ==========
epochs_to_save = [50]
total_epochs = epochs_to_save[-1]  # 总共训练40个epoch

lr = 0.0001
batch_size = 64
weight_decay = 0.05
model_load_path = None
train_data_dir = '/root/autodl-tmp/train'
val_data_dir = '/root/autodl-tmp/val'
height_threshold = 4.5

# 定义多个sampling_N_list组合
sampling_N_combinations = [
    [6,11,14,19]
]

# 修改模型保存路径格式
base_model_save_dir = '/root/autodl-tmp/program/model_vit/model_vit/sample_1/'
base_tensorboard_dir = '/root/autodl-tmp/tensorboard_logs_vit'

# 早停机制参数
patience = 5  # 在验证集上性能没有提升的epoch数
min_delta = 0.0001  # 认为有提升的最小变化量

# Excel记录文件路径
excel_log_path = '/root/autodl-tmp/computational_efficiency_stats_vit.xlsx'


# ========== Early Stopping Class ==========
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, model_save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss


# ========== ViT Model ==========
class ViT(nn.Module):
    def __init__(self, num_classes=2, pretrained_weight_path=None, target_image_size=16):
        super().__init__()
        self.target_image_size = target_image_size
        self.mobilenet_input_size = 224

        # 输入预处理（与MobileNet相同）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.mobilenet_input_size, self.mobilenet_input_size))

        # 修改 MobileNetV2 并禁用全局池化
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

        # ViT 配置
        config = ViTConfig(
            image_size=target_image_size,
            patch_size=4,
            num_channels=32,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            classifier_dropout=0.1,
        )
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        x = self.adaptive_pool(x)  # [B, 1, H, W] -> [B, 1, 224, 224]
        x = self.backbone.features(x)  # [B, 1, 224, 224] -> [B, 1280, 7, 7]
        x = self.projection(x)  # [B, 1280, 7, 7] -> [B, 32, 16, 16]
        outputs = self.vit(x)  # ViT 处理
        sequence_output = outputs.last_hidden_state
        features = sequence_output.mean(dim=1) + sequence_output.max(dim=1).values
        return self.classifier(features)


# ========== Image Sampler ==========
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

        sample_tensor = torch.FloatTensor(sampled_values).unsqueeze(0)  # Shape: [1, N, N]

        label = 0
        if self.alpha is not None:
            max_pixel_value = np.max(img)
            label = 1 if max_pixel_value >= self.alpha else 0

        return sample_tensor, label


# ========== Dataset Classes ==========
class HeightThresholdDataset(Dataset):
    def __init__(self, image_dir, sampling_N, alpha, transform=None):
        self.image_dir = image_dir
        self.sampling_N = sampling_N
        self.alpha = alpha
        self.transform = transform
        self.sampler = ImageSampler(image_dir, sampling_N, alpha)
        self.image_files = self.sampler.image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        sample, label = self.sampler.sample_single_image(img_path)

        if sample is None or label is None:
            # Return placeholder if sampling failed
            sample = torch.zeros((1, self.sampling_N, self.sampling_N))
            label = 0

        if self.transform:
            sample = self.transform(sample)

        return sample, torch.tensor(label, dtype=torch.long)


# Data preprocessing
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
])


# Custom collate function for handling different input sizes
def collate_fn(batch):
    size_groups = {}
    for sample, label in batch:
        size_key = sample.shape[-1]  # Get sampling size N
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


def train_model_with_sampling_N(sampling_N_list, model_save_dir, tensorboard_dir):
    """训练单个模型，使用指定的sampling_N_list组合"""
    print(f"\n{'=' * 60}")
    print(f"Training ViT model with sampling_N_list: {sampling_N_list}")
    print(f"{'=' * 60}")

    start_time = time.time()  # 记录开始时间

    # ========== Data Loading ==========
    print("Loading datasets...")

    # 训练集：使用sampling_N_list中的所有采样值
    all_train_datasets = []
    for sampling_N in sampling_N_list:
        dataset = HeightThresholdDataset(train_data_dir, sampling_N, height_threshold, transform)
        all_train_datasets.append(dataset)

    # 验证集：使用与训练集相同的sampling_N_list组合
    all_val_datasets = []
    for sampling_N in sampling_N_list:
        dataset = HeightThresholdDataset(val_data_dir, sampling_N, height_threshold, transform)
        all_val_datasets.append(dataset)

    train_dataset = ConcatDataset(all_train_datasets)
    val_dataset = ConcatDataset(all_val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ========== Training Setup ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建ViT模型、优化器、损失函数和TensorBoard writer
    model = ViT(num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 添加余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    criterion = nn.CrossEntropyLoss()

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    model_save_path = os.path.join(model_save_dir, 'model_vit.pth')

    print(f"Starting ViT training for {total_epochs} epochs")
    print(f"Model will be saved with early stopping")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Number of sampling resolutions: {len(sampling_N_list)}")
    print(f"Sampling resolutions: {sampling_N_list}")

    # ========== Training Loop ==========
    total_training_time = 0
    actual_epochs_trained = 0
    epoch_times = []

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_correct, total_samples = 0.0, 0, 0
        train_batches = 0

        # 训练阶段
        for batch_idx, (batch_samples, batch_labels) in enumerate(train_loader):
            for samples, labels in zip(batch_samples, batch_labels):
                samples, labels = samples.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * samples.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total_samples += samples.size(0)
                train_batches += 1

        # 在每个epoch结束时更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 计算训练指标
        epoch_train_loss = train_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = 100 * train_correct / total_samples if total_samples > 0 else 0

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_batches = 0

        with torch.no_grad():
            for batch_samples, batch_labels in val_loader:
                for samples, labels in zip(batch_samples, batch_labels):
                    samples, labels = samples.to(device), labels.to(device)
                    outputs = model(samples)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * samples.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += samples.size(0)
                    val_batches += 1

        # 计算验证指标
        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        epoch_times.append(epoch_time)
        actual_epochs_trained = epoch + 1

        # 详细打印每个epoch的信息
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{total_epochs} - ViT Model")
        print(f"{'=' * 80}")

        # 时间信息
        print(f"⏰ Time Metrics:")
        print(f"   - Epoch Time: {epoch_time:.2f}s")
        print(f"   - Cumulative Time: {total_training_time:.2f}s ({total_training_time / 60:.2f} minutes)")

        # 训练信息
        print(f"📊 Training Metrics:")
        print(f"   - Loss: {epoch_train_loss:.4f}")
        print(f"   - Accuracy: {epoch_train_acc:.2f}%")
        print(f"   - Correct/Total: {train_correct}/{total_samples}")
        print(f"   - Batches Processed: {train_batches}")

        # 验证信息
        print(f"🔍 Validation Metrics:")
        print(f"   - Loss: {epoch_val_loss:.4f}")
        print(f"   - Accuracy: {epoch_val_acc:.2f}%")
        print(f"   - Correct/Total: {val_correct}/{val_total}")
        print(f"   - Batches Processed: {val_batches}")

        # 学习率和优化信息
        print(f"⚙️  Optimization Metrics:")
        print(f"   - Learning Rate: {current_lr:.2e}")
        print(f"   - Early Stopping Counter: {early_stopping.counter}/{patience}")

        # 进度信息
        progress = (epoch + 1) / total_epochs * 100
        print(f"📈 Progress:")
        print(f"   - Progress: {progress:.1f}% ({epoch + 1}/{total_epochs})")
        print(f"   - Estimated Remaining Time: {(total_epochs - epoch - 1) * np.mean(epoch_times):.2f}s")

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        writer.add_scalar('Time/Epoch', epoch_time, epoch)

        # 早停机制检查
        early_stopping(epoch_val_loss, model, model_save_path)

        if early_stopping.early_stop:
            print(f"\n🚨 Early stopping triggered at epoch {epoch + 1}!")
            print(f"   - Best validation loss: {early_stopping.val_loss_min:.6f}")
            print(f"   - Total epochs trained: {actual_epochs_trained}")
            break

    # 计算总训练时间
    total_training_time = time.time() - start_time

    # 收集计算效率统计信息
    stats = {
        # 模型识别信息
        'Model_ID': f"vit_model_{'_'.join(map(str, sampling_N_list))}",
        'Sampling_Resolutions': str(sampling_N_list),
        'Training_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

        # 数据效率指标
        'Num_Resolutions': len(sampling_N_list),
        'Training_Samples': len(train_dataset),
        'Validation_Samples': len(val_dataset),
        'Data_Reduction_Ratio': f"{len(sampling_N_list)}/{len(sampling_N_combinations[0])}",
        'Efficiency_Gain_Percentage': f"{(1 - len(sampling_N_list) / len(sampling_N_combinations[0])) * 100:.1f}%",

        # 时间效率指标
        'Total_Training_Time_Seconds': total_training_time,
        'Total_Training_Time_Minutes': total_training_time / 60,
        'Average_Epoch_Time_Seconds': np.mean(epoch_times),
        'Std_Epoch_Time_Seconds': np.std(epoch_times),
        'Min_Epoch_Time_Seconds': np.min(epoch_times),
        'Max_Epoch_Time_Seconds': np.max(epoch_times),

        # 收敛效率指标
        'Total_Epochs_Planned': total_epochs,
        'Actual_Epochs_Trained': actual_epochs_trained,
        'Early_Stopping_Triggered': early_stopping.early_stop,
        'Best_Val_Loss': early_stopping.val_loss_min,

        # 训练配置
        'Batch_Size': batch_size,
        'Final_Train_Accuracy': epoch_train_acc,
        'Final_Val_Accuracy': epoch_val_acc
    }

    print(f"\n{'=' * 80}")
    print(f"🏁 ViT Training Completed Summary for {sampling_N_list}")
    print(f"{'=' * 80}")
    print(f"✅ Total training time: {total_training_time:.2f}s ({total_training_time / 60:.2f} minutes)")
    print(f"✅ Average epoch time: {np.mean(epoch_times):.2f}s")
    print(f"✅ Data reduction: {stats['Efficiency_Gain_Percentage']}")
    print(f"✅ Actual epochs trained: {actual_epochs_trained}/{total_epochs}")
    print(f"✅ Final Train Accuracy: {epoch_train_acc:.2f}%")
    print(f"✅ Final Val Accuracy: {epoch_val_acc:.2f}%")
    print(f"✅ Best Validation Loss: {early_stopping.val_loss_min:.6f}")

    writer.close()

    return stats


# ========== Main Training Loop for Multiple Models ==========
if __name__ == "__main__":
    # 创建DataFrame来存储所有统计信息
    all_stats = []

    for i, sampling_N_list in enumerate(sampling_N_combinations):
        # 为每个模型组合创建独立的目录
        sampling_str = "_".join(map(str, sampling_N_list))
        model_save_dir = os.path.join(base_model_save_dir, f"model_{sampling_str}")
        tensorboard_dir = os.path.join(base_tensorboard_dir, f"model_{sampling_str}")

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        # 训练当前组合的模型（验证集使用相同的采样组合）
        stats = train_model_with_sampling_N(sampling_N_list, model_save_dir, tensorboard_dir)
        all_stats.append(stats)

    # 将统计信息保存到Excel
    df = pd.DataFrame(all_stats)

    # 重新排列列的顺序，让重要信息在前面
    column_order = [
        'Model_ID',
        'Sampling_Resolutions',
        'Num_Resolutions',
        'Data_Reduction_Ratio',
        'Efficiency_Gain_Percentage',
        'Training_Samples',
        'Validation_Samples',
        'Total_Training_Time_Seconds',
        'Total_Training_Time_Minutes',
        'Average_Epoch_Time_Seconds',
        'Std_Epoch_Time_Seconds',
        'Min_Epoch_Time_Seconds',
        'Max_Epoch_Time_Seconds',
        'Total_Epochs_Planned',
        'Actual_Epochs_Trained',
        'Early_Stopping_Triggered',
        'Best_Val_Loss',
        'Final_Train_Accuracy',
        'Final_Val_Accuracy',
        'Batch_Size',
        'Training_Date'
    ]

    df = df[column_order]

    # 保存到Excel
    df.to_excel(excel_log_path, index=False, engine='openpyxl')

    print(f"\n{'=' * 80}")
    print(f"🎉 All ViT models training completed!")
    print(f"📊 Total models trained: {len(sampling_N_combinations)}")
    print(f"💾 Computational efficiency statistics saved to: {excel_log_path}")
    print(f"{'=' * 80}")

    # 打印计算效率汇总统计
    print("\n📈 Computational Efficiency Summary:")
    print(
        f"   Average training time: {df['Total_Training_Time_Seconds'].mean():.2f}s ({df['Total_Training_Time_Minutes'].mean():.2f} minutes)")
    print(f"   Average epoch time: {df['Average_Epoch_Time_Seconds'].mean():.2f}s")
    print(f"   Average data reduction: {df['Efficiency_Gain_Percentage'].iloc[0]}")
    print(f"   Average actual epochs: {df['Actual_Epochs_Trained'].mean():.1f}")

    print(f"   Average final validation accuracy: {df['Final_Val_Accuracy'].mean():.2f}%")


