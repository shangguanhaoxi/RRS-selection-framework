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
import pandas as pd
import time
from datetime import datetime

# ========== Configuration Parameters ==========
epochs_to_save = [20]
total_epochs = epochs_to_save[-1]

lr = 0.0001
batch_size = 64
weight_decay = 0.05
model_load_path = None

train_data_dir = '/root/autodl-tmp/train'
val_data_dir = '/root/autodl-tmp/val'
height_threshold = 4.5

sampling_N_combinations = [
    [6,11,14,19]
]

base_model_save_dir = '/root/autodl-tmp/program/model_mamba/sample_1/'
base_tensorboard_dir = '/root/autodl-tmp/tensorboard_logs'

patience = 10
min_delta = 0.0001

excel_log_path = '/root/autodl-tmp/computational_efficiency_stats.xlsx'


# ========== Mamba Block Implementation ==========
class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)

        # 1D convolution for input projection
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # Linear projections
        self.x_proj = nn.Linear(self.d_inner, self.d_inner * 2, bias=False)
        self.z_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # State parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, L, D]
        B, L, D = x.shape
        residual = x

        x = self.norm(x)

        # Expand dimension
        x = nn.Linear(D, self.d_inner)(x)  # [B, L, d_inner]

        # 1D convolution
        x_conv = x.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]  # [B, d_inner, L]
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]

        # Split for x and z
        x_z = self.x_proj(x_conv)  # [B, L, 2*d_inner]
        x_part, z_part = x_z.chunk(2, dim=-1)  # Each [B, L, d_inner]

        z = torch.sigmoid(self.z_proj(x_conv))  # [B, L, d_inner]

        # Simplified SSM (for computational efficiency)
        # In a full implementation, this would use selective scan
        y = torch.tanh(x_part) * z  # Simplified operation

        # Output projection
        y = self.out_proj(y)  # [B, L, D]

        return y + residual


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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss


# ========== Mamba Model ==========
class MambaModel(nn.Module):
    def __init__(self, num_classes=2, pretrained_weight_path=None, target_image_size=16):
        super().__init__()
        self.target_image_size = target_image_size
        self.mobilenet_input_size = 224

        # 输入预处理（与MobileNet相同）
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

        # 如果提供了预训练权重，则加载
        if pretrained_weight_path and os.path.exists(pretrained_weight_path):
            print(f"Loading pretrained weights from {pretrained_weight_path}")
            self.load_state_dict(torch.load(pretrained_weight_path))

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

        sample_tensor = torch.FloatTensor(sampled_values).unsqueeze(0)
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
            sample = torch.zeros((1, self.sampling_N, self.sampling_N))
            label = 0

        if self.transform:
            sample = self.transform(sample)

        return sample, torch.tensor(label, dtype=torch.long)


transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def collate_fn(batch):
    size_groups = {}
    for sample, label in batch:
        size_key = sample.shape[-1]
        if size_key not in size_groups:
            size_groups[size_key] = []
        size_groups[size_key].append((sample, label))

    batched_samples, batched_labels = [], []
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
    print(f"\n{'=' * 60}")
    print(f"Training model with sampling_N_list: {sampling_N_list}")
    print(f"{'=' * 60}")

    start_time = time.time()
    print("Loading datasets...")
    data_loading_start = time.time()

    all_train_datasets, all_val_datasets = [], []
    for sampling_N in sampling_N_list:
        all_train_datasets.append(HeightThresholdDataset(train_data_dir, sampling_N, height_threshold, transform))
        all_val_datasets.append(HeightThresholdDataset(val_data_dir, sampling_N, height_threshold, transform))

    train_dataset = ConcatDataset(all_train_datasets)
    val_dataset = ConcatDataset(all_val_datasets)
    # Note: num_workers > 0 can be beneficial
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                            pin_memory=True)
    data_loading_time = time.time() - data_loading_start

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaModel(num_classes=2, pretrained_weight_path=model_load_path).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    model_save_path = os.path.join(model_save_dir, 'model_64_10.pth')

    print(f"Starting training for {total_epochs} epochs")
    print(f"Data loading and preprocessing time: {data_loading_time:.2f}s")
    print(f"Using device: {device}")

    total_training_time = 0
    actual_epochs_trained = 0
    epoch_times, pre_network_times = [], []
    forward_times, backward_times, update_times = [], [], []

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_correct, total_samples = 0.0, 0, 0

        data_loading_time_epoch = 0.0
        forward_time_epoch, backward_time_epoch, update_time_epoch = 0.0, 0.0, 0.0

        # 统计实际处理的batch数量
        actual_batches_processed = 0

        for batch_idx, (batch_samples, batch_labels) in enumerate(train_loader):
            # 数据加载部分计时 (CPU to GPU transfer)
            data_start = time.time()
            # Move all data to GPU at once
            batch_samples_gpu = [samples.to(device, non_blocking=True) for samples in batch_samples]
            batch_labels_gpu = [labels.to(device, non_blocking=True) for labels in batch_labels]
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保数据传输完成
            data_loading_time_epoch += time.time() - data_start

            # 网络计算部分计时 - 细分为Forward、Backward、Update
            # Process each batch group separately
            for samples, labels in zip(batch_samples_gpu, batch_labels_gpu):
                actual_batches_processed += 1

                # 重置梯度
                optimizer.zero_grad()

                # Forward pass
                forward_start = time.time()
                outputs = model(samples)
                loss = criterion(outputs, labels)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 确保前向传播完成
                forward_time_epoch += time.time() - forward_start

                # Backward pass
                backward_start = time.time()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 确保反向传播完成
                backward_time_epoch += time.time() - backward_start

                # Update parameters
                update_start = time.time()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 确保参数更新完成
                update_time_epoch += time.time() - update_start

                train_loss += loss.item() * samples.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total_samples += samples.size(0)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # Add epsilon to prevent division by zero if total_samples is 0
        epoch_train_loss = train_loss / (total_samples + 1e-6)
        epoch_train_acc = 100 * train_correct / (total_samples + 1e-6)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_forward_time = 0.0

        with torch.no_grad():
            for batch_samples, batch_labels in val_loader:
                for samples, labels in zip(batch_samples, batch_labels):
                    # Move data to device right before using it
                    samples, labels = samples.to(device), labels.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 确保数据传输完成

                    val_forward_start = time.time()
                    outputs = model(samples)
                    loss = criterion(outputs, labels)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 确保前向传播完成
                    val_forward_time += time.time() - val_forward_start

                    val_loss += loss.item() * samples.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += samples.size(0)

        # Add epsilon to prevent division by zero if val_total is 0
        epoch_val_loss = val_loss / (val_total + 1e-6)
        epoch_val_acc = 100 * val_correct / (val_total + 1e-6)
        epoch_time = time.time() - epoch_start_time

        epoch_times.append(epoch_time)
        pre_network_times.append(data_loading_time_epoch)
        forward_times.append(forward_time_epoch)
        backward_times.append(backward_time_epoch)
        update_times.append(update_time_epoch)
        actual_epochs_trained = epoch + 1
        total_training_time += epoch_time

        network_compute_time_epoch = forward_time_epoch + backward_time_epoch + update_time_epoch
        accounted_time = data_loading_time_epoch + network_compute_time_epoch
        unaccounted_time = epoch_time - accounted_time

        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{total_epochs}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print(f"Actual batches processed: {actual_batches_processed}")
        print(f"Data Loading Time: {data_loading_time_epoch:.2f}s ({data_loading_time_epoch / epoch_time * 100:.1f}%)")
        print(
            f"Network Compute Time: {network_compute_time_epoch:.2f}s ({network_compute_time_epoch / epoch_time * 100:.1f}%)")
        print(
            f"  - Forward Time: {forward_time_epoch:.2f}s ({forward_time_epoch / network_compute_time_epoch * 100:.1f}%)")
        print(
            f"  - Backward Time: {backward_time_epoch:.2f}s ({backward_time_epoch / network_compute_time_epoch * 100:.1f}%)")
        print(
            f"  - Update Time: {update_time_epoch:.2f}s ({update_time_epoch / network_compute_time_epoch * 100:.1f}%)")
        print(f"Validation Forward Time: {val_forward_time:.2f}s")
        print(f"Unaccounted Time (overhead): {unaccounted_time:.2f}s ({unaccounted_time / epoch_time * 100:.1f}%)")
        print(f"Accounted/Total Ratio: {accounted_time / epoch_time * 100:.1f}%")
        print(f"Training Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%")
        print(f"Validation Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
        writer.add_scalar('Time/Epoch', epoch_time, epoch)
        writer.add_scalar('Time/Data_Loading', data_loading_time_epoch, epoch)
        writer.add_scalar('Time/Network_Compute', network_compute_time_epoch, epoch)
        writer.add_scalar('Time/Forward', forward_time_epoch, epoch)
        writer.add_scalar('Time/Backward', backward_time_epoch, epoch)
        writer.add_scalar('Time/Update', update_time_epoch, epoch)
        writer.add_scalar('Time/Unaccounted', unaccounted_time, epoch)

        early_stopping(epoch_val_loss, model, model_save_path)
        if early_stopping.early_stop:
            print(f"🚨 Early stopping triggered at epoch {epoch + 1}")
            break

    total_training_time = time.time() - start_time

    # Handle case where training stops early (epoch_times is empty)
    if not epoch_times:
        epoch_times = [0]
        pre_network_times = [0]
        forward_times = [0]
        backward_times = [0]
        update_times = [0]

    avg_epoch_time = np.mean(epoch_times)
    avg_pre_network_time = np.mean(pre_network_times)
    avg_forward_time = np.mean(forward_times)
    avg_backward_time = np.mean(backward_times)
    avg_update_time = np.mean(update_times)
    avg_network_training_time = avg_forward_time + avg_backward_time + avg_update_time

    pre_network_ratio = avg_pre_network_time / avg_epoch_time * 100
    network_ratio = avg_network_training_time / avg_epoch_time * 100
    forward_ratio = avg_forward_time / avg_network_training_time * 100
    backward_ratio = avg_backward_time / avg_network_training_time * 100
    update_ratio = avg_update_time / avg_network_training_time * 100

    stats = {
        'Model_ID': f"model_{'_'.join(map(str, sampling_N_list))}",
        'Sampling_Resolutions': str(sampling_N_list),
        'Training_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Num_Resolutions': len(sampling_N_list),
        'Training_Samples': len(train_dataset),
        'Validation_Samples': len(val_dataset),
        'Data_Loading_Time_Seconds': data_loading_time,
        'Average_Pre_Network_Time_Seconds': avg_pre_network_time,
        'Average_Forward_Time_Seconds': avg_forward_time,
        'Average_Backward_Time_Seconds': avg_backward_time,
        'Average_Update_Time_Seconds': avg_update_time,
        'Average_Network_Training_Time_Seconds': avg_network_training_time,
        'Pre_Network_Time_Ratio_Percentage': pre_network_ratio,
        'Network_Training_Time_Ratio_Percentage': network_ratio,
        'Forward_Time_Ratio_Percentage': forward_ratio,
        'Backward_Time_Ratio_Percentage': backward_ratio,
        'Update_Time_Ratio_Percentage': update_ratio,
        'Total_Training_Time_Seconds': total_training_time,
        'Total_Training_Time_Minutes': total_training_time / 60,
        'Average_Epoch_Time_Seconds': avg_epoch_time,
        'Total_Epochs_Planned': total_epochs,
        'Actual_Epochs_Trained': actual_epochs_trained,
        'Final_Train_Accuracy': epoch_train_acc,
        'Final_Val_Accuracy': epoch_val_acc,
        'Batch_Size': batch_size,
        'Best_Val_Loss': early_stopping.val_loss_min,
    }

    writer.close()
    return stats


if __name__ == "__main__":
    all_stats = []
    for i, sampling_N_list in enumerate(sampling_N_combinations):
        sampling_str = "_".join(map(str, sampling_N_list))
        model_save_dir = os.path.join(base_model_save_dir, f"model_{sampling_str}")
        tensorboard_dir = os.path.join(base_tensorboard_dir, f"model_{sampling_str}")
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        stats = train_model_with_sampling_N(sampling_N_list, model_save_dir, tensorboard_dir)
        all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    df.to_excel(excel_log_path, index=False, engine='openpyxl')
    print(f"\n🎉 All models training completed!")

    print(f"📊 Statistics saved to {excel_log_path}")
