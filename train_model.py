import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import os
import itertools

# Model's params
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_CLASSES = 4
TRAIN_DIR = './data/train'
VALID_DIR = './data/validation'
RESULTS_DIR = './results_gridsearch'

# Class definition
CLASS_DESCRIPTIONS = {
    'class_a': 'Фильтр воздушный',
    'class_b': 'Фильтр масляный',
    'class_c': 'Фильтр салона',
    'class_d': 'Фильтр топливный'}


# Class early stopping
class EarlyStoppingCallback:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_f1, model):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop


# Define plot history function
def plot_training_history(history, filename):
    plt.figure(figsize=(12, 6))
    epochs_ran = len(history['val_f1'])
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_ran + 1), history['train_loss'][:epochs_ran], label='Train Loss')
    plt.plot(range(1, epochs_ran + 1), history['val_loss'][:epochs_ran], label='Validation Loss')
    plt.title('Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_ran + 1), history['train_acc'][:epochs_ran], label='Train Acc')
    plt.plot(range(1, epochs_ran + 1), history['val_acc'][:epochs_ran], label='Validation Acc')
    plt.plot(range(1, epochs_ran + 1), history['val_f1'][:epochs_ran], label='Validation F1 (Weighted)')
    plt.title('Metrics by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Train model function
def train_model(hyperparams, dataloaders, dataset_sizes, class_names):
    model_ft = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.last_channel
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=hyperparams['dropout_rate']),
        nn.Linear(num_ftrs, NUM_CLASSES))
    model_ft = model_ft.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    if hyperparams['optimizer_name'] == 'SGD':
        optimizer_ft = optim.SGD(model_ft.classifier.parameters(), lr=hyperparams['learning_rate'],
                                 momentum=hyperparams['momentum'], weight_decay=hyperparams['weight_decay'])
    else:
        OptimizerClass = getattr(optim, hyperparams['optimizer_name'])
        optimizer_ft = OptimizerClass(model_ft.classifier.parameters(), lr=hyperparams['learning_rate'],
                                      weight_decay=hyperparams['weight_decay'])

    early_stopping = EarlyStoppingCallback(patience=hyperparams['early_stop_patience'])
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_f1 = 0.0
    best_val_loss = float('inf')
    epochs_run = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(NUM_EPOCHS):
        epochs_run += 1
        for phase in ['train', 'validation']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                history['val_f1'].append(epoch_f1)

                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model_ft.state_dict())

        # eraly stopping
        if early_stopping(history['val_f1'][-1], model_ft):
            epochs_run -= 1
            break

        if early_stopping.early_stop:
            break

    model_ft.load_state_dict(best_model_wts)
    return best_f1, history, model_ft, best_val_loss, epochs_run

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    param_grid = {
        'dropout_rate': [0.2, 0.3, 0.4],
        'weight_decay': [0.0001, 0.0005, 0.001],
        'early_stop_patience': [5],
        'learning_rate': [0.001, 0.005, 0.01],
        'optimizer_name': ['SGD'],
        'momentum': [0.9],
        'loss_function_name': ['CrossEntropyLoss']}

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALID_DIR):
        print(f"Ошибка: Не найдены папки с данными. Создайте структуру для {NUM_CLASSES} классов.")
        return

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'validation': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'validation': datasets.ImageFolder(VALID_DIR, data_transforms['validation'])}

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    print(f"Total combination for check: {len(combinations)}")

    best_overall_f1 = 0.0
    best_overall_params = {}
    results_list = []

    for i, combination in enumerate(combinations):
        hyperparams = dict(zip(keys, combination))
        if hyperparams['optimizer_name'] != 'SGD' and 'momentum' in hyperparams:
            pass

        start_time = time.time()
        val_f1, history, model, best_val_loss, epochs_run = train_model(hyperparams, dataloaders, dataset_sizes,
                                                                        class_names)
        end_time = time.time()
        duration = end_time - start_time
        run_results = hyperparams.copy()
        run_results['best_val_f1'] = val_f1
        run_results['best_val_loss'] = best_val_loss
        run_results['epochs_run'] = epochs_run
        run_results['duration_sec'] = duration
        results_list.append(run_results)

        run_id = f"run_{i + 1:03d}_f1_{val_f1:.4f}"
        plot_training_history(history, os.path.join(RESULTS_DIR, f"plot_{run_id}.png"))
        merged_dict = history | hyperparams
        pd.DataFrame(merged_dict).to_excel(os.path.join(RESULTS_DIR, f"history_{run_id}.xlsx"), index=False)
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"model_{run_id}.pth"))

        print(f"Завершено {i + 1}/{len(combinations)}. F1 мера: {val_f1:.4f}. Время: {duration:.1f} сек.")

        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_overall_params = hyperparams
            print(f"--> Новая лучшая модель! F1: {best_overall_f1:.4f}")

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='best_val_f1', ascending=False)
    results_df.to_excel(os.path.join(RESULTS_DIR, "grid_search_summary.xlsx"), index=False)

    print("\n" + "=" * 40)
    print(f"F1 estimation: {best_overall_f1:.4f}")
    print(f"Best params: {best_overall_params}")
    print("Report is saved:", RESULTS_DIR)
    print("=" * 40 + "\n")

if __name__ == '__main__':
    main()
