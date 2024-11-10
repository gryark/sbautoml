# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:21:32 2024

@author: ataberk.urfali
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm
import copy
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Veri dönüşümleri
def get_data_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    }

# Model seçimi
def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'ResNet-18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet-50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet-101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B1':
        model = models.efficientnet_b1(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B2':
        model = models.efficientnet_b2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B3':
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B4':
        model = models.efficientnet_b4(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B5':
        model = models.efficientnet_b5(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B6':
        model = models.efficientnet_b6(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'EfficientNet-B7':
        model = models.efficientnet_b7(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'DenseNet169':
        model = models.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'DenseNet201':
        model = models.densenet201(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'MobileNetV1':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v3_small(pretrained=pretrained)  # MobileNet V2 as MobileNetV3 Small
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'MobileNetV3':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'VGG19':
        model = models.vgg19(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)
    return model

# Optimizer seçimi
def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adadelta':
        return optim.Adadelta(model.parameters(), lr=learning_rate)

# Loss fonksiyonu seçimi
def get_loss_function(loss_name):
    if loss_name == 'Cross Entropy Loss':
        return nn.CrossEntropyLoss()
    elif loss_name == 'Binary Cross Entropy Loss':
        return nn.BCELoss()
    elif loss_name == 'Binary Cross Entropy with Logits Loss':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'Mean Squared Error Loss':
        return nn.MSELoss()
    elif loss_name == 'Mean Absolute Error Loss (L1 Loss)':
        return nn.L1Loss()
    elif loss_name == 'Kullback-Leibler Divergence Loss':
        return nn.KLDivLoss()
    elif loss_name == 'Smooth L1 Loss (Huber Loss)':
        return nn.SmoothL1Loss()
    

# Eğitim fonksiyonu
def train_model(data_dir, model_name, optimizer_name, loss_name, epochs, batch_size, learning_rate, progress=gr.Progress()):
    data_transforms = get_data_transforms()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'val']}
    
    model = get_model(model_name, num_classes=len(image_datasets['train'].classes))
    criterion = get_loss_function(loss_name)
    optimizer = get_optimizer(optimizer_name, model, learning_rate)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics_output = ""
    
    # Kayıpları saklamak için listeler
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_metrics = f"Epoch {epoch + 1}/{epochs}\n"
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} - Epoch {epoch+1}/{epochs}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            epoch_metrics += f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}\n"

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            progress(1 / (epochs * len(dataloaders[phase])))  # Gradio progress güncellemesi

        metrics_output += epoch_metrics + "\n"
        
    model.load_state_dict(best_model_wts)
    
    # Kayıp grafiği oluşturma
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("loss_plot.png")  # Kaydedilen loss grafiği

    return model, metrics_output, "loss_plot.png"

# Test fonksiyonu
def test_model(model, data_dir, batch_size, num_examples=8):  # num_examples parametresini 8 yaptık
    data_transforms = get_data_transforms()
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    dataloader = DataLoader(image_datasets, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    example_images_no_tumor = []
    example_images_tumor = []

    model.eval()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # İki sınıf için dört örnek toplama
        for i in range(inputs.size(0)):
            if len(example_images_no_tumor) < num_examples // 2 and labels[i].item() == 0:
                example_images_no_tumor.append((inputs[i].cpu(), labels[i].item(), preds[i].item()))
            elif len(example_images_tumor) < num_examples // 2 and labels[i].item() == 1:
                example_images_tumor.append((inputs[i].cpu(), labels[i].item(), preds[i].item()))
            if len(example_images_no_tumor) >= num_examples // 2 and len(example_images_tumor) >= num_examples // 2:
                break

    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    # Konfüzyon matrisi
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=image_datasets.classes, yticklabels=image_datasets.classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Test metrikleri
    test_metrics = (
        f"F1 Score: {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Accuracy: {accuracy:.4f}"
    )

    # Örnek görüntüleri kaydetme
    example_outputs = []
    selected_examples = example_images_no_tumor + example_images_tumor
    for i, (img, true_label, pred_label) in enumerate(selected_examples):
        plt.figure()
        plt.imshow(img.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
        plt.axis('off')
        plt.title(f"Gerçek: {image_datasets.classes[true_label]}\nTahmin: {image_datasets.classes[pred_label]}")
        example_path = f"example_{i}.png"
        plt.savefig(example_path)
        plt.close()
        example_outputs.append(example_path)

    return test_metrics, "confusion_matrix.png", example_outputs

# Tabular classification için fonksiyon
def tabular_classification_interface(data_path, model_type, target_column, batch_size, learning_rate):
    data = pd.read_csv(data_path)
    data = data.fillna(0)  
    data = data.replace([np.inf, -np.inf], 0)
    
    # Mevcut sütunları al ve kullanıcıya seçmesi için listele
    columns = list(data.columns)
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    test_metrics = (
        f"F1 Score: {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Accuracy: {accuracy:.4f}"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Tabular Classification")
    plt.savefig("tabular_confusion_matrix.png")

    return test_metrics, "tabular_confusion_matrix.png"

def gradio_interface(data_dir, model_name, optimizer_name, loss_name, epochs, batch_size, learning_rate):
    batch_size = int(batch_size)
    model, metrics_output, loss_plot = train_model(data_dir, model_name, optimizer_name, loss_name, epochs, batch_size, learning_rate)
    test_results, cm_image, example_outputs = test_model(model, data_dir, batch_size)

    return metrics_output, test_results, cm_image, loss_plot, example_outputs

with gr.Blocks() as demo:
    gr.Markdown("# Sağlık Bakanlığı AutoML Uygulaması")
    gr.Markdown("Veri yolu, model seçimi, optimizer ve loss fonksiyonu seçimi ile model eğitimi ve test işlemleri gerçekleştirilir.")

    with gr.Tab("Image Classification"):
        with gr.Row():
            with gr.Column():
                data_dir = gr.Textbox(label="Dataset Yolunu Girin")
                model_name = gr.Dropdown(["ResNet-18", "ResNet-50", "ResNet-101", "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201", "MobileNetV1", "MobileNetV2", "MobileNetV3", "VGG16", "VGG19"], label="Model Seçimi")
                optimizer_name = gr.Dropdown(["SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta"], label="Optimizer Seçimi")
                loss_name = gr.Dropdown(["Cross Entropy Loss", "Binary Cross Entropy Loss", "Binary Cross Entropy with Logits Loss", "Mean Squared Error Loss", "Mean Absolute Error Loss (L1 Loss)", "Kullback-Leibler Divergence Loss", "Smooth L1 Loss (Huber Loss)"], label="Loss Fonksiyonu Seçimi")
                epochs = gr.Slider(1, 100, step=1, label="Epoch Sayısı")
                batch_size = gr.Dropdown([1, 2, 4, 8, 16, 32, 64, 128, 256], label="Batch Size") # Simplified for example
                learning_rate = gr.Number(value=0.001, label="Learning Rate")
                train_button = gr.Button("Eğit")
            with gr.Column():
                metrics_output = gr.Textbox(label="Eğitim Metrikleri")
                test_results = gr.Textbox(label="Test Metrikleri")
                cm_image = gr.Image(label="Konfüzyon Matrisi")
                loss_plot = gr.Image(label="Loss Grafiği")
                example_outputs = gr.Gallery(label="Örnek Görüntüler")

            train_button.click(
                gradio_interface,
                inputs=[data_dir, model_name, optimizer_name, loss_name, epochs, batch_size, learning_rate],
                outputs=[metrics_output, test_results, cm_image, loss_plot, example_outputs]
            )

    with gr.Tab("Tabular Classification"):
        with gr.Row():
            with gr.Column():
                data_path = gr.Textbox(label="Tabular Veri Dosya Yolu")
                model_type = gr.Dropdown(["RandomForest", "DecisionTree"], label="Model Tipi")
                
                # Tek seçimli Dropdown ile sütun seçimi
                target_column = gr.Dropdown([], label="Seçilecek Sütun", interactive=True)
                
                batch_size_tabular = gr.Dropdown([1, 2, 4, 8, 16, 32, 64, 128, 256], label="Batch Size")
                learning_rate_tabular = gr.Number(value=0.001, label="Learning Rate")
                train_button_tabular = gr.Button("Eğit")
            
            with gr.Column():
                tabular_metrics_output = gr.Textbox(label="Tabular Eğitim Sonuçları")
                tabular_cm_image = gr.Image(label="Konfüzyon Matrisi - Tabular")
    
            # Sütunları Dropdown içine doldurmak için bir veri yolu girişi değişikliği olduğunda çağrılan olay
            def update_columns(data_path):
                data = pd.read_csv(data_path)
                columns = list(data.columns)
                return gr.Dropdown.update(choices=columns)
    
            data_path.change(update_columns, inputs=data_path, outputs=target_column)
    
            train_button_tabular.click(
                tabular_classification_interface,
                inputs=[data_path, model_type, target_column, batch_size_tabular, learning_rate_tabular],
                outputs=[tabular_metrics_output, tabular_cm_image]
            )

demo.launch()

