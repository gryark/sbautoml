# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:21:32 2024

@author: ataberk.urfali
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
    # Add other models similarly...

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
    # Add other optimizers...

# Loss fonksiyonu seçimi
def get_loss_function(loss_name):
    if loss_name == 'Cross Entropy Loss':
        return nn.CrossEntropyLoss()
    # Add other loss functions...

# Eğitim fonksiyonu
def train_model(data_dir, model_name, optimizer_name, loss_name, epochs, batch_size, learning_rate):
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
    loss_plot = "loss_plot.png"
    plt.savefig(loss_plot)

    return model, metrics_output, loss_plot

# Test fonksiyonu
def test_model(model, data_dir, batch_size, num_examples=8):
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

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=image_datasets.classes, yticklabels=image_datasets.classes)
    cm_image = "confusion_matrix.png"
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(cm_image)

    test_metrics = (
        f"F1 Score: {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Accuracy: {accuracy:.4f}"
    )

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

    return test_metrics, cm_image, example_outputs


# Function for Tabular Classification
def tabular_classification_interface(data_path, model_type, target_column, batch_size, learning_rate):
    data = pd.read_csv(data_path)
    data = data.fillna(0)  
    data = data.replace([np.inf, -np.inf], 0)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Tabular Classification")
    cm_image = "tabular_confusion_matrix.png"
    plt.savefig(cm_image)

    # Return the metrics and the confusion matrix image
    test_metrics = (
        f"F1 Score: {f1:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Accuracy: {accuracy:.4f}"
    )
    
    return test_metrics, cm_image

# Streamlit interface
st.markdown("Veri yolu, model seçimi, optimizer ve loss fonksiyonu seçimi ile model eğitimi ve test işlemleri gerçekleştirilir.")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Image Classification", "Tabular Classification"])
with tab1:
    st.subheader("Image Classification")

    # Dataset input
    data_dir = st.text_input("Dataset Yolunu Girin")
    
    # Model, optimizer, loss function selection
    model_name = st.selectbox("Model Seçimi", ["ResNet-18", "ResNet-50", "ResNet-101", "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201", "MobileNetV1", "MobileNetV2", "MobileNetV3", "VGG16", "VGG19"])
    optimizer_name = st.selectbox("Optimizer Seçimi", ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta"])
    loss_name = st.selectbox("Loss Fonksiyonu Seçimi", ["Cross Entropy Loss", "Binary Cross Entropy Loss", "Binary Cross Entropy with Logits Loss", "Mean Squared Error Loss", "Mean Absolute Error Loss (L1 Loss)", "Kullback-Leibler Divergence Loss", "Smooth L1 Loss (Huber Loss)"])
    
    epochs_1 = st.slider("Epoch Sayısı", min_value=1, max_value=100, step=1)
    batch_size_1 = st.selectbox("Batch Size", [1, 2, 4, 8, 16, 32, 64, 128, 256], key="batch_size_tabular_1")
    learning_rate_1 = st.number_input("Learning Rate", value=0.001)

    if st.button("Eğit"):
        # Here you'd call the image classification training function (not implemented in this part)
        st.write("Eğitim Başlatıldı...")
        # After training, display metrics and images
        metrics_output = "Metrics will be displayed here"
        test_results = "Test Results will be displayed here"
        cm_image = "path_to_confusion_matrix_image.png"  # Replace with actual image path
        loss_plot = "path_to_loss_plot.png"  # Replace with actual plot image
        example_outputs = ["example1.png", "example2.png"]  # Replace with actual outputs
        
        st.text(metrics_output)
        st.text(test_results)
        st.image(cm_image)
        st.image(loss_plot)
        st.image(example_outputs[0])  # Display the first example output


with tab2 :
    st.subheader("Tabular Classification")

    # Dataset input for tabular data
    data_path = st.text_input("Tabular Veri Dosya Yolu")

    # Dropdown for model selection
    model_type = st.selectbox("Model Tipi", ["RandomForest", "DecisionTree"])

    # Dynamic column selection based on dataset
    @st.cache_data
    def get_columns(data_path):
        data = pd.read_csv(data_path)
        return list(data.columns)

    if data_path:
        columns = get_columns(data_path)
        target_column = st.selectbox("Seçilecek Sütun", columns, key="id")
        
        batch_size_tabular_2 = st.selectbox("Batch Size", [1, 2, 4, 8, 16, 32, 64, 128, 256], key="batch_size_tabular_2")
        learning_rate_tabular_2 = st.number_input("Learning Rate", value=0.001, key="test1")

        if st.button("Eğit "):
            st.write("Model Eğitiliyor...")
            test_metrics, cm_image = tabular_classification_interface(
                data_path, model_type, target_column, batch_size_tabular_2, learning_rate_tabular_2
            )
            
            # Display results for tabular classification
            st.subheader("Tabular Eğitim Sonuçları")
            st.text(test_metrics)
            st.subheader("Konfüzyon Matrisi - Tabular")
            st.image(cm_image)