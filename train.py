import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_stack_embeddings(folder_path):
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])
    all_embeddings = []
    labels = None
    for file in file_list:
        data = torch.load(os.path.join(folder_path, file))
        emb = data['embeddings']  # (N, D)
        all_embeddings.append(emb)
        if labels is None and 'labels' in data:
            labels = data['labels']
    if labels is None:
        raise RuntimeError(f"No 'labels' found in {folder_path}")
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return torch.stack(all_embeddings, dim=1), labels

class HierarchicalCNN(nn.Module):
    def __init__(self, input_dim, num_classes, kernel_sizes=[2,3,4], dropout=0.5):
        super(HierarchicalCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=k, groups=input_dim)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim * len(kernel_sizes), 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            conv_x = F.relu(conv(x))
            pooled = F.avg_pool1d(conv_x, kernel_size=conv_x.shape[2])
            conv_outs.append(pooled.squeeze(2))
        out = torch.cat(conv_outs, dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, logits, labels, is_multilabel=False):
        if is_multilabel:
            ce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, labels.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def infoNCE_loss(features, labels, temperature=0.05):
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature
    labels = labels.view(-1,1)
    mask = torch.eq(labels, labels.T).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

train_folder = "./train"
val_folder = "./val"
test_folder = "./test"
batch_size = 64
epochs = 50
lr = 1e-4
weight_decay = 1e-5
patience = 5
save_path = "./modelsave/best_model.pt"

X_train, y_train = load_and_stack_embeddings(train_folder)
X_val, y_val = load_and_stack_embeddings(val_folder)
X_test, y_test = load_and_stack_embeddings(test_folder)
is_multilabel = (y_train.ndim == 2)
num_classes = y_train.shape[1] if is_multilabel else int(y_train.max()) + 1
embed_dim = X_train.shape[2]
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
model = HierarchicalCNN(input_dim=embed_dim, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
ce_loss_fn = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
focal_loss_fn = FocalLoss(gamma=2.0)
beta1, beta2, beta3 = 1.0, 0.5, 0.5

best_f1 = 0
counter = 0
for epoch in range(1, epochs+1):
    model.train()
    train_loss, train_preds, train_labels = 0.0, [], []
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if is_multilabel:
            ce_loss = ce_loss_fn(logits, yb.float())
            focal_loss = focal_loss_fn(logits, yb, is_multilabel=True)
            contrastive_loss_val = 0.0
        else:
            ce_loss = ce_loss_fn(logits, yb.long())
            focal_loss = focal_loss_fn(logits, yb)
            contrastive_loss_val = infoNCE_loss(logits, yb)
        loss = beta1*ce_loss + beta2*focal_loss + beta3*contrastive_loss_val
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        preds = (torch.sigmoid(logits) if is_multilabel else logits.argmax(dim=1))
        train_preds.extend(preds.cpu().tolist())
        train_labels.extend(yb.cpu().tolist())
    train_loss /= len(train_loader.dataset)
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    model.eval()
    val_loss, val_preds, val_labels = 0.0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if is_multilabel:
                ce_loss = ce_loss_fn(logits, yb.float())
                focal_loss = focal_loss_fn(logits, yb, is_multilabel=True)
                contrastive_loss_val = 0.0
            else:
                ce_loss = ce_loss_fn(logits, yb.long())
                focal_loss = focal_loss_fn(logits, yb)
                contrastive_loss_val = infoNCE_loss(logits, yb)
            loss = beta1*ce_loss + beta2*focal_loss + beta3*contrastive_loss_val
            val_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) if is_multilabel else logits.argmax(dim=1))
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(yb.cpu().tolist())
    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
          f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

    if val_f1 >= best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), save_path)
        print("Saved Best Model")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
def test_model(model, test_loader, is_multilabel, num_classes, model_path=None):
    device = next(model.parameters()).device
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if is_multilabel:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.6).int()
            else:
                preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(yb.cpu().tolist())
    print("Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4))
    print(f"ACCURACY: {accuracy_score(test_labels, test_preds)}")
test_model(model, test_loader, is_multilabel, num_classes, model_path=save_path)