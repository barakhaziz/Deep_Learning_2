import os
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from siameseNetwork import SiameseNetwork

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4 
WEIGHT_DECAY = 1e-6
IMAGE_SIZE = 105
DATA_PATH = "data/lfwa_data/lfw2/lfw2"
TRAIN_CSV = "data/parsed_train.csv"
TEST_CSV = "data/parsed_test.csv"
CHECKPOINT_PATH = "model_checkpnt.pt"

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(),        # Convert to grayscale
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize images
    transforms.ToTensor(),         # Convert to PyTorch tensor
])

# Dataset class for loading image pairs
class SiamesePairsDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        """
        Args:
            csv_file: CSV file with image pairs
            data_dir: Directory containing the images
            transform: Image transformations to apply
        """
        self.pairs_df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        # Get pair information
        row = self.pairs_df.iloc[idx]
        name1, img_id1 = row['name_1'], row['image_1']
        name2, img_id2 = row['name_2'], row['image_2']
        label = row['posative_pair']
        
        # Build image paths
        img1_path = os.path.join(self.data_dir, name1, f"{name1}_{img_id1:04d}.jpg")
        img2_path = os.path.join(self.data_dir, name2, f"{name2}_{img_id2:04d}.jpg")
        
        # Load images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Early stopping class to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0
    
    def __call__(self, epoch, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            self.epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            self.counter = 0
            self.epoch = epoch

# Function to plot training results
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(history['train loss'], label='Train Loss')
    ax1.plot(history['validation loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # AUC curves
    ax2.plot(history['train ROC-AUC'], label='Train AUC')
    ax2.plot(history['validation ROC-AUC'], label='Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Training and Validation ROC-AUC')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # Load training data
    print("Loading training data...")
    train_dataset = SiamesePairsDataset(
        csv_file=TRAIN_CSV,
        data_dir=DATA_PATH,
        transform=transform
    )
    
    # Split into training and validation
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))),
        test_size=0.2,
        random_state=42
    )
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4
    )
    
    # Load test data
    print("Loading test data...")
    test_dataset = SiamesePairsDataset(
        csv_file=TEST_CSV,
        data_dir=DATA_PATH,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model = SiameseNetwork(expanded_linear=True)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Define early stopping
    early_stopping = EarlyStopping(patience=20)
    
    # Train the model
    print("Starting training...")
    history = model.train_model(
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        epoch=NUM_EPOCHS,
        optimizer=optimizer,
        loss_criterion=criterion,
        scheduler=scheduler,
        early_stopping=early_stopping
    )
    
    # Plot training results
    plot_training_history(history)
    
    # Evaluate model on test data
    print("Evaluating model on test data...")
    test_loss, test_auc, test_accuracy, _ = model.evaluate_model(test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "siamese_model_final.pt")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
