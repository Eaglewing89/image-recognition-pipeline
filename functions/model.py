# Import necessary libraries
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from config import NUM_CLASSES, device


# Define the model architecture with ResNet50 backbone
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2, freeze_backbone=True):
        super(AnimalClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionally unfreeze the last convolutional block
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the original FC layer
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# Function to create model, loss function, and optimizer
def create_model(learning_rate, dropout_rate, weight_decay, class_weights=None):
    model = AnimalClassifier(NUM_CLASSES, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Create weighted loss function if class weights are provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=weight_decay)
    
    return model, criterion, optimizer