import torch
import torch.optim as optim
from tqdm.auto import tqdm
import mlflow
from config import device
from functions.utility import print_gpu_memory

# Training function
def train_epoch(model, train_loader, criterion, optimizer, scheduler=None, disable_progress=False, batch_size=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(train_loader, desc="Training", disable=disable_progress)
    
    # Limit batches during training to avoid exhausting dataset
    max_batches = int(1024 / batch_size)  # 1024 samples per epoch
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Stop after max_batches
        if batch_idx >= max_batches:
            break
            
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
    
    # Calculate epoch statistics (only for the batches we processed)
    if total > 0:  # Avoid division by zero
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
    else:
        epoch_loss = 0
        epoch_acc = 0
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, val_loader, criterion, disable_progress=False, batch_size=8):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Limit batches for evaluation to avoid exhausting dataset
    max_batches = int(512 / batch_size)  # 512 samples for evaluation
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Evaluating", disable=disable_progress)):
            # Stop after max_batches
            if batch_idx >= max_batches:
                break
                
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for later analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate statistics
    if total > 0:  # Avoid division by zero
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
    else:
        epoch_loss = 0
        epoch_acc = 0
        all_preds = []
        all_labels = []
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Full training loop with early stopping
def train_model(model, criterion, optimizer, train_loader, val_loader, max_epochs=20, patience=5, verbose=True, batch_size=8):
    # Initialize the learning rate scheduler (OneCycleLR)
    # WebDataset doesn't support len(), so we hardcode the steps_per_epoch
    steps_per_epoch = int(1024/batch_size)  # We're limiting to 100 batches per epoch in train_epoch anyway
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        anneal_strategy='cos'
    )
    
    # Initialize tracking variables
    best_val_acc = 0.0
    best_model_state = None
    no_improve_epochs = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Clear GPU cache before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print_gpu_memory()
    
    # Training loop
    for epoch in range(max_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{max_epochs}")
            print("-" * 20)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, disable_progress=not verbose, batch_size=batch_size)
        
        # Clear intermediate tensors to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, disable_progress=not verbose, batch_size=batch_size)
        
        # Print results if verbose mode is enabled
        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print_gpu_memory()
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Log metrics with MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early stopping check
        if no_improve_epochs >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc
