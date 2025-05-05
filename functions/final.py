
import os
import random
import tempfile
import shutil
import io
import webdataset as wds
import torch
import torch.optim as optim
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from config import CLASS_NAMES, CLASS_WEIGHTS, TRAIN_PATHS, TEST_PATHS
from functions.utility import print_gpu_memory
from functions.model import create_model
from functions.train import evaluate_full
from torchvision import transforms
from torch.utils.data import DataLoader
from config import DEVICE
from functions.dataload import get_transforms



def visualize_classifications(model, test_loader, save_dir="./"):
    """
    Visualize correctly and incorrectly classified images from the test set.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test dataset
        save_dir: Directory to save visualization images
    
    Returns:
        tuple: Paths to the saved visualization images
    """
    model.eval()
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Dictionaries to store correctly classified images by class
    correct_samples = {class_name: [] for class_name in CLASS_NAMES}
    incorrect_samples = []
    
    # Process images and collect samples
    print("Collecting sample images for visualization...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Finding samples"):
            # Move data to device
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Make predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Move data back to CPU for storage
            inputs_cpu = inputs.cpu()
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            # Check each image in the batch
            for i in range(len(labels_cpu)):
                true_label = labels_cpu[i]
                pred_label = preds_cpu[i]
                true_class = CLASS_NAMES[true_label]
                pred_class = CLASS_NAMES[pred_label]
                
                # If correctly classified, add to the correct class list
                if true_label == pred_label:
                    # Only keep one sample per class
                    if len(correct_samples[true_class]) < 1:
                        correct_samples[true_class].append({
                            'image': inputs_cpu[i],
                            'true_class': true_class,
                            'pred_class': pred_class
                        })
                # If incorrectly classified, add to incorrect list
                else:
                    incorrect_samples.append({
                        'image': inputs_cpu[i],
                        'true_class': true_class,
                        'pred_class': pred_class
                    })
            
            # Check if we have enough samples
            all_correct_filled = all(len(samples) >= 1 for samples in correct_samples.values())
            if all_correct_filled and len(incorrect_samples) >= 10:
                break
    
    # Function to denormalize images for display
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)
    
    # Create figure for correct classifications
    plt.figure(figsize=(15, 6))
    plt.suptitle("Correctly Classified Samples (1 per class)", fontsize=16)
    
    # Create a grid of correct images
    for i, class_name in enumerate(CLASS_NAMES):
        if correct_samples[class_name]:
            sample = correct_samples[class_name][0]
            plt.subplot(2, 5, i + 1)
            
            # Denormalize and convert to displayable format
            img = denormalize(sample['image']).permute(1, 2, 0).numpy()
            
            plt.imshow(img)
            plt.title(f"{class_name}", fontsize=10)
            plt.axis('off')
        else:
            plt.subplot(2, 5, i + 1)
            plt.text(0.5, 0.5, f"No correct\n{class_name}", 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    correct_path = os.path.join(save_dir, "correct_classifications.png")
    plt.savefig(correct_path, dpi=150)
    plt.close()  # Close the figure to free memory
    
    # Create figure for incorrect classifications
    plt.figure(figsize=(15, 6))
    plt.suptitle("Incorrectly Classified Samples", fontsize=16)
    
    # Randomly select incorrect samples if we have more than needed
    if len(incorrect_samples) > 10:
        incorrect_samples = random.sample(incorrect_samples, 10)
    
    # Create a grid of incorrect images
    for i, sample in enumerate(incorrect_samples[:10]):
        plt.subplot(2, 5, i + 1)
        
        # Denormalize and convert to displayable format
        img = denormalize(sample['image']).permute(1, 2, 0).numpy()
        
        plt.imshow(img)
        plt.title(f"True: {sample['true_class']}\nPred: {sample['pred_class']}", fontsize=9)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    incorrect_path = os.path.join(save_dir, "incorrect_classifications.png")
    plt.savefig(incorrect_path, dpi=150)
    plt.close()  # Close the figure to free memory
    
    print(f"Visualization images saved to {correct_path} and {incorrect_path}")
    
    return correct_path, incorrect_path




# Function to train the final model with best hyperparameters
def train_final_model(study, epochs_factor=1.2):
    """Train a final model with the best hyperparameters using all available data.
    Uses early stopping information from the best trial to avoid overfitting.
    """
    if study is None or len(study.trials) == 0:
        print("No valid study available. Cannot train final model.")
        return None, 0
        
    # Get best hyperparameters and trial
    best_trial = study.best_trial
    best_params = best_trial.params
    print("Training final model with best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Determine ideal training duration based on early stopping from best trial
    mlflow_run_id = best_trial.user_attrs.get("mlflow_run_id")
    
    if mlflow_run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(mlflow_run_id)

            final_epochs = int(run.data.params["recommended_epochs"])
            print(f"Using recommended_epochs from k-fold CV: {final_epochs}")
        except Exception as e:
            print(f"Could not retrieve early stopping info: {e}")
            final_epochs = int(best_params["max_epochs"] * 0.8)
            print(f"Using fallback: will train for {final_epochs} epochs")
    else:
        final_epochs = int(best_params["max_epochs"] * 0.8)
        print(f"No run info available. Will train for {final_epochs} epochs")
    
    # Create combined train+val loader
    combined_paths = TRAIN_PATHS
    print(f"Using all {len(combined_paths)} training shards")
    
    # Use the best hyperparameters
    augmentation_intensity = best_params["augmentation_intensity"]
    batch_size = best_params["batch_size"]
    
    # Create dataloaders
    train_transform, _ = get_transforms(augmentation_intensity)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define preprocessing functions
    def preprocess_train(sample):
        image = Image.open(io.BytesIO(sample['jpg']))
        tensor_image = train_transform(image)
        class_name = sample['cls'].decode('utf-8')
        class_idx = CLASS_NAMES.index(class_name)
        return tensor_image, class_idx
    
    def preprocess_test(sample):
        image = Image.open(io.BytesIO(sample['jpg']))
        tensor_image = test_transform(image)
        class_name = sample['cls'].decode('utf-8')
        class_idx = CLASS_NAMES.index(class_name)
        return tensor_image, class_idx
    
    # Create datasets
    full_train_dataset = (
        wds.WebDataset(combined_paths, shardshuffle=True)
        .map(preprocess_train)
        .shuffle(1000)
    )
    
    test_dataset = (
        wds.WebDataset(TEST_PATHS, shardshuffle=False, resampled=False)
        .map(preprocess_test)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    
    # Create model with best hyperparameters
    model, criterion, optimizer = create_model(
        learning_rate=best_params["learning_rate"],
        dropout_rate=best_params["dropout_rate"],
        weight_decay=best_params["weight_decay"],
        class_weights=CLASS_WEIGHTS
    )
    
    # Create temporary directory for artifacts
    artifact_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for artifacts: {artifact_dir}")
    
    # Start MLflow run for final model
    with mlflow.start_run(run_name="final_model_full_training"):
        mlflow.log_params({
            **best_params,
            "training_type": "full_dataset",
            "early_stopping": False,
            "actual_epochs": final_epochs
        })
        
        # Calculate steps per epoch
        steps_per_epoch = int(16384/batch_size)  # A reasonable default estimate
        
        # Training loop with fixed number of epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=final_epochs,
            anneal_strategy='cos'
        )
        
        # Store best model state
        best_model_state = None
        
        # Training loop
        for epoch in range(final_epochs):
            print(f"\nEpoch {epoch+1}/{final_epochs}")
            print("-" * 20)
            
            # Train for one epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc="Training")
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                # Move data to device
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
                
                # Periodically clear GPU cache
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / total if total > 0 else 0
            epoch_acc = 100 * correct / total if total > 0 else 0
            
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
            print_gpu_memory()

            # Call scheduler.step() once per epoch
            scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_acc": epoch_acc
            }, step=epoch)
            
            # Save checkpoint to the artifact directory
            checkpoint_path = os.path.join(artifact_dir, 'model_checkpoint.pth')
            torch.save(model.state_dict(), checkpoint_path)
            
            # Only in the last epoch, keep the state for final evaluation
            if epoch == final_epochs - 1:
                best_model_state = model.state_dict().copy()
        
        # Load the best model state if we have one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save the final model to the artifact directory
        model_path = os.path.join(artifact_dir, 'final_model.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "final_model")
        
        # Only evaluate once on the full test set
        print("\nEvaluating final model on test set...")
        test_loss, test_acc, test_preds, test_labels = evaluate_full(model, test_loader, criterion)
        
        print(f"\nFinal Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        
        # Log test metrics
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        # Generate and log per-class metrics
        if len(test_preds) > 0 and len(test_labels) > 0:
            try:
                report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, output_dict=True)
                
                # Log per-class metrics
                for class_name in CLASS_NAMES:
                    mlflow.log_metric(f"test_f1_{class_name}", report[class_name]['f1-score'])
                    mlflow.log_metric(f"test_precision_{class_name}", report[class_name]['precision'])
                    mlflow.log_metric(f"test_recall_{class_name}", report[class_name]['recall'])
                
                # Create and log confusion matrix to artifact directory
                cm = confusion_matrix(test_labels, test_preds)
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Final Model Confusion Matrix (Test Set)')
                plt.tight_layout()
                
                cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)

                # Generate classification visualizations
                print("\nGenerating classification visualizations...")
                visualize_classifications(model, test_loader, save_dir=artifact_dir)
                
                # Log all artifacts from the directory
                for artifact_file in os.listdir(artifact_dir):
                    if artifact_file.endswith('.png'):
                        mlflow.log_artifact(os.path.join(artifact_dir, artifact_file))

            except Exception as e:
                print(f"Error generating classification report: {e}")
        
        # Clean up the temporary directory
        try:
            shutil.rmtree(artifact_dir)
            print(f"Cleaned up temporary artifact directory: {artifact_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")
                
    return model, test_acc


def display_model_visualizations(run_id=None, experiment_name="animals10"):
    """
    Find and display visualization artifacts (confusion matrix, correct and 
    incorrect classifications) from the best MLflow run.
    
    Args:
        run_id (str, optional): Specific MLflow run ID to use. If None, uses the best run.
        experiment_name (str): Name of the MLflow experiment to search.
    
    Returns:
        dict: Dictionary with paths to downloaded artifacts
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    
    # Create MLflow client
    client = MlflowClient()
    
    # Find experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    
    # Get the run
    if run_id is None:
        # Find all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            print(f"No runs found in experiment '{experiment_name}'.")
            return None
            
        # Try to find final_model runs with test accuracy
        final_model_runs = runs[runs["tags.mlflow.runName"].str.contains("final_model", na=False)]
        
        if len(final_model_runs) > 0 and "metrics.test_acc" in final_model_runs.columns:
            # Use the final model run with highest test accuracy
            best_run = final_model_runs.sort_values("metrics.test_acc", ascending=False).iloc[0]
            run_id = best_run.run_id
            test_acc = best_run["metrics.test_acc"]
            print(f"Using final model run {run_id} with test accuracy: {test_acc:.2f}%")
        elif "metrics.test_acc" in runs.columns:
            # Fall back to any run with test accuracy
            best_run = runs.sort_values("metrics.test_acc", ascending=False).iloc[0]
            run_id = best_run.run_id
            test_acc = best_run["metrics.test_acc"]
            print(f"Using run {run_id} with test accuracy: {test_acc:.2f}%")
        else:
            # Just take the most recent run
            run_id = runs.iloc[0].run_id
            print(f"Using most recent run: {run_id}")
    
    print(f"Searching for visualization artifacts in run: {run_id}")
    
    # Define the artifacts we're looking for
    target_artifacts = [
        "confusion_matrix.png",
        "correct_classifications.png",
        "incorrect_classifications.png"
    ]
    
    # Recursive function to search for artifacts in nested directories
    def find_artifact_paths(client, run_id, artifact_names, path=""):
        found_paths = {}
        artifacts = client.list_artifacts(run_id, path)
        
        for artifact in artifacts:
            full_path = os.path.join(path, artifact.path) if path else artifact.path
            
            if artifact.is_dir:
                # Search recursively in subdirectories
                nested_paths = find_artifact_paths(client, run_id, artifact_names, full_path)
                found_paths.update(nested_paths)
            elif os.path.basename(full_path) in artifact_names:
                found_paths[os.path.basename(full_path)] = full_path
        
        return found_paths
    
    # Find all matching artifacts
    artifact_paths = find_artifact_paths(client, run_id, target_artifacts)
    
    if not artifact_paths:
        print("\nNo visualization artifacts found in this run.")
        print("Available artifacts:")
        for a in client.list_artifacts(run_id):
            print(f"- {a.path}")
        return {}
    
    # Dictionary to store downloaded artifacts
    downloaded_artifacts = {}
    
    # Download and display each artifact if available
    for target in target_artifacts:
        if target in artifact_paths:
            artifact_path = artifact_paths[target]
            
            # Download the artifact
            local_path = client.download_artifacts(run_id, artifact_path)
            downloaded_artifacts[target] = local_path
            
            # Create a nice title
            title = target.replace('.png', '').replace('_', ' ').title()
            print(f"\n## {title}")
            
            try:
                # Load and display the image
                img = Image.open(local_path)
                plt.figure(figsize=(14, 12))
                plt.imshow(img)
                plt.axis('off')
                plt.title(title, fontsize=14)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error displaying {target}: {e}")
        else:
            print(f"\n## {target.replace('.png', '').replace('_', ' ').title()}")
            print(f"Artifact '{target}' not found in run {run_id}")
    
    return downloaded_artifacts