       
from functions.utility import print_gpu_memory

import optuna
from optuna.samplers import TPESampler
import torch
from config import TRAIN_PATHS, CLASS_NAMES, CLASS_WEIGHTS, SEED
import webdataset as wds
import io
from PIL import Image
import numpy as np
import mlflow
from scipy import stats
from torch.utils.data import DataLoader
from functions.model import create_model
from functions.train import train_model, train_epoch, evaluate
from functions.dataload import get_transforms
from functions.dataload import create_stratified_kfolds
import torch.optim as optim

import collections
import matplotlib.pyplot as plt
from IPython.display import display



def objective_kfold(trial, k=3, verbose=True, first_fold_min_acc=90.0, notebook=False):
    """
    Objective function with k-fold cross validation for Optuna optimization.
    Implements a hybrid pruning approach:
    1. Early pruning based on first fold performance
    2. Epoch-level pruning within folds
    3. Progressive fold evaluation
    
    Args:
        trial: Optuna trial object
        k: Number of folds for cross-validation
        verbose: Whether to print verbose output
        
    Returns:
        float: t-statistic lower bound of validation accuracy
    """
    
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    augmentation_intensity = trial.suggest_categorical("augmentation_intensity", ["low", "medium", "high"])
    patience = trial.suggest_int("patience", 3, 6)
    max_epochs = trial.suggest_int("max_epochs", 10, 20)
    
    # Force smaller batch size for GPU memory constraints
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 8.0:
            batch_size = min(batch_size, 32)
        torch.cuda.empty_cache()
    
    # Create k-fold split of the training shards
    folds = create_stratified_kfolds(TRAIN_PATHS, k=k, seed=SEED, verbose=verbose)
    
    # Start MLflow run for this trial
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}_kfold"):
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "augmentation_intensity": augmentation_intensity,
            "patience": patience,
            "max_epochs": max_epochs,
            "k_folds": k
        })
        
        # Set threshold for first-fold pruning
        # This should be set based on domain knowledge; here using a conservative value
        first_fold_min_acc = first_fold_min_acc  # Minimum acceptable accuracy for first fold (%)
        
        # Track validation accuracies for each fold
        fold_accuracies = []
        epoch_accuracies = {}  # Dictionary to store accuracies by epoch
        
        # For each fold
        for fold_idx in range(k):
            if verbose:
                print(f"\n--- Fold {fold_idx+1}/{k} ---")
            
            # Define train and validation shards for this fold
            val_shard_paths = folds[fold_idx]
            train_shard_paths = []
            for i in range(k):
                if i != fold_idx:
                    train_shard_paths.extend(folds[i])
            
            if verbose:
                print(f"Training on {len(train_shard_paths)} shards, validating on {len(val_shard_paths)} shards")
            
            # Get transforms
            train_transform, val_transform = get_transforms(augmentation_intensity)
            
            # Define preprocessing functions
            def preprocess_train(sample):
                image = Image.open(io.BytesIO(sample['jpg']))
                tensor_image = train_transform(image)
                class_name = sample['cls'].decode('utf-8')
                class_idx = CLASS_NAMES.index(class_name)
                return tensor_image, class_idx
            
            def preprocess_val(sample):
                image = Image.open(io.BytesIO(sample['jpg']))
                tensor_image = val_transform(image)
                class_name = sample['cls'].decode('utf-8')
                class_idx = CLASS_NAMES.index(class_name)
                return tensor_image, class_idx
            
            # Create datasets
            train_dataset = (
                wds.WebDataset(train_shard_paths, shardshuffle=True)
                .map(preprocess_train)
                .shuffle(1000)
            )
            
            val_dataset = (
                wds.WebDataset(val_shard_paths, shardshuffle=False)
                .map(preprocess_val)
            )
            
            # Create data loaders
            if notebook:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
            else:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True)
            
            # Create model with the specified hyperparameters
            model, criterion, optimizer = create_model(
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                class_weights=CLASS_WEIGHTS
            )
            
            try:
                # For the first fold, implement custom training with per-epoch pruning checks
                if fold_idx == 0:
                    # Implement a custom training loop that allows epoch-level pruning for first fold
                    scheduler = optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=optimizer.param_groups[0]['lr'],
                        steps_per_epoch=int(1024/batch_size),  # Same as in train_model
                        epochs=max_epochs,
                        anneal_strategy='cos'
                    )
                    
                    # Initialize tracking variables
                    best_val_acc = 0.0
                    best_model_state = None
                    no_improve_epochs = 0
                    fold_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
                    
                    # Training loop with pruning checks
                    for epoch in range(max_epochs):
                        if verbose:
                            print(f"Epoch {epoch+1}/{max_epochs}")
                        
                        # Train for one epoch
                        train_loss, train_acc = train_epoch(
                            model, train_loader, criterion, optimizer, scheduler, 
                            disable_progress=not verbose,
                            batch_size=batch_size
                        )
                        
                        # Clear intermediate tensors to free up memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Evaluate on validation set
                        val_loss, val_acc, _, _ = evaluate(
                            model, val_loader, criterion, 
                            disable_progress=not verbose,
                            batch_size=batch_size
                        )
                        
                        # Print results if verbose
                        if verbose:
                            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                        
                        # Update history
                        fold_history["train_loss"].append(train_loss)
                        fold_history["train_acc"].append(train_acc)
                        fold_history["val_loss"].append(val_loss)
                        fold_history["val_acc"].append(val_acc)
                        
                        # Store the validation accuracy for this epoch
                        if epoch not in epoch_accuracies:
                            epoch_accuracies[epoch] = []
                        epoch_accuracies[epoch].append(val_acc)
                        
                        # Report to Optuna for pruning
                        trial.report(val_acc, epoch)
                        
                        # Check if this trial should be pruned based on current epoch performance
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned(
                                f"Trial pruned at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%"
                            )
                        
                        # Early stopping check
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_model_state = model.state_dict().copy()
                            no_improve_epochs = 0
                        else:
                            no_improve_epochs += 1
                        
                        if no_improve_epochs >= patience:
                            if verbose:
                                print(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    
                    # After first fold, check if performance is promising
                    if best_val_acc < first_fold_min_acc:
                        if verbose:
                            print(f"First fold accuracy ({best_val_acc:.2f}%) below threshold ({first_fold_min_acc:.2f}%). Pruning trial.")
                        
                        # Log that we pruned based on first fold
                        mlflow.log_param("pruned_first_fold", True)
                        mlflow.log_metric("first_fold_acc", best_val_acc)
                        
                        # Raise pruned exception
                        raise optuna.exceptions.TrialPruned(
                            f"Trial pruned after first fold with best val acc: {best_val_acc:.2f}%"
                        )
                    
                    # Add the best accuracy from first fold
                    fold_accuracies.append(best_val_acc)
                    
                # For subsequent folds, use the standard train_model function
                else:
                    # Train the model normally for subsequent folds
                    model, history, best_val_acc = train_model(
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        max_epochs=max_epochs,
                        patience=patience,
                        verbose=verbose and fold_idx == 0,  # Only show verbose output for first fold
                        batch_size=batch_size
                    )
                    
                    # Store the validation accuracy for this fold
                    fold_accuracies.append(best_val_acc)

                    # Store validation accuracies by epoch
                    for epoch, val_acc in enumerate(history['val_acc']):
                        if epoch not in epoch_accuracies:
                            epoch_accuracies[epoch] = []
                        epoch_accuracies[epoch].append(val_acc)
                
                # After each fold, check if the trial is promising compared to others
                if fold_idx > 0:  # Skip check for first fold (already handled)
                    # Calculate average accuracy across completed folds
                    avg_acc_so_far = sum(fold_accuracies) / len(fold_accuracies)
                    
                    # Log the current average
                    mlflow.log_metric(f"avg_acc_after_{fold_idx+1}_folds", avg_acc_so_far)
                    
                    # Get median of completed trials so far (if we have enough)
                    completed_trials = [t for t in trial.study.trials 
                                      if t.state == optuna.trial.TrialState.COMPLETE]
                    
                    if len(completed_trials) >= 20:  # Need at least 20 completed trials for meaningful comparison
                        values = [t.value for t in completed_trials if t.value is not None]
                        if values:  # Make sure we have valid values
                            median_value = np.median(values)
                            
                            # If we're significantly below median, prune
                            if avg_acc_so_far < (median_value * 0.85):  # Threshold at 85% of median
                                if verbose:
                                    print(f"Performance ({avg_acc_so_far:.2f}%) below 85% of median ({median_value:.2f}%). Pruning.")
                                
                                # Log pruning decision
                                mlflow.log_param("pruned_progressive", True)
                                mlflow.log_param("pruned_after_fold", fold_idx + 1)
                                mlflow.log_metric("avg_acc_at_pruning", avg_acc_so_far)
                                mlflow.log_metric("median_value_at_pruning", median_value)
                                
                                # Raise pruned exception
                                raise optuna.exceptions.TrialPruned(
                                    f"Trial pruned after fold {fold_idx+1}/{k} with avg_acc={avg_acc_so_far:.2f}% (below median)"
                                )
                
                # Clear GPU memory after each fold
                del model
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                # Handle CUDA memory errors
                if "CUDA" in str(e) and ("out of memory" in str(e) or "unknown error" in str(e)):
                    if verbose:
                        print(f"CUDA memory error in fold {fold_idx+1}")
                    torch.cuda.empty_cache()
                    return float('-inf')
                else:
                    raise e
                
        # Calculate metrics using the epoch-wise approach
        epoch_avg_accuracies = {}
        epoch_std_accuracies = {}
        
        # Only use epochs where all folds have data
        for epoch, accs in epoch_accuracies.items():
            if len(accs) == k:  # Only consider epochs with data from all k folds
                # Calculate average
                epoch_avg = sum(accs) / k
                epoch_avg_accuracies[epoch] = epoch_avg
                
                # Log with consistent metric name and varying step value
                mlflow.log_metric("epoch_avg_val_acc", epoch_avg, step=epoch)
                
                # Calculate standard deviation
                if len(accs) > 1:  # Need at least 2 values for std
                    epoch_std = np.std(accs)
                    epoch_std_accuracies[epoch] = epoch_std
                    
                    # Log standard deviation as a separate metric with the same step
                    mlflow.log_metric("epoch_std_val_acc", epoch_std, step=epoch)

        # Find best average validation accuracy and its epoch
        if epoch_avg_accuracies:
            best_epoch = max(epoch_avg_accuracies.items(), key=lambda x: x[1])[0]
            best_avg_val_acc = epoch_avg_accuracies[best_epoch]
            
            mlflow.log_metric("best_avg_epoch", best_epoch)
            mlflow.log_metric("best_avg_val_acc", best_avg_val_acc)

            # Log this as the recommended number of epochs for final training
            mlflow.log_param("recommended_epochs", best_epoch + 1)  # +1 because epochs are 0-indexed
            
            # Log the standard deviation of the best epoch if available
            if best_epoch in epoch_std_accuracies:
                best_std_val_acc = epoch_std_accuracies[best_epoch]
                
                # Use 80% confidence with t-distribution for small samples
                confidence_level = 0.80
                t_critical = stats.t.ppf(confidence_level, df=k-1)
                
                # Calculate lower bound
                lower_bound = best_avg_val_acc - (t_critical * best_std_val_acc / np.sqrt(k))
                
                # Log all metrics
                mlflow.log_metric("best_avg_val_acc", best_avg_val_acc)
                mlflow.log_metric("best_std_val_acc", best_std_val_acc)
                mlflow.log_metric("lower_confidence_bound", lower_bound)
                
                print(f"Best avg validation at epoch {best_epoch+1}: {best_avg_val_acc:.2f}% ± {best_std_val_acc:.2f}%")
                print(f"Objective value - t-dist Lower confidence bound ({confidence_level*100}%): {lower_bound:.2f}%")

                # Store MLflow run id in trial user attributes for later reference
                trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
                
                return lower_bound
        else:
            return float('-inf')
 




def run_kfold_optuna_optimization(n_trials=5, k=5, verbose=True, study_name="animals10_kfold", storage=None, load_if_exists=True, first_fold_min_acc=90.0, notebook=False):
    """
    Run Optuna hyperparameter optimization with k-fold cross validation
    
    Args:
        n_trials: Number of Optuna trials to run
        k: Number of folds for cross-validation
        verbose: Whether to print verbose output
        study_name: Name of the Optuna study
        storage: Storage URL for the Optuna study
        load_if_exists: Whether to load existing study if it exists
        
    Returns:
        optuna.Study: The completed Optuna study
    """
    # Prepare storage string for SQLite if provided
    storage_string = None
    if storage:
        storage_string = f"sqlite:///{storage}"
        print(f"Using SQLite storage at: {storage_string}")
    
    # Configure logging for Optuna
    if not verbose:
        import logging
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.setLevel(logging.WARNING)
    
    # Create a fresh study object or load existing one
    study = None
    
    if storage and load_if_exists:
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_string
            )
            
            if len(study.trials) > 0:
                print(f"Loaded existing study with {len(study.trials)} previous trials.")
                print(f"Best value so far: {study.best_value} t-dist 80% lower bound, with parameters:")
                for key, value in study.best_params.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Could not load existing study: {e}")
            study = None
    
    if study is None:
        # Create a pruner - here we use the MedianPruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20,  # Number of trials to run before pruning starts
            n_warmup_steps=10,    # Number of epochs to run before pruning can happen in a trial
            interval_steps=1     # Evaluate pruning conditions after each epoch
        )
        
        study = optuna.create_study(
            direction="maximize",  # Maximize validation accuracy
            sampler=TPESampler(seed=SEED),
            pruner=pruner,        # Add the pruner here
            study_name=study_name,
            storage=storage_string,
            load_if_exists=load_if_exists
        )
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print_gpu_memory()
    
    # Define objective function wrapper
    def objective_wrapper(trial):
        return objective_kfold(trial, k=k, verbose=verbose, first_fold_min_acc=first_fold_min_acc, notebook=notebook)
    
    # Create callback function to show progress
    already_printed_trials = set()
    
    def trial_callback(study, trial):
        if trial.number not in already_printed_trials and trial.state in [
            optuna.trial.TrialState.COMPLETE, 
            optuna.trial.TrialState.PRUNED
        ]:
            already_printed_trials.add(trial.number)
            if not verbose:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    print(f"Trial {trial.number} completed with value: {trial.value:.2f}")
                elif trial.state == optuna.trial.TrialState.PRUNED:
                    print(f"Trial {trial.number} pruned at epoch {trial.last_step}")
    
    # Run optimization
    try:
        study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[trial_callback])
        
        # Print summary
        print("\nK-Fold Study statistics:")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Best trial:")
        trial = study.best_trial
        
        if trial.value is not None:
            print(f"    Value: {trial.value} t-dist 80% lower bound")
            print(f"    Params:")
            for key, value in trial.params.items():
                print(f"      {key}: {value}")
        else:
            print("    No valid value found for best trial.")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    return study


def visualize_best_trial_metrics(study, mlflow_client=None):
    """
    Visualize epoch-wise metrics from the best trial in an Optuna study.
    Shows average validation accuracy and standard deviation over epochs.
    
    Args:
        study: Optuna study object
        mlflow_client: MLflow client (optional, will be created if None)
    """
    if study is None or not study.best_trial:
        print("No best trial available for visualization.")
        return
    
    best_trial = study.best_trial
    print(f"Best Trial: #{best_trial.number}")
    print(f"Value: {best_trial.value}")
    print("\nParameters:")
    for param_name, param_value in best_trial.params.items():
        print(f"  {param_name}: {param_value}")
    
    # Get MLflow run ID from trial user attributes
    mlflow_run_id = best_trial.user_attrs.get("mlflow_run_id")
    if not mlflow_run_id:
        print("\nCannot find MLflow run ID in trial user attributes.")
        return
    
    print(f"\nMLflow Run ID: {mlflow_run_id}")
    
    # Create MLflow client if not provided
    if mlflow_client is None:
        mlflow_client = mlflow.tracking.MlflowClient()
    
    # Get run data
    try:
        run = mlflow_client.get_run(mlflow_run_id)
    except Exception as e:
        print(f"Error retrieving MLflow run: {e}")
        return
    
    # Fetch all metrics history
    avg_acc_history = mlflow_client.get_metric_history(mlflow_run_id, "epoch_avg_val_acc")
    std_acc_history = mlflow_client.get_metric_history(mlflow_run_id, "epoch_std_val_acc")
    
    if not avg_acc_history:
        print("No epoch_avg_val_acc metrics found for this run.")
        return
    
    # Extract epoch (step) and values
    epochs = [m.step for m in avg_acc_history]
    avg_values = [m.value for m in avg_acc_history]
    
    std_values = []
    if std_acc_history:
        # Make sure std_values aligns with avg_values epochs
        std_dict = {m.step: m.value for m in std_acc_history}
        std_values = [std_dict.get(epoch, 0) for epoch in epochs]
    
    # Best epoch from params (subtract 1 because epochs are 0-indexed in the code)
    best_avg_epoch = int(run.data.params.get("recommended_epochs", 0)) - 1
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot average validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(epochs, avg_values, 'b-', marker='o', label='Avg Validation Accuracy')
    if best_avg_epoch in epochs:
        best_value = avg_values[epochs.index(best_avg_epoch)]
        plt.axvline(x=best_avg_epoch, color='r', linestyle='--', 
                   label=f'Best Epoch: {best_avg_epoch+1} (Acc: {best_value:.2f}%)')
        plt.plot(best_avg_epoch, best_value, 'ro', markersize=8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Average Validation Accuracy Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot standard deviation
    if std_values:
        plt.subplot(2, 1, 2)
        plt.plot(epochs, std_values, 'g-', marker='o', label='Std Dev of Validation Accuracy')
        if best_avg_epoch in epochs:
            best_std = std_values[epochs.index(best_avg_epoch)]
            plt.axvline(x=best_avg_epoch, color='r', linestyle='--', 
                       label=f'Best Epoch: {best_avg_epoch+1} (Std: {best_std:.2f}%)')
            plt.plot(best_avg_epoch, best_std, 'ro', markersize=8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Standard Deviation (%)')
        plt.title('Standard Deviation of Validation Accuracy Across Folds')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Display best epoch metrics
    if best_avg_epoch in epochs:
        idx = epochs.index(best_avg_epoch)
        best_avg = avg_values[idx]
        best_std = std_values[idx] if std_values else 0
        
        print(f"\nBest epoch metrics (Epoch {best_avg_epoch+1}):")
        print(f"  Average validation accuracy: {best_avg:.2f}%")
        if std_values:
            print(f"  Standard deviation: {best_std:.2f}%")
            # Calculate confidence intervals
            k = int(run.data.params.get("k_folds", 3))
            confidence_level = 0.8  # 80% confidence level used in your study
            t_critical = stats.t.ppf(confidence_level, df=k-1)
            lower_bound = best_avg - (t_critical * best_std / np.sqrt(k))
            upper_bound = best_avg + (t_critical * best_std / np.sqrt(k))
            print(f"  {confidence_level*100:.0f}% confidence interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
            print(f"  Lower bound (study objective): {lower_bound:.2f}%")


def enhanced_optuna_analysis(study, show_plots=True, top_n_trials=10):
    """
    Display comprehensive insights about an Optuna hyperparameter optimization study.
    
    Args:
        study: Optuna study object
        show_plots: Whether to display visualizations
        top_n_trials: Number of top trials to analyze in detail
        
    Returns:
        dict: Dictionary with summary statistics
    """
    if study is None or len(study.trials) == 0:
        print("No valid study or trials available for analysis.")
        return {}
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    # Basic statistics
    stats = {
        "study_name": study.study_name,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "completion_rate": len(completed_trials) / len(study.trials) if study.trials else 0,
    }
    
    print(f"Study: {stats['study_name']}")
    print(f"Total trials: {stats['total_trials']}")
    print(f"Completed trials: {stats['completed_trials']} ({stats['completion_rate']:.1%})")
    print(f"Pruned trials: {stats['pruned_trials']} ({1-stats['completion_rate']:.1%})")
    
    if len(completed_trials) == 0:
        print("No completed trials available for analysis.")
        return stats
    
    # Get trials dataframe and sort by value (descending)
    trials_df = study.trials_dataframe()
    if 'value' in trials_df.columns:
        trials_df = trials_df.sort_values('value', ascending=False)
        display(trials_df.head(5))  # Show top 5 trials
    else:
        display(trials_df.head())
    
    # Analyze values of completed trials
    values = [t.value for t in completed_trials if t.value is not None]
    if values:
        stats["best_value"] = max(values)
        stats["worst_value"] = min(values)
        stats["median_value"] = np.median(values)
        stats["mean_value"] = np.mean(values)
        stats["std_value"] = np.std(values)
        
        print("\nPerformance statistics (objective value):")
        print(f"  Best: {stats['best_value']:.2f}")
        print(f"  Median: {stats['median_value']:.2f}")
        print(f"  Mean: {stats['mean_value']:.2f} ± {stats['std_value']:.2f}")
        print(f"  Worst completed: {stats['worst_value']:.2f}")
    
    # Best trial analysis
    if completed_trials:
        print("\nBest trial:")
        best_trial = study.best_trial
        print(f"  Trial {best_trial.number}: {best_trial.value:.2f}")
        for key, value in best_trial.params.items():
            print(f"     {key}: {value}")
        print()
    
    # Parameter importance analysis
    if len(completed_trials) >= 2:
        print("\nParameter importance analysis:")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f"  {param}: {importance:.4f}")
            
            # Parameter distributions
            print("\nParameter distributions in successful trials:")
            param_names = list(study.best_params.keys())
            for param in param_names:
                values = [t.params[param] for t in completed_trials if param in t.params]
                if not values:
                    continue
                    
                if isinstance(values[0], (int, float)):
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = min(values)
                    max_val = max(values)
                    best_val = study.best_params[param]
                    
                    print(f"  {param}:")
                    print(f"    Range: {min_val} to {max_val}")
                    print(f"    Mean: {mean_val:.4g} ± {std_val:.4g}")
                    print(f"    Best trial: {best_val}")
                else:
                    # For categorical parameters
                    counter = collections.Counter(values)
                    best_val = study.best_params[param]
                    print(f"  {param}:")
                    for val, count in counter.most_common():
                        marker = " (best)" if val == best_val else ""
                        print(f"    {val}: {count} trials{marker}")
        except Exception as e:
            print(f"Could not analyze parameter importance: {e}")
    
    # Visualizations
    if show_plots and len(completed_trials) >= 2:
        # Optimization history
        plt.figure(figsize=(12, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title('Optimization History')
        plt.tight_layout()
        plt.show()
        
        # Parameter importances
        try:
            plt.figure(figsize=(12, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title('Parameter Importances')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
        
        # Parallel coordinate plot for different parameters
        try:
            plt.figure(figsize=(14, 7))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.title('Parallel Coordinate Plot')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot parallel coordinates: {e}")
            
    
    return stats
