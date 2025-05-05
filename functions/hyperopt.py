       
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





def objective_kfold(trial, k=3, verbose=True, first_fold_min_acc=90.0):
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
    patience = trial.suggest_int("patience", 3, 4)
    max_epochs = trial.suggest_int("max_epochs", 5, 10)
    
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
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)
            
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
                    
                    if len(completed_trials) >= 5:  # Need at least 5 completed trials for meaningful comparison
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
 




def run_kfold_optuna_optimization(n_trials=5, k=5, verbose=True, study_name="animals10_kfold", storage=None, load_if_exists=True, first_fold_min_acc=90.0):
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
            n_startup_trials=5,  # Number of trials to run before pruning starts
            n_warmup_steps=5,    # Number of epochs to run before pruning can happen in a trial
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
        return objective_kfold(trial, k=k, verbose=verbose, first_fold_min_acc=first_fold_min_acc)
    
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
