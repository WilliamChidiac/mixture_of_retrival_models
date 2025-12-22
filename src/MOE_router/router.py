import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Tuple, List

class MOERouter(nn.Module):
    """
    Neural Router for Mixture of Experts.
    Takes a query embedding and outputs weights for each expert.
    """
    def __init__(self, hidden_dim: List[int] = [128, 64], embedding_model: str = 'all-mpnet-base-v2', norm: str = "L1", dropout: float = 0.3):
        """
        Initialize the MOERouter.

        Args:
            hidden_dim (int): Dimension of the hidden layer. Defaults to [128, 64].
            embedding_model (str): Sentence Transformer model name. Defaults to 'all-mpnet-base-v2'.
            norm (str): Normalization method for output weights. Defaults to "L1".
            dropout (float): Dropout rate. Defaults to 0.3.
        """
        super(MOERouter, self).__init__()
        self.embedding_model = SentenceTransformer(embedding_model)
        input_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.fcs = nn.ModuleList()
        for h_dim in hidden_dim:
            self.fcs.append(nn.Linear(input_dim, h_dim))
            self.fcs.append(nn.BatchNorm1d(h_dim))
            self.fcs.append(nn.ReLU())
            self.fcs.append(nn.Dropout(dropout))
            input_dim = h_dim
        self.fcs.append(nn.Linear(input_dim, 2))  # Assuming two experts: sparse and dense
        self.norm = norm.lower()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the router.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output weights of shape (batch_size, 2).
        """
        # x: (batch_size, input_dim)
        for layer in self.fcs:
            x = layer(x)
        return x  # Raw scores; normalization will be applied later 
    def save_model(self, path: Path) -> None:
        """
        Save the model state to the specified path.

        Args:
            path (Path): Directory path to save the model.
        """
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / f"moe_router_{self.norm}.pth")

    def get_weights(self, queries: list[str]) -> np.ndarray:
        """
        Infer expert weights for the test set.

        Args:
            queries (list[str]): List of query strings.

        Returns:
            np.ndarray: Array of shape (num_samples, 2) with expert weights.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        all_weights = []
        query_embeddings = self.embedding_model.encode(queries, convert_to_tensor=True).cpu().numpy()
        new_test_df = pd.DataFrame()
        new_test_df['query_embedding'] = list(query_embeddings)

        with torch.no_grad():
            embeddings = torch.tensor(np.stack(new_test_df['query_embedding'].values)).to(device)
            outputs = self(embeddings)
            all_weights.append(outputs.cpu().numpy())
        all_weights = np.vstack(all_weights)

        # Always apply Softmax to logits to get valid probabilities.
        # L1 normalization on logits (which can be negative) is incorrect.
        exp_weights = np.exp(all_weights - np.max(all_weights, axis=1, keepdims=True))
        normalized_weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        return normalized_weights
    
    def load_model(self, path: Path) -> None:
        """
        Load the model state from the specified path.

        Args:
            path (Path): Directory path to load the model from.
        """
        self.load_state_dict(torch.load(path / f"moe_router_{self.norm}.pth"))
        
def balance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Balances the DataFrame by undersampling the majority class based on which expert has a higher score.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'sparse' and 'dense' score columns.

    Returns:
        pd.DataFrame: Balanced DataFrame.
    """
    sparse_better = df[df['sparse'] > df['dense']]
    dense_better = df[df['dense'] > df['sparse']]

    min_count = min(len(sparse_better), len(dense_better))
    
    if min_count == 0:
        # If one class is empty, return the original df or handle gracefully
        return df

    sparse_sampled = sparse_better.sample(n=min_count, random_state=42)
    dense_sampled = dense_better.sample(n=min_count, random_state=42)

    balanced_df = pd.concat([sparse_sampled, dense_sampled]).reset_index(drop=True)
    return balanced_df

def train(model: MOERouter, train_df: pd.DataFrame, batch_size: int = 32, lr: float = 1e-3, epochs: int = 100, score_norm: str = "l1" ) -> MOERouter:
    """
    Train the MOE Router.
    Args:
        model (MOERouter): The MOERouter model to train.
        train_df (pd.DataFrame): DataFrame with columns 'query_embedding' and 'label'.
        lr (float): Learning rate. Defaults to 1e-3.
        epochs (int): Number of training epochs. Defaults to 100.
        score_norm (str): Score normalization method used. Defaults to 'l1'.
    Returns:
        MOERouter: The trained MOERouter model.
    """
    labels_name = [ f'{s}_{score_norm}' for s in ['sparse', 'dense']]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
            
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    query_text = train_df['query'].tolist()
    print("Generating query embeddings...")
    query_embeddings = model.embedding_model.encode(query_text, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
    
    full_train_df = pd.DataFrame()
    full_train_df['label'] = list(zip(*(train_df[labels_name].values.T)))
    full_train_df['query_embedding'] = list(query_embeddings)
    full_train_df['sparse'] = train_df['sparse']
    full_train_df['dense'] = train_df['dense']
    
    # Split into train and validation (90/10)
    val_size = int(0.1 * len(full_train_df))
    train_size = len(full_train_df) - val_size
    
    # Shuffle before splitting
    full_train_df = full_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    val_df = full_train_df.iloc[:val_size]
    train_data_df = full_train_df.iloc[val_size:]
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 15
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        # Use all training data without balancing
        epoch_df = train_data_df.sample(frac=1).reset_index(drop=True)
        
        model.train()
        for i in tqdm(range(0, len(epoch_df), batch_size), desc=f"Epoch {epoch+1}"):
            batch_df = epoch_df.iloc[i:i+batch_size]
            
            # Skip batches with only 1 sample to avoid BatchNorm error
            if len(batch_df) <= 1:
                continue
                
            batch_embeddings = torch.tensor(np.stack(batch_df['query_embedding'].values)).to(device)
            batch_labels = torch.tensor(list(batch_df['label'].values)).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / (len(epoch_df) / batch_size)
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch_df = val_df.iloc[i:i+batch_size]
                batch_embeddings = torch.tensor(np.stack(batch_df['query_embedding'].values)).to(device)
                batch_labels = torch.tensor(list(batch_df['label'].values)).to(device)
                
                outputs = model(batch_embeddings)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, batch_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(val_df) / batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model
        

def test_model(model: MOERouter, test_df: pd.DataFrame, score_norm: str = 'l1', batch_size: int = 32) -> Tuple[pd.DataFrame, float]:
    """
    Test the MOE Router model.

    Args:
        model (MOERouter): The trained MOERouter model.
        test_df (pd.DataFrame): DataFrame with columns 'query_embedding' and 'label'.
        score_norm (str): Score normalization method used. Defaults to 'l1'.
        batch_size (int): Batch size for testing. Defaults to 32.

    Returns:
        pd.DataFrame: DataFrame with test results including true weights and predicted weights.
        float: Accuracy of the router on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    labels_name = [ f'{s}_{score_norm}' for s in ['sparse', 'dense']]
    new_test_df = pd.DataFrame()
    new_test_df['query'] = test_df['query']
    new_test_df['label'] = list(zip(*(test_df[labels_name].values.T)))
    
    correct = 0
    total = 0
    res_file = {'query': [], 'true_labels': [], 'predicted_weights': []}
    with torch.no_grad():
        for i in tqdm(range(0, len(new_test_df), batch_size), desc="Testing"):
            batch_df = new_test_df.iloc[i:i+batch_size]
            batch_labels = torch.tensor(list(batch_df['label'].values)).to(device)
            batch_query = batch_df['query'].tolist()
            predicted_weights = model.get_weights(batch_query)
            test_labels = batch_labels.cpu().numpy()
            res_file['query'].extend(batch_query)
            res_file['true_labels'].extend(test_labels.tolist())
            res_file['predicted_weights'].extend(predicted_weights.tolist())
            predicted_classes = np.argmax(predicted_weights, axis=1)
            true_classes = np.argmax(test_labels, axis=1)
            
            # Identify cases where experts have equal scores (e.g. 0.5, 0.5)
            is_equal = np.isclose(test_labels[:, 0], test_labels[:, 1])
            
            # Filter out equal cases for accuracy calculation
            valid_indices = ~is_equal
            
            if valid_indices.sum() > 0:
                correct += (predicted_classes[valid_indices] == true_classes[valid_indices]).sum()
                total += valid_indices.sum()
                
    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f}")
    return pd.DataFrame(res_file), accuracy
    
    