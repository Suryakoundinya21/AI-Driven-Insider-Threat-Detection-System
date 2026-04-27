import torch
import torch.nn as nn
import numpy as np
import logging
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.config import config

logger = logging.getLogger(__name__)


def train_autoencoder(
    model,
    X_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 8,
    val_split: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = model.to(device)

    X_tensor = torch.FloatTensor(X_train)

    dataset    = TensorDataset(X_tensor, X_tensor)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    patience_count = 0
    history        = {"train_loss": [], "val_loss": []}

    logger.info(f"Training : {train_size:,} | Validation: {val_size:,}")
    logger.info(f"Epochs={epochs} | Batch={batch_size} | LR={lr} | Patience={patience}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss   = criterion(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                output = model(xb)
                loss   = criterion(output, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Patience: {patience_count}/{patience}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            save_path = config.MODELS_DIR / "autoencoder" / "best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    hist_path = config.MODELS_DIR / "autoencoder" / "training_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return model, history
