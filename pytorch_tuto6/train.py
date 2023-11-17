import pytorch_lightning as pl
from utils import CHECKPOINT_PATH, device
import os
from novel_level_gnn_model import NodeLevelGNN
import torch_geometric.data as geom_data
from pytorch_lightning.callbacks import ModelCheckpoint
from pre_trained import cora_dataset
import torch

pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")
        ],
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=200,
        enable_progress_bar=False,
    )  # False because epoch size is 1
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(
            model_name=model_name,
            c_in=dataset.num_node_features,
            c_out=dataset.num_classes,
            **model_kwargs,
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Test best model on the test set
    test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
    return model, result



# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")


if __name__ == "__main__":
    node_mlp_model, node_mlp_result = train_node_classifier(
        model_name="MLP", dataset=cora_dataset, c_hidden=16, num_layers=2, dp_rate=0.1
    )

    print_results(node_mlp_result)

    print("#"*30)
    node_gnn_model, node_gnn_result = train_node_classifier(
        model_name="GNN",
        layer_name="GCN",
        dataset=cora_dataset,
        c_hidden=16,
        num_layers=2,
        dp_rate=0.1,
    )
    print_results(node_gnn_result)

