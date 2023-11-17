import torch
import torch_geometric.data as geom_data

from pre_trained import tu_dataset

torch.manual_seed(42)
tu_dataset.shuffle()
train_dataset = tu_dataset[:150]
test_dataset = tu_dataset[150:]

graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
graph_val_loader = geom_data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)  # Additional loader if you want to change to a larger dataset
graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == "__main__":
    batch = next(iter(graph_test_loader))
    print("Batch:", batch)
    print("Labels:", batch.y[:10])
    print("Batch indices:", batch.batch[:40])
