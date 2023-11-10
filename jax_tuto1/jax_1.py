import flax.linen as nn
from matplotlib.colors import to_rgba
from flax.training import train_state, checkpoints
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from torch import seed
import torch.utils.data as data
from torch.utils.data import dataloader
from tqdm import tqdm


class SimpleClassifier(nn.Module):
    num_hidden: int
    num_output: int

    def setup(self) -> None:
        self.linear1 = nn.Dense(self.num_hidden)
        self.liear2 = nn.Dense(self.num_output)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.tanh(x)
        y = self.liear2(x)
        return y


class SimpleClassifierCompact(nn.Module):
    num_hidden: int
    num_output: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        y = nn.Dense(features=self.num_output)(x)
        return y


class XORDataset(data.Dataset):
    def __init__(self, size, seed, std=0.1) -> None:
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(
            np.float32
        )
        label = (data.sum(axis=1) == 1).astype(np.int32)

        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc

@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss_acc, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch)

    state = state.apply_gradients(grads=grads)

    return state, loss, acc


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

def train_model(state, data_loader, num_epoch=100):
    for _ in tqdm(range(num_epoch)):
        for batch in data_loader:
            state, _, _ = train_step(state, batch)
    return state


# Training
model = SimpleClassifier(8, 1)
rng = jax.random.PRNGKey(42)
# params = model.init(rng, jnp.empty((8, 2)))
rng, inp_rng, init_rng = jax.random.split(rng, 3)
inp = jax.random.normal(inp_rng, (8, 2))  # Batch size 8, input size 2
# Initialize the model
params = model.init(init_rng, inp)

# dataset = XORDataset(size=200, seed=42)
# #
# # visualize_samples(dataset.data, dataset.label)
#
# data_loader = data.DataLoader(
#     dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate
# )
#
# data_inputs, data_labels = next(iter(data_loader))

# print("Data inputs", data_inputs.shape, "\n", data_inputs)
# print("Data labels", data_labels.shape, "\n", data_labels)

optimizer = optax.sgd(learning_rate=0.1)
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer
)

train_dataset = XORDataset(size=2500, seed=42)
train_data_loader = data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate
)

train_model_state = train_model(model_state, train_data_loader, num_epoch=100)

checkpoints.save_checkpoint(
    ckpt_dir="./my_checkpoints/",
    target=train_model_state,
    step=100,
    prefix="my_model",
    overwrite=True,
)


# Eval
def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

test_dataset = XORDataset(size=500, seed=123)

test_dataloader = data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    collate_fn=numpy_collate,
)
eval_model(train_model_state, test_dataloader)


def visualize_classification(model, data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(4, 4), dpi=500)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    # Let's make use of a lot of operations we have learned above
    c0 = np.array(to_rgba("C0"))
    c1 = np.array(to_rgba("C1"))
    x1 = jnp.arange(-0.5, 1.5, step=0.01)
    x2 = jnp.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing="ij")  # Meshgrid function as in numpy
    model_inputs = np.stack([xx1, xx2], axis=-1)
    logits = model(model_inputs)
    preds = nn.sigmoid(logits)
    output_image = (1 - preds) * c0[None, None] + preds * c1[
        None, None
    ]  # Specifying "None" in a dimension creates a new one
    # output_image = jax.device_get(
    #     output_image
    # )  # Convert to numpy array.
    # This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig


trained_model = model.bind(train_model_state.params)
dataset = XORDataset(size=200, seed=42)
_ = visualize_classification(trained_model, dataset.data, dataset.label)
plt.show()
