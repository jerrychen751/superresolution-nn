import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # save to file without needing a display
import matplotlib.pyplot as plt
from pathlib import Path
from models import CAE
from data_utils import generate_laminar_channel

latent_dim = 64
lr = 0.001
epochs = 10000
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
data = generate_laminar_channel(n_samples = 500, size = 64).to(device)

# loss_history stores the loss value at every single epoch.
# We append to it inside the loop so we have a complete record
# to plot once training finishes.
loss_history = []

print("Beginning of training for laminar flow with Autoencoder")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)

    loss.backward()
    optimizer.step()

    # .item() converts the loss from a PyTorch tensor to a plain
    # Python float so it can be stored in a regular list.
    loss_history.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "cae_laminar.pth")
print("End of training. Model saved as cae_laminar.pth")


# ==============================================================================
# LOSS CURVE
# ==============================================================================
# loss_history is now a list of 500 floats — one per epoch.
# We plot epoch number on the x axis and loss value on the y axis.
# A healthy training run shows the loss falling steeply at first,
# then levelling off as the model converges.

def plot_loss_curve(loss_history, save_path='ae_loss_curve.png'):
    """
    Plots the autoencoder reconstruction loss as a function of epochs
    and saves it as a PNG file.

    Args:
        loss_history (list of float): loss value recorded at every epoch
        save_path (str): filename to save the plot to
    """
    epochs_axis = list(range(1, len(loss_history) + 1))
    # range(1, 501) gives [1, 2, 3, ..., 500] — human-friendly epoch numbers
    # starting at 1 rather than 0.

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs_axis, loss_history,
            color='steelblue', linewidth=1.5, label='Reconstruction loss (MSE)')
    # steelblue is a clean, readable colour for a single line.

    # Mark the final loss value at the end of the curve so you can
    # read the converged value without hovering over the plot.
    ax.annotate(
        f"Final: {loss_history[-1]:.6f}",
        xy=(epochs_axis[-1], loss_history[-1]),
        xytext=(-80, 15),
        textcoords='offset points',
        fontsize=9,
        color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1)
    )
    # loss_history[-1] is the last item in the list — the final epoch's loss.
    # xytext=(-80, 15) shifts the label 80 pixels left and 15 pixels up
    # from the annotated point so it doesn't overlap the line.

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss  (lower = better)', fontsize=11)
    ax.set_title('Autoencoder Training Loss Curve\n'
                 'Reconstruction error vs. number of training epochs',
                 fontsize=12)

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Use a log scale on the y axis if the loss drops by more than
    # two orders of magnitude — makes the early rapid drop visible
    # at the same time as the fine detail of late-stage convergence.
    if max(loss_history) / (min(loss_history) + 1e-12) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('MSE Loss  (log scale, lower = better)', fontsize=11)

    plt.tight_layout()
    plt.savefig(Path(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")


plot_loss_curve(loss_history, save_path=Path(__file__).parent/'ae_loss_curve.png')