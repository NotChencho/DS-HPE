"""
Visualization script for DenseNN architecture with specific hyperparameters.
Uses torchviz for detailed network visualization.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_models import DenseNN

# Hyperparameters from the image
HYPERPARAMETERS = {
    "batch_size": 1028,
    "dropout": 0.15559998021833732,
    "epochs": 150,
    "hidden_dims": [512, 256, 128],
    "lr": 0.0009719478103336032,
    "patience": 80,
    "weight_decay": 0.00001,
}

def visualize_network_architecture(model, hyperparameters, save_path="densenn_architecture"):
    """
    Create a visual representation of the DenseNN architecture using torchviz.
    """
    # Create sample input
    sample_input = torch.randn(1, 75)
    
    # Use torchviz to visualize
    print("Generating network visualization with torchviz...")
    try:
        from torchviz import make_graph
        
        # Generate the graph
        y = model(sample_input)
        graph = make_graph(
            y, 
            params=dict(model.named_parameters()),
            title="DenseNN Architecture"
        )
        
        # Save as PDF and PNG
        graph.render(save_path, format='pdf', cleanup=True)
        graph.render(save_path, format='png', cleanup=True)
        print(f"Architecture visualization saved to: {save_path}.pdf and {save_path}.png")
        
    except ImportError:
        print("torchviz not installed. Using matplotlib fallback...")
        visualize_network_alternative(model, hyperparameters, f"{save_path}.png")
    except Exception as e:
        print(f"Error with torchviz: {e}")
        print("Falling back to matplotlib visualization...")
        visualize_network_alternative(model, hyperparameters, f"{save_path}.png")


def visualize_network_alternative(model, hyperparameters, save_path="densenn_architecture.png"):
    """
    Fallback visualization using torchviz if torchlens is not available.
    """
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Model configuration
    input_dim = 75
    hidden_dims = hyperparameters["hidden_dims"]
    output_dim = 3
    
    # Layer dimensions
    all_dims = [input_dim] + hidden_dims + [output_dim]
    num_layers = len(all_dims)
    
    # Visualization parameters
    layer_spacing = 2.5
    node_radius = 0.15
    
    # Colors for different layer types
    input_color = "#FF6B6B"
    hidden_color = "#4ECDC4"
    output_color = "#FFE66D"
    
    # Draw layers
    for layer_idx, num_nodes in enumerate(all_dims):
        x_pos = layer_idx * layer_spacing
        
        # Determine layer color
        if layer_idx == 0:
            color = input_color
            layer_label = f"Input\n({num_nodes})"
        elif layer_idx == num_layers - 1:
            color = output_color
            layer_label = f"Output\n({num_nodes})"
        else:
            color = hidden_color
            layer_label = f"Hidden {layer_idx}\n({num_nodes})"
        
        # Draw nodes
        if num_nodes <= 20:
            node_spacing = 0.3
            y_start = -(num_nodes - 1) * node_spacing / 2
            node_positions = [(x_pos, y_start + i * node_spacing) for i in range(num_nodes)]
        else:
            num_display = min(10, num_nodes)
            node_spacing = 0.3
            y_start = -(num_display - 1) * node_spacing / 2
            node_positions = [(x_pos, y_start + i * node_spacing) for i in range(num_display)]
            ax.text(x_pos, 0, "...", fontsize=14, ha="center", va="center", weight="bold")
        
        # Draw node circles
        for x, y in node_positions:
            circle = plt.Circle((x, y), node_radius, color=color, ec="black", linewidth=2, zorder=2)
            ax.add_patch(circle)
        
        # Add layer label
        if num_nodes > 20:
            y_label = y_start - 0.8
        else:
            y_label = (y_start - num_nodes * node_spacing / 2) - 0.6
        ax.text(x_pos, y_label, layer_label, fontsize=10, ha="center", weight="bold")
    
    # Add hyperparameter information
    info_text = f"""DenseNN Hyperparameters:

Batch Size: {hyperparameters['batch_size']}
Dropout: {hyperparameters['dropout']:.4f}
Epochs: {hyperparameters['epochs']}
Learning Rate: {hyperparameters['lr']:.6f}
Weight Decay: {hyperparameters['weight_decay']:.6f}
Patience: {hyperparameters['patience']}

Architecture:
Input: {input_dim}
Hidden: {' → '.join(map(str, hidden_dims))}
Output: {output_dim}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary_text = f"Total Parameters: {total_params:,}\nTrainable Parameters: {trainable_params:,}"
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')
    
    ax.set_xlim(-1, (num_layers - 1) * layer_spacing + 1)
    ax.set_ylim(-3, 2.5)
    ax.axis("off")
    ax.set_aspect("equal")
    
    plt.title("DenseNN Architecture Visualization", fontsize=16, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Fallback visualization saved to: {save_path}")
    plt.show()


def export_to_onnx(model, save_path="densenn_model.onnx"):
    """
    Export the DenseNN model to ONNX format for visualization on netron.app.
    """
    print("Exporting model to ONNX format...")
    try:
        # Check if onnx is installed
        try:
            import onnx
        except ImportError:
            print("ONNX module not installed.")
            print("\nTo install ONNX, run:")
            print("  pip install onnx")
            return
        
        # Create sample input
        sample_input = torch.randn(1, 75)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            save_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        
        print(f"\nONNX model exported successfully to: {save_path}")
        print(f"\nTo visualize on netron.app:")
        print(f"1. Go to https://netron.app/")
        print(f"2. Click 'Open Model' and select '{save_path}'")
        print(f"3. Or drag and drop '{save_path}' into the page")
        
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        print("\nTroubleshooting:")
        print("1. Install ONNX: pip install onnx")
        print("2. Ensure PyTorch is properly installed")

def print_model_summary(model):
    """
    Print a detailed summary of the model architecture.
    """
    print("\n" + "="*60)
    print("DenseNN Model Summary")
    print("="*60)
    print(model)
    print("\n" + "-"*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("="*60 + "\n")


def main():
    # Create model with hyperparameters
    model = DenseNN(
        input_dim=75,
        hidden_dims=HYPERPARAMETERS["hidden_dims"],
        output_dim=3,
        dropout=HYPERPARAMETERS["dropout"]
    )
    
    # Print summary
    # print_model_summary(model)
    
    # Visualize architecture with torchviz (or fallback to matplotlib)
    visualize_network_architecture(model, HYPERPARAMETERS)
    
    # Export to ONNX for netron.app visualization
    export_to_onnx(model)


if __name__ == "__main__":
    main()
