import torch
from model import *
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)



stage1_weight_path = "/content/netG_epoch_5.pth"
stage2_weight_path = "/content/netG_epoch_5_G2.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Stage 1 Generator
STAGE1_G = STAGE1_G()  # Instantiate the model class

# Load weights from stage1.pth
STAGE1_G.load_state_dict(torch.load(stage1_weight_path, map_location=torch.device('cpu')))

# Set Stage 1 to evaluation mode (optional but recommended)
STAGE1_G.eval()

# Create Stage 2 Generator


STAGE2_G = STAGE2_G(STAGE1_G)  # Pass Stage 1 model to Stage 2

# Load weights from stage2.pth
STAGE2_G.load_state_dict(torch.load(stage2_weight_path, map_location=torch.device('cpu')))

# Set Stage 2 to evaluation mode (optional but recommended)
STAGE2_G.eval()

"""# Generation"""
model = SentenceTransformer('intfloat/e5-large-v2')

"""# Generating picture"""


while True:
    clear_output(wait=True)
    text = input("What do you want to generate? Enter 'exit' to stop: ")
    if text.lower() == "exit":
        break

    text_embeddings = model.encode(text)
    noise = torch.randn(1, 100)
    text_embeddings = torch.tensor(text_embeddings).unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation for stage 1 generator
        stage1_img, stage2_img, _, _ = STAGE2_G(text_embeddings, noise)

    # Detach the tensor and convert it to a NumPy array
    stage1_img_np = stage2_img.detach().cpu().numpy()

    # Squeeze the batch dimension if it's of size 1
    if stage1_img_np.shape[0] == 1:
        stage1_img_np = stage1_img_np.squeeze(0)  # Remove the batch dimension

    # Assuming the image is in a channels-first format (C, H, W),
    # you might need to transpose it to (H, W, C) for Matplotlib
    stage1_img_np = stage1_img_np.transpose(1, 2, 0)

    # Display the image
    plt.imshow(stage1_img_np)
    plt.show()\

    del text

