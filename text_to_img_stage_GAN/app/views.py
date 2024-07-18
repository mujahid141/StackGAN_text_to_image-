import os
import torch
from django.shortcuts import render
from django.http import HttpResponse
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from text_to_img_stage_GAN import settings
from .model import STAGE1_G, STAGE2_G  # Adjust import based on your project structure
from django.views.decorators.csrf import csrf_exempt

# Load SentenceTransformer model
sentence_model = SentenceTransformer('intfloat/e5-large-v2')

# Load the models once globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stage1_weight_path = os.path.join(settings.BASE_DIR, 'app', 'netG_epoch_5.pth')
stage2_weight_path = os.path.join(settings.BASE_DIR, 'app', 'netG_epoch_5_G2.pth')

# Create and load Stage 1 Generator
STAGE1_G_model = STAGE1_G().to(device)
STAGE1_G_model.load_state_dict(torch.load(stage1_weight_path, map_location=device))
STAGE1_G_model.eval()

# Create and load Stage 2 Generator
STAGE2_G_model = STAGE2_G(STAGE1_G_model).to(device)
STAGE2_G_model.load_state_dict(torch.load(stage2_weight_path, map_location=device))
STAGE2_G_model.eval()

@csrf_exempt
def generate_image(request):
    if request.method == 'POST':
        text = request.POST['text']
        text_embeddings = sentence_model.encode(text)
        noise = torch.randn(1, 100).to(device)
        text_embeddings = torch.tensor(text_embeddings).unsqueeze(0).to(device)

        with torch.no_grad():
            _, stage2_img, _, _ = STAGE2_G_model(text_embeddings, noise)

        stage2_img_np = stage2_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

        # Normalize the image if needed (e.g., values in [0, 1] or [0, 255])
        stage2_img_np = (stage2_img_np - stage2_img_np.min()) / (stage2_img_np.max() - stage2_img_np.min())

        # Convert the image to a format suitable for displaying in HTML
        fig, ax = plt.subplots()
        ax.imshow(stage2_img_np)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        plt.close()  # Close the matplotlib plot to free memory

        return render(request, 'app/result.html', {'image_base64': image_base64})

    return render(request, 'app/generate.html')
