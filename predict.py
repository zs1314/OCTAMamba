import cv2
import torch
import os
import numpy as np
from modelszoo.Umamba import *
from modelszoo.AC_Mamba import *
from modelszoo.VM_UNet import *
from modelszoo.Swin_UNet import *
from modelszoo.MISSFormer import *
from modelszoo.H2Former import *
from modelszoo.VM_UNet2 import *
from modelszoo.R2UNet import *
from modelszoo.H_vmunet import *
from our_model.OCTAMamba import *
from modelszoo.unetpp import *

# load model
model = OCTAMamba().to('cuda')
model.load_state_dict(torch.load('/root/OCTA-FRNet-main/result/OCTAMamba/ROSSA/model_best.pth'))
model.eval()  # Setting up the model for evaluation mode

# Enter the image folder path
image_folder = '/root/OCTA-FRNet-main/dataset/ROSSA/test/image/'
# Output mask folder path
output_folder = '/root/OCTA-FRNet-main/output/ROSSA/OCTAMamba_v6_output_masks/'
os.makedirs(output_folder, exist_ok=True)

# Get a list of all image files in a folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# Iterate over all image files and process
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32")
    image /= 255  # normalize

    # Resize to 224x224
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    image = image.reshape((1, 1, image.shape[0], image.shape[1]))

    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).to('cuda' if torch.cuda.is_available() else 'cpu')

    # predict
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, list):
            output = output[0]  # If the model returns multiple outputs, select the first one
        output = output.squeeze().cpu().numpy()

    # Convert the output to a masked image
    output = (output > 0.5).astype(np.uint8)  # Convert probabilities to binary masks based on thresholds
    output = output * 255  # Extend the mask value to the range 0-255

    # Save the mask image with the same name as the original image name
    mask_save_path = os.path.join(output_folder, image_file)
    cv2.imwrite(mask_save_path, output)
    print(f'Mask saved to {mask_save_path}')
