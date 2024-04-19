import torch
import SimpleITK as sitk
from RMI import RMILoss
import numpy as np
# Assuming you have your 3D data in 'input_data' and 'target_data'

def read_nifti_to_tensor(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    # Normalize the data if needed
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return torch.from_numpy(data).float()

# Replace 'path_to_file1.nii.gz' and 'path_to_file2.nii.gz' with your actual file paths
file_path1 = './vol_0004_brain.nii.gz'
file_path2 = './vol_0007_brain.nii.gz'

tensor1 = read_nifti_to_tensor(file_path1)
tensor2 = read_nifti_to_tensor(file_path2)
tensor1 = tensor1.unsqueeze(0).unsqueeze(0)
tensor2 = tensor2.unsqueeze(0).unsqueeze(0)
print (tensor1.shape)
# Instantiate RMILoss
rmi_loss = RMILoss(with_logits=True, radius=5, bce_weight=0.5, downsampling_method='max', stride=2, use_log_trace=True, use_double_precision=True)

loss = rmi_loss(tensor2, tensor1)
print(loss)
    

