import torch
import pickle
from scipy.spatial import distance

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def numpy_to_tensor(arr, use_gpu=True, device='cuda:0'):
  if use_gpu:
    return torch.tensor(arr).to(device).float()
  else:
    return torch.tensor(arr).float()

def tensor_to_numpy(tensor):
  return tensor.cpu().detach().numpy()

def pickle_load(f):
  return pickle.load(open(f, 'rb'))

def pickle_dump(obj, f):
  pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)