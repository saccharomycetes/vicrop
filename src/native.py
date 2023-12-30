from PIL import Image
import numpy as np
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import median_filter
from scipy.ndimage import label
from skimage.measure import block_reduce
import torch

class native_crop_tools:
    
    def __init__(self, device='cpu', blip_model_id='pretrain_flant5xl'):

        self.blip_model_id = blip_model_id
        self.device = device

        if self.blip_model_id.find('t5') != -1:
            self.model_name = 'blip2_t5'
        elif self.blip_model_id.find('opt') != -1:
            self.model_name = 'blip2_opt'
        else:
            raise ValueError('invalid blip model id')
        self.model, self.vis_processors, _ = load_model_and_preprocess(name=self.model_name, model_type=self.blip_model_id, is_eval=True, device=self.device)
    

    def high_pass_filter(self, processed_image, km, kh):

        l = TF.gaussian_blur(processed_image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
        h = processed_image.squeeze().detach().cpu().numpy() - l
        h_brightness = np.sqrt(np.square(h).sum(axis=0))
        h_brightness = median_filter(h_brightness, size=km)
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

        return h_brightness

    def largest_connected_component_bounding_box(self, weight_map, binary_map):

        structure = np.ones((3, 3), dtype=np.int32)
        labeled_array, num_features = label(binary_map, structure=structure)
        component_sizes = np.bincount(labeled_array.flat, weights=weight_map.flat)
        if len(component_sizes) == 1:
            return 0, 0, 16, 16
        largest_component = component_sizes[1:].argmax() + 1
        y, x = np.where(labeled_array == largest_component)

        y_min, y_max = y.min(), y.max() + 1
        x_min, x_max = x.min(), x.max() + 1
        
        return x_min, y_min, x_max, y_max

    def att_crop(self, image_path, question, k=30, km=7, kh=3, use_high_pass=True, return_maps=False):

        raw_image = Image.open(image_path).convert('RGB')
        patch_sizes = (raw_image.size[0]//16, raw_image.size[1]//16)
        processed_image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        sample = {'image': processed_image, 'prompt' : f'Question:{question} Short answer:'}

        q_former_att, q_fromer_att_grad, lm_att, lm_att_grad = self.model.att(sample)
        filtered_q_former_att = F.relu(torch.stack(q_fromer_att_grad)) * torch.stack(q_former_att)
        filtered_lm_att = F.relu(torch.stack(lm_att_grad)) * torch.stack(lm_att)

        filtered_q_former_att = filtered_q_former_att.squeeze().detach().cpu().numpy().sum(axis=(0,1))[:,1:].transpose(1, 0)
        filtered_lm_att = filtered_lm_att.to(torch.float32).squeeze().detach().cpu().numpy().sum(axis=(0,1))

        att = np.matmul(filtered_q_former_att, filtered_lm_att).reshape(16, 16)
        h_pass_map = self.high_pass_filter(processed_image, km, kh)

        if use_high_pass:
            imp_map = att * h_pass_map
        else:
            imp_map = att

        binary_map = (imp_map >= np.partition(imp_map.flatten(), -k)[-k])

        bbox = self.largest_connected_component_bounding_box(imp_map, binary_map)
        bbox = [bbox[0]*patch_sizes[0], bbox[1]*patch_sizes[1], bbox[2]*patch_sizes[0], bbox[3]*patch_sizes[1]]
        
        if return_maps:
            return bbox, att, h_pass_map, imp_map
        
        return bbox
    
    
    def grad_crop(self, image_path, question, k=30, km=7, kh=3, use_high_pass=True, return_maps=False):

        raw_image = Image.open(image_path).convert('RGB')
        patch_sizes = (raw_image.size[0]//16, raw_image.size[1]//16)
        processed_image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        sample = {'image': processed_image, 'prompt' : f'Question:{question} Short answer:'}

        grad = self.model.grad(sample)
        grad = np.linalg.norm(grad.squeeze().detach().cpu().numpy(), ord=2, axis=0)
        grad= block_reduce(grad, block_size=(14, 14), func=np.sum)

        h_pass_map = self.high_pass_filter(processed_image, km, kh)
        
        if use_high_pass:
            imp_map = grad * h_pass_map
        else:
            imp_map = grad

        binary_map = (imp_map >= np.partition(imp_map.flatten(), -k)[-k])

        bbox = self.largest_connected_component_bounding_box(imp_map, binary_map)
        bbox = [bbox[0]*patch_sizes[0], bbox[1]*patch_sizes[1], bbox[2]*patch_sizes[0], bbox[3]*patch_sizes[1]]

        if return_maps:
            return bbox, grad, h_pass_map, imp_map
        
        return bbox