import os

import cv2
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

loss_fn = torch.nn.L1Loss()


class Controller:
    def __init__(self, self_layers=(0, 16)):
        self.num_self_layers = -1
        self.cur_self_layer = 0
        self.self_layers = list(range(*self_layers))

    def step(self):
        self.cur_self_layer = 0


class UnetDataCache:
    def __init__(self):
        self.up_features = {}

    def clear(self):
        self.up_features.clear()

    def add_up_feature(self, block_idx, feature):
        self.up_features[block_idx] = feature

    def get_features(self):
        return self.up_features[1], self.up_features[2]


class AttnDataCache:
    def __init__(self):
        self.q = []
        self.k = []
        self.v = []
        self.out = []

    def clear(self):
        self.q.clear()
        self.k.clear()
        self.v.clear()
        self.out.clear()

    def add(self, q, k, v, out):
        self.q.append(q)
        self.k.append(k)
        self.v.append(v)
        self.out.append(out)

    def get(self):
        return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()


def load_image(image_path, size=None, mode="RGB"):
    img = Image.open(image_path).convert(mode)
    if size is None:
        width, height = img.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)
    img = img.resize(size, Image.Resampling.BICUBIC)
    return ToTensor()(img).unsqueeze(0)


def register_attn_control(unet, controller, cache=None):
    def attn_forward(self):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
        ):
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            q = self.to_q(hidden_states)
            is_self = encoder_hidden_states is None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            inner_dim = k.shape[-1]
            head_dim = inner_dim // self.heads

            q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            if is_self and controller.cur_self_layer in controller.self_layers:
                cache.add(q, k, v, hidden_states)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            hidden_states = hidden_states.to(q.dtype)

            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            if is_self:
                controller.cur_self_layer += 1

            return hidden_states

        return forward

    def modify_forward(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":
                net.forward = attn_forward(net)
                return count + 1
            elif hasattr(net, "children"):
                count = modify_forward(subnet, count)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        cross_att_count += modify_forward(net, 0)
    controller.num_self_layers = cross_att_count // 2


def register_unet_feature_extraction(unet, cache):
    hooks = []
    target_up_blocks = [1, 2]

    def up_block_hook(block_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                feature = output[0]
            else:
                feature = output
            cache.add_up_feature(block_idx, feature)
        return hook

    for i, block in enumerate(unet.up_blocks):
        if i in target_up_blocks:
            hooks.append(block.register_forward_hook(up_block_hook(i)))
    return hooks

def visualize_clustering(features, cluster_mask, original_image, image_name=None):
    device = features.device  
    if torch.is_tensor(original_image):
        x = original_image
        if x.dim() == 4:
            x = x.squeeze(0).permute(1, 2, 0)
        x = x.detach()
        if x.max() <= 1:
            x = x * 255.0
        original_image = x.to(device=device, dtype=torch.float32)
        ori_np_for_plot = original_image.detach().cpu().numpy().astype(np.uint8)
    else:
        ori_np_for_plot = original_image
        if ori_np_for_plot.max() <= 1.0:
            ori_np_for_plot = (ori_np_for_plot * 255).astype(np.uint8)
        original_image = torch.from_numpy(ori_np_for_plot).to(device=device, dtype=torch.float32)

    H, W, _ = original_image.shape
    cm = cluster_mask
    if not torch.is_tensor(cm):
        cm = torch.from_numpy(cm)
    cm = cm.to(device=device)
    if cm.dim() == 2:
        cm = cm.unsqueeze(0).unsqueeze(0)
    elif cm.dim() == 3:
        cm = cm.unsqueeze(0)
    cm_resized = F.interpolate(cm.float(), size=(H, W), mode='nearest').squeeze().to(torch.long)

    unique_clusters = torch.unique(cm_resized, sorted=True)
    K = unique_clusters.numel()
    color_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#ff00ff',
        '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#f5b041', '#e67e22',
        '#4B0082', '#a3a500', '#8c6d31', '#ff7f50', '#ff4500', '#6a5acd', '#b22222', '#ff1493', '#00ced1', '#ffd700',
        '#32cd32', '#6b8e23', '#7fff00', '#ff6347', '#ffa500', '#4682b4', '#5f9ea0', '#d2691e', '#20b2aa', '#dc143c',
        '#add8e6', '#ffdab9',
    ]
    palette = torch.tensor([mcolors.to_rgb(c) for c in color_list[:K]],
                           device=device, dtype=torch.float32)

    max_id = int(cm_resized.max().item())
    id2idx = torch.full((max_id + 1,), -1, device=device, dtype=torch.long)
    for i, uid in enumerate(unique_clusters):
        id2idx[int(uid.item())] = i
    cm_idx = id2idx[cm_resized]
    colored_mask = palette[cm_idx]

    alpha = 0.85
    overlay = (1 - alpha) * original_image + alpha * (colored_mask * 255.0)
    overlay = overlay.clamp(0, 255).to(torch.uint8)
    overlay_image = overlay.detach().cpu().numpy()

    plt.figure(figsize=(25, 5))
    
    plt.subplot(141)
    plt.title('Original Feature Map')
    feat0 = features[0, 0].detach().cpu().numpy()
    im0 = plt.imshow(feat0, cmap='viridis')
    plt.colorbar(im0)

    plt.subplot(142)
    plt.title('Original Image')
    plt.imshow(ori_np_for_plot)
    plt.axis('off')

    plt.subplot(143)
    plt.title('Clustering Mask')
    custom_cmap = mcolors.ListedColormap(color_list[:K])
    
    plt.imshow(cm_idx.detach().cpu().numpy(), cmap=custom_cmap, vmin=0, vmax=K-1)
    plt.axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i]) for i in range(K)]
    labels = [str(int(x.item())) for x in unique_clusters]
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
               title='Clusters', frameon=True)

    plt.subplot(144)
    plt.title('Clustered Image Overlay')
    plt.imshow(overlay_image)
    plt.axis('off')

    plt.tight_layout()
    if image_name:
        os.makedirs(f"outputs/{image_name}", exist_ok=True)
        plt.savefig(f"outputs/{image_name}/cluster_visualize.png", bbox_inches='tight', dpi=300)
    plt.close()

    return overlay_image


def convert_mask_to_array(mask_path, input_size=(512, 512), output_size=(64, 64), device=torch.device('cpu')):
    mask = cv2.imread(mask_path)
    if mask is None:
        raise ValueError(f"Unable to read image: {mask_path}")
    if mask.shape[:2] != input_size:
        raise ValueError(f"Image size must be {input_size[0]}x{input_size[1]}, current size is {mask.shape[:2]}")

    rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device=device, dtype=torch.int32)
    H, W, _ = t.shape

    codes = t[..., 0] + (t[..., 1] << 8) + (t[..., 2] << 16)
    flat = codes.view(-1)

    uniq_codes, inverse = torch.unique(flat, return_inverse=True)

    N = flat.numel()
    sort_inv, perm_idx = torch.sort(inverse)
    change = torch.ones_like(sort_inv, dtype=torch.bool)
    change[1:] = sort_inv[1:] != sort_inv[:-1]
    first_pos = perm_idx[change]
    order = torch.argsort(first_pos, stable=True)
    new_label_by_old = torch.empty_like(order)
    new_label_by_old[order] = torch.arange(order.numel(), device=device)

    labels = new_label_by_old[inverse]
    black_code = torch.tensor(0, device=device, dtype=uniq_codes.dtype)
    black_old = (uniq_codes == black_code).nonzero(as_tuple=False)
    
    if black_old.numel() > 0:
        is_black_pixel = (inverse == black_old[0, 0])
        labels = labels + 1
        labels[is_black_pixel] = 0
    else:
        labels = labels + 1

    label_mask = labels.view(H, W).to(torch.int32)
    resized_mask = F.interpolate(
        label_mask.unsqueeze(0).unsqueeze(0).float(),
        size=output_size, mode='nearest'
    ).squeeze().to(torch.int32)

    return resized_mask.detach().cpu().numpy()


def content_loss(q_list, qc_list):
    loss = 0
    for q, qc in zip(q_list, qc_list):
        loss += loss_fn(q, qc.detach())
    return loss


def style_loss(q_list, ks_list, vs_list, self_out_list, scale=1):
    loss = 0
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        target_out = F.scaled_dot_product_attention(
            q * scale, ks, vs
        )
        loss += loss_fn(self_out, target_out.detach())
    return loss


def minimum_enclosing_circle(points):
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    center_list = [center[0], center[1]]
    return center_list, radius


def circle_intersection_area(c1, r1, c2, r2):
    d = np.linalg.norm(np.array(c1) - np.array(c2))
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2

    r1_sq = r1 ** 2
    r2_sq = r2 ** 2
    d_sq = d ** 2

    part1_val = np.clip((d_sq + r1_sq - r2_sq) / (2 * d * r1), -1.0, 1.0)
    part2_val = np.clip((d_sq + r2_sq - r1_sq) / (2 * d * r2), -1.0, 1.0)

    part1 = r1_sq * np.arccos(part1_val)
    part2 = r2_sq * np.arccos(part2_val)
    part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

    return part1 + part2 - part3


def calculate_overlap_ratio(c1, r1, c2, r2):
    intersection_area = circle_intersection_area(c1, r1, c2, r2)
    area1 = np.pi * r1 ** 2
    area2 = np.pi * r2 ** 2
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def get_route(original_path, ori_file_name=False):
    directory = os.path.dirname(original_path)
    first_part = os.path.split(directory)[0]
    last_directory = os.path.basename(directory)
    if ori_file_name is False:
        file_name = os.path.splitext(os.path.basename(original_path))[0]
    else:
        file_name = last_directory
    new_path = os.path.join(first_part, 'processed_data', last_directory, file_name + ".pt")

    return new_path


def remove_small_regions(mask, device, min_size=20):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    cleaned = mask.copy()

    cross = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], np.uint8)

    for u in np.unique(mask):
        comp = (mask == u).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=4)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                continue

            comp_i = (labels == i).astype(np.uint8)
            dil = cv2.dilate(comp_i, cross, iterations=1)
            ring = (dil.astype(bool) & ~comp_i.astype(bool))
            neighbor_vals = mask[ring]
            neighbor_vals = neighbor_vals[neighbor_vals != u]

            if neighbor_vals.size == 0:
                continue

            vals, counts = np.unique(neighbor_vals, return_counts=True)
            new_label = vals[counts.argmax()]

            cleaned[labels == i] = new_label

    cleaned = torch.from_numpy(cleaned).long().to(device)
    return cleaned


def ad_loss(
    q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None
):
    loss = 0
    attn_mask = None
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask = attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale, ks, vs, attn_mask=attn_mask
        )
        loss += loss_fn(self_out, target_out.detach())
    return loss