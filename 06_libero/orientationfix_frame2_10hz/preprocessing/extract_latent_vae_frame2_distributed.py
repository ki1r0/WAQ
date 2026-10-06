"""
extract_latent_vae.py — VAE latent extraction with randomised chunk offsets
============================================================================

Chunk sampling strategy
-----------------------
Fixed-length chunking without any offset causes every episode to tile at the
same phase boundaries (e.g. chunk 0 always covers the approach phase, chunk 1
always covers the manipulation phase, etc.).  At inference time the model sees
each new chunk with no prior KV-cache context from the matching task phase,
creating a train/test distribution mismatch at every chunk boundary.

To break this phase alignment we use the following scheme per episode:

  Chunk 0 (fixed):   [0, cs)
  Offset tiling:     [r, r+cs),  [r+cs, r+2cs),  [r+2cs, r+3cs),  ...

where cs = chunk_size and r is a deterministic pseudo-random offset derived
from the episode ID. This keeps the randomized phase boundary while making
all camera streams for an episode use the identical temporal windows.

Chunk 0 guarantees the model always trains on the task start from frame 0.
The offset tiling then covers the rest of the episode with randomised
boundaries: the only overlap with chunk 0 is the small region [r, cs), which
is at most cs-1 frames.  Total chunk count per episode is roughly the same as
the original fixed tiling — there is no significant data duplication.

The offset r is in *sampled-frame* units (after FPS downsampling) so that the
(n-1) % 4 == 0 VAE constraint is automatically satisfied for every chunk.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS
from wan_va.modules.utils import load_vae, load_tokenizer, load_text_encoder
from wan_va.utils import init_logger, logger


class WanVAEEncoder:
    """Wan2.2 VAE Encoder - 用于批量编码视频帧"""
    
    def __init__(self, vae_path, device):
        self.device = device
        self.vae = load_vae(vae_path, torch_dtype=torch.float32, torch_device=device)
        self.vae_stride = (4, 8, 8)  # VAE的时空压缩倍数
        
        # 加载VAE的均值和标准差
        self.latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        # Keep inverse-std directly to avoid repeated reciprocal.
        self.latents_inv_std = (1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32)).view(1, -1, 1, 1, 1)

        # 初始化video_processor用于后处理
        from diffusers.video_processor import VideoProcessor
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        
    def normalize_latents(self, latents, latents_mean, latents_std):
        """归一化latents"""
        # 确保latents是tensor
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(latents)}")
            
        # 获取latents的device
        device = latents.device
        
        # 调整均值和标准差的形状并移动到正确的device
        latents_mean = latents_mean.to(device=device)
        latents_std = latents_std.to(device=device)
        
        # 归一化
        latents = ((latents.float() - latents_mean) * latents_std).to(latents.dtype)
        return latents
        
    def encode_video(self, video_tensor):
        """
        编码整个视频tensor
        
        Args:
            video_tensor: [3, F, H, W] 归一化到[-1, 1]
            
        Returns:
            latent: [C_lat, F_lat, H_lat, W_lat] 归一化后的latent
        """
        with torch.inference_mode():
            # 添加batch维度 [1, C, F, H, W]
            if self.device.type == "cuda":
                video_tensor = video_tensor.pin_memory()
            video = video_tensor.unsqueeze(0).to(self.device, non_blocking=(self.device.type == "cuda"))
            
            # VAE编码
            if hasattr(self.vae, 'encode'):
                enc_out = self.vae.encode(video)
                
                # 处理不同类型的输出
                if hasattr(enc_out, 'latent_dist'):
                    # AutoencoderKLOutput 类型
                    mu = enc_out.latent_dist.mean
                elif isinstance(enc_out, tuple):
                    # 如果是tuple，可能是 (mu, logvar)
                    mu, logvar = enc_out
                else:
                    # 其他情况
                    mu = enc_out
            else:
                # 使用streaming_vae
                from wan_va.modules.utils import WanVAEStreamingWrapper
                streaming_vae = WanVAEStreamingWrapper(self.vae)
                enc_out = streaming_vae.encode_chunk(video)
                mu, logvar = torch.chunk(enc_out, 2, dim=1)
            
            # 确保mu是tensor
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"Expected mu to be torch.Tensor, got {type(mu)}")
            
            # 对latent进行归一化
            mu_norm = self.normalize_latents(mu, self.latents_mean, self.latents_inv_std)
            
            # 去除batch维度
            latent = mu_norm.squeeze(0)
            # Flatten latent: [C, F_lat, H_lat, W_lat] -> [F_lat * H_lat * W_lat, C]
            C = latent.shape[0]
            F_lat = latent.shape[1]
            H_lat = latent.shape[2]
            W_lat = latent.shape[3]
            
            # 重新排列维度: [C, F, H, W] -> [F, H, W, C] -> [F*H*W, C]
            latent = latent.permute(1, 2, 3, 0).contiguous()  # [F_lat, H_lat, W_lat, C]
            latent = latent.view(-1, C)  # [F_lat * H_lat * W_lat, C]
            
            # 转移到CPU
            latent = latent.cpu()
            
        return latent, C, F_lat, H_lat, W_lat

    def encode_video_batch(self, video_tensors):
        """
        批量编码视频tensor列表（要求每个tensor形状一致）

        Args:
            video_tensors: list of tensors, each [3, F, H, W]

        Returns:
            list of (latent, C, F_lat, H_lat, W_lat)
        """
        if len(video_tensors) == 0:
            return []
        if len(video_tensors) == 1:
            return [self.encode_video(video_tensors[0])]

        with torch.inference_mode():
            videos = torch.stack(video_tensors, dim=0)
            if self.device.type == "cuda":
                videos = videos.pin_memory()
            videos = videos.to(self.device, non_blocking=(self.device.type == "cuda"))  # [B, 3, F, H, W]

            if hasattr(self.vae, 'encode'):
                enc_out = self.vae.encode(videos)
                if hasattr(enc_out, 'latent_dist'):
                    mu = enc_out.latent_dist.mean
                elif isinstance(enc_out, tuple):
                    mu, _ = enc_out
                else:
                    mu = enc_out
            else:
                from wan_va.modules.utils import WanVAEStreamingWrapper
                streaming_vae = WanVAEStreamingWrapper(self.vae)
                enc_out = streaming_vae.encode_chunk(videos)
                mu, _ = torch.chunk(enc_out, 2, dim=1)

            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"Expected mu to be torch.Tensor, got {type(mu)}")

            mu_norm = self.normalize_latents(mu, self.latents_mean, self.latents_inv_std)  # [B, C, F, H, W]

            B, C, F_lat, H_lat, W_lat = mu_norm.shape
            results = []
            for b in range(B):
                latent = mu_norm[b].permute(1, 2, 3, 0).contiguous().view(-1, C).cpu()
                results.append((latent, C, F_lat, H_lat, W_lat))

        return results
    

class VideoProcessor:
    """视频处理器 - 负责帧采样和预处理"""
    
    def __init__(self, target_fps=10, target_height=256, target_width=256, chunk_size=450):
        self.target_fps = target_fps
        self.target_height = target_height
        self.target_width = target_width
        self.chunk_size = chunk_size  # 每个pth文件包含的帧数
        
        # 验证尺寸要求
        assert target_height % 32 == 0, f"Height must be multiple of 32, got {target_height}"
        assert target_width % 32 == 0, f"Width must be multiple of 32, got {target_width}"
    
    def process_video(self, video_path):
        """
        处理视频文件：采样帧、调整大小、返回所有帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            frames: list of RGB np.ndarray (H, W, 3), uint8
            frame_ids: list of frame indices
            metadata: dict with fps, num_frames, etc.
        """
        def decode_with_pyav():
            # OpenCV on zeng lacks AV1 support, while PyAV uses libdav1d.
            import av

            decoded = []
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                for frame in container.decode(stream):
                    decoded.append(frame.to_ndarray(format="rgb24"))
            return decoded, fps

        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded_frames = None
        if orig_fps <= 0 or total_frames <= 0:
            cap.release()
            decoded_frames, orig_fps = decode_with_pyav()
            total_frames = len(decoded_frames)
            logger.info(f"Using PyAV decode fallback for {video_path}")
        
        # 计算采样步长
        if self.target_fps >= orig_fps:
            step = 1
            actual_fps = orig_fps
        else:
            step = round(orig_fps / self.target_fps)
            actual_fps = orig_fps / step

        if step < 1 or not np.isclose(actual_fps, self.target_fps):
            raise ValueError(
                f"Cannot make exact {self.target_fps} Hz stream from {orig_fps} Hz with integer decimation"
            )
        
        # 确定采样帧数
        # Keep literal source-frame decimation: linspace alternates strides when
        # a video length is not divisible by the sampling period.
        sample_indices = list(range(0, total_frames, step))
        target_frame_count = len(sample_indices)
        
        # 调整帧数以满足 (num_frames - 1) % 4 == 0
        # 即 num_frames % 4 == 1
        adjusted_frame_count = target_frame_count
        if (adjusted_frame_count - 1) % 4 != 0:
            adjusted_frame_count = ((adjusted_frame_count - 1) // 4) * 4 + 1
        
        if adjusted_frame_count > target_frame_count:
            adjusted_frame_count -= 4
        sample_indices = sample_indices[:adjusted_frame_count]
        
        logger.info(f"Video: {video_path}")
        logger.info(f"  Original: {total_frames} frames @ {orig_fps:.2f}fps")
        logger.info(f"  Target: {adjusted_frame_count} frames @ {actual_fps:.2f}fps")
        
        # 采样所有帧
        all_frames = []
        all_frame_ids = []
        
        # 顺序解码比逐帧 cap.set 随机seek 更快、更稳定。
        if decoded_frames is None:
            want_ptr = 0
            cur_idx = 0
            while want_ptr < len(sample_indices):
                ret, frame = cap.read()
                if not ret:
                    break
                if cur_idx == sample_indices[want_ptr]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame,
                        (self.target_width, self.target_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    all_frames.append(frame)
                    all_frame_ids.append(int(cur_idx))
                    want_ptr += 1
                cur_idx += 1
            cap.release()

        if decoded_frames is None and len(all_frames) != len(sample_indices):
            logger.warning(
                f"OpenCV decoded {len(all_frames)}/{len(sample_indices)} frames; retrying with PyAV"
            )
            decoded_frames, pyav_fps = decode_with_pyav()
            if not np.isclose(pyav_fps, orig_fps):
                raise RuntimeError(f"Decoder FPS mismatch: OpenCV={orig_fps}, PyAV={pyav_fps}")

        if decoded_frames is not None:
            if len(decoded_frames) < total_frames:
                raise RuntimeError(f"PyAV decoded {len(decoded_frames)}/{total_frames} frames")
            all_frames = [
                cv2.resize(decoded_frames[idx], (self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)
                for idx in sample_indices
            ]
            all_frame_ids = sample_indices
        
        metadata = {
            'orig_fps': orig_fps,
            'target_fps': actual_fps,
            'orig_frames': total_frames,
            'total_sampled_frames': len(all_frames),
            'height': self.target_height,
            'width': self.target_width
        }
        
        return all_frames, all_frame_ids, metadata
    
    def split_into_chunks(self, frames, frame_ids, *, episode_id: int):
        """
        Split a sampled frame sequence into chunks using the two-series tiling
        scheme described in the module docstring.

        Always extracts chunk 0 as [0, chunk_size).  If the episode is longer
        than one chunk, samples r ~ randint(1, chunk_size) and tiles the rest of
        the episode starting at r: [r, r+cs), [r+cs, r+2cs), ...  The only
        overlap with chunk 0 is [r, cs) — at most cs-1 frames.  See the module
        docstring for the full motivation.

        Args:
            frames: list of PIL Images (sampled frames for one episode)
            frame_ids: list of original video frame indices
            episode_id: stable episode identifier shared by every camera stream
                for this trajectory

        Returns:
            chunks: list of (chunk_frames, chunk_frame_ids, start_idx, end_idx)
        """
        chunks = []
        total_frames = len(frames)

        def try_add_chunk(start_idx):
            if start_idx >= total_frames:
                return
            end_idx = min(start_idx + self.chunk_size, total_frames)
            chunk_frames = frames[start_idx:end_idx]
            chunk_frame_ids = frame_ids[start_idx:end_idx]

            # Enforce (n-1) % 4 == 0 required by the VAE temporal compression.
            chunk_len = len(chunk_frames)
            if (chunk_len - 1) % 4 != 0:
                adjusted_len = ((chunk_len - 1) // 4) * 4 + 1
                chunk_frames = chunk_frames[:adjusted_len]
                chunk_frame_ids = chunk_frame_ids[:adjusted_len]
                logger.info(
                    f"  Adjusted chunk from {chunk_len} to {adjusted_len} "
                    f"frames to satisfy (n-1)%4==0"
                )

            # Drop chunks that are too small after adjustment.  chunk_len 2-4
            # all collapse to 1 frame, which causes IndexError on
            # latent_frame_ids[1] during training.  Minimum useful size is 5
            # frames (produces 2 latent frames at temporal stride 4).
            if len(chunk_frames) < 5:
                logger.info(
                    f"  Skipping chunk at start_idx={start_idx}: "
                    f"only {len(chunk_frames)} frame(s) after adjustment (minimum is 5)"
                )
                return

            chunks.append((chunk_frames, chunk_frame_ids, start_idx, end_idx))

        # Fixed first chunk starting at frame 0.
        try_add_chunk(0)

        # Offset tiling covers the rest of the episode with randomized-looking
        # boundaries. Derive r deterministically from the episode ID and frame
        # count: every camera for a trajectory then writes the same windows,
        # which is required by the multi-camera latent loader.
        if total_frames > self.chunk_size:
            if self.chunk_size <= 1:
                raise ValueError("chunk_size must be greater than one for offset tiling")
            offset_seed = (
                int(episode_id) * 1_103_515_245
                + total_frames * 12_345
                + self.chunk_size * 2_654_435_761
            ) & 0xFFFFFFFF
            r = 1 + (offset_seed % (self.chunk_size - 1))
            for start_idx in range(r, total_frames, self.chunk_size):
                try_add_chunk(start_idx)

        return chunks
    
    def frames_to_tensor(self, frames):
        """
        将RGB np.ndarray列表转换为归一化的视频tensor
        
        Args:
            frames: list of np.ndarray(H, W, 3), RGB
            
        Returns:
            tensor: [3, F, H, W] 归一化到[-1, 1]
        """
        if len(frames) == 0:
            raise ValueError("frames_to_tensor got empty frame list")
        # [F, H, W, C] uint8 -> [F, C, H, W] float in [-1, 1]
        arr = np.stack(frames, axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        # [F, C, H, W] -> [C, F, H, W]
        video_tensor = tensor.permute(1, 0, 2, 3).contiguous()
        return video_tensor


def extract_text_embeddings(text, tokenizer, text_encoder, device, num_videos_per_prompt=1, max_length=512, dtype=torch.float32):
    """提取文本的T5 embeddings"""
    if not text or text == "":
        return None
    
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    
    text = [text] if isinstance(text, str) else text
    text = [prompt_clean(u) for u in text]
    batch_size = len(text)
    
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        mask.to(device)
    ).last_hidden_state

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack([
        torch.cat(
            [u, u.new_zeros(max_length - u.size(0), u.size(1))])
        for u in prompt_embeds
    ],
                                dim=0)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                        seq_len, -1)
    
    return prompt_embeds.cpu()


def detect_camera_key(latents_dir: Path):
    """从latents目录结构中自动检测第一个可用的摄像头键名。"""
    for chunk_dir in sorted(latents_dir.glob('chunk-*')):
        if not chunk_dir.is_dir():
            continue
        for cam_dir in sorted(chunk_dir.iterdir()):
            if cam_dir.is_dir():
                return cam_dir.name
    return None


def get_episode_windows(latents_dir: Path, episode_id: int, camera_key: str):
    """
    扫描latent文件名获取指定episode的所有时间窗口。
    文件名格式: episode_{id:06d}_{start_frame}_{end_frame}.pth
    过滤掉 end - start <= 1 的单帧chunk（训练时会导致 IndexError）。
    返回按start_frame排序的 (start, end) 列表。
    """
    windows = []
    for pth_file in latents_dir.glob(f'chunk-*/{camera_key}/episode_{episode_id:06d}_*.pth'):
        parts = pth_file.stem.split('_')
        try:
            start, end = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            logger.warning(f"无法解析latent文件名: {pth_file.name}")
            continue
        if end - start <= 1:
            logger.info(f"跳过单帧chunk: {pth_file.name}")
            continue
        windows.append((start, end))
    return sorted(windows)


def update_episodes_jsonl(dataset_root: Path, meta_dir: Path, latents_dir: Path):
    """
    latent提取完成后，由rank 0调用。
    扫描latent目录，将每个episode的实际时间窗口写入episodes.jsonl的action_config字段。
    原始文件备份为episodes_original.jsonl（仅在首次调用时备份）。
    """
    import shutil

    episodes_file = meta_dir / 'episodes.jsonl'
    backup_file = meta_dir / 'episodes_original.jsonl'

    if not episodes_file.exists():
        logger.error(f"episodes.jsonl不存在: {episodes_file}")
        return

    camera_key = detect_camera_key(latents_dir)
    if camera_key is None:
        logger.error(f"在 {latents_dir} 下未找到任何latent文件，跳过episodes.jsonl更新")
        return
    logger.info(f"更新episodes.jsonl，使用摄像头键名: {camera_key}")

    if not backup_file.exists():
        shutil.copy2(episodes_file, backup_file)
        logger.info(f"原始文件已备份到: {backup_file}")

    episodes = []
    with open(episodes_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            episode_id = ep['episode_index']
            windows = get_episode_windows(latents_dir, episode_id, camera_key)

            tasks = ep.get('tasks', [])
            action_text = (tasks[0] if isinstance(tasks, list) else tasks) if tasks else ''

            ep['action_config'] = [
                {'start_frame': s, 'end_frame': e, 'action_text': action_text, 'skill': ''}
                for s, e in windows
            ]
            if not windows:
                logger.warning(f"Episode {episode_id}: 未找到latent文件，action_config为空")
            episodes.append(ep)

    with open(episodes_file, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + '\n')

    total_windows = sum(len(ep['action_config']) for ep in episodes)
    logger.info(f"episodes.jsonl更新完成: {len(episodes)} 个episode，{total_windows} 个action_config窗口")


def load_metadata(meta_path):
    """
    加载episodes.jsonl文件
    
    Args:
        meta_path: path to episodes.jsonl
    
    Returns:
        dict: mapping from episode_id to metadata
    """
    metadata = {}
    
    if not os.path.exists(meta_path):
        logger.warning(f"Metadata file not found: {meta_path}")
        return metadata
    
    with open(meta_path, 'r') as f:
        for line in f:
            try:
                episode_meta = json.loads(line.strip())
                # 假设jsonl中每条记录包含episode_id和text等信息
                episode_index = episode_meta.get('episode_index')
                if episode_index is not None:
                    metadata[str(episode_index)] = episode_meta
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line: {line}")
    
    logger.info(f"Loaded metadata for {len(metadata)} episodes")
    return metadata


def process_episode(video_processor, vae_encoder, tokenizer, text_encoder,
                    video_path, text, episode_id, save_dir, device, chunk_size=450, batch_size=1):
    """
    处理单个episode：采样帧 -> 分块 -> 编码 -> 保存多个chunk文件
    
    Args:
        video_processor: VideoProcessor实例
        vae_encoder: WanVAEEncoder实例
        tokenizer: T5 tokenizer
        text_encoder: T5 text encoder
        video_path: 视频文件路径
        text: 文本指令
        episode_id: episode ID (用于文件名)
        save_dir: 保存目录
        device: torch device
        chunk_size: 每个chunk的帧数
    """
    # Step 1: 采样所有帧
    all_frames, all_frame_ids, metadata = video_processor.process_video(video_path)
    
    if len(all_frames) == 0:
        logger.error(f"No frames sampled from {video_path}")
        return
    
    # Step 2: 分成多个chunk
    video_processor.chunk_size = chunk_size
    chunks = video_processor.split_into_chunks(
        all_frames,
        all_frame_ids,
        episode_id=int(episode_id),
    )
    
    # Step 3: 提取text embeddings (只需一次)
    text_emb = None
    if text:
        with torch.no_grad():
            text_emb = extract_text_embeddings(text, tokenizer, text_encoder, device)
    
    # Step 4: 先转tensor，再按相同帧长做小批量VAE编码
    prepared_chunks = []
    for chunk_frames, chunk_frame_ids, start_idx, end_idx in chunks:
        if len(chunk_frames) == 0:
            continue
        video_tensor = video_processor.frames_to_tensor(chunk_frames)  # [3, F, H, W]
        prepared_chunks.append({
            "video_tensor": video_tensor,
            "chunk_frames": chunk_frames,
            "chunk_frame_ids": chunk_frame_ids,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

    i = 0
    effective_batch_size = max(1, int(batch_size))
    while i < len(prepared_chunks):
        current_f = prepared_chunks[i]["video_tensor"].shape[1]
        batch_items = [prepared_chunks[i]]
        j = i + 1
        while (
            j < len(prepared_chunks)
            and len(batch_items) < effective_batch_size
            and prepared_chunks[j]["video_tensor"].shape[1] == current_f
        ):
            batch_items.append(prepared_chunks[j])
            j += 1

        latents_info = vae_encoder.encode_video_batch([item["video_tensor"] for item in batch_items])

        for item, (latent, C, F_lat, H_lat, W_lat) in zip(batch_items, latents_info):
            chunk_frames = item["chunk_frames"]
            chunk_frame_ids = item["chunk_frame_ids"]
            latent_num_frames = F_lat

            global_start_frame = chunk_frame_ids[0]
            global_end_frame = chunk_frame_ids[-1] + 1

            save_data = {
                'latent': latent,
                'latent_num_frames': latent_num_frames,
                'latent_height': H_lat,
                'latent_width': W_lat,
                'video_num_frames': len(chunk_frames),
                'video_height': metadata['height'],
                'video_width': metadata['width'],
                'text_emb': text_emb.squeeze(0) if text_emb is not None else None,
                'text': text,
                'frame_ids': chunk_frame_ids,
                'start_frame': global_start_frame,
                'end_frame': global_end_frame,
                'fps': metadata['target_fps'],
                'ori_fps': metadata['orig_fps'],
            }

            save_filename = f'episode_{episode_id:06d}_{global_start_frame}_{global_end_frame}.pth'
            save_path = save_dir / save_filename
            torch.save(save_data, save_path)

            logger.info(f"Saved {save_path}")
            logger.info(f"  Video frames: {len(chunk_frames)} -> Latent frames: {latent_num_frames}")
            logger.info(f"  Frame range: {global_start_frame} - {global_end_frame}")

        i = j
    
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Extract latents using Wan2.2 VAE")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin',
        help="config name from VA_CONFIGS"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of your_dataset (containing videos/, latents/, meta/)"
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        nargs='+',
        default=['cam_high'],
        help="Camera keys to process (e.g., cam_high, cam_left_wrist, cam_right_wrist)"
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10,
        help="Target FPS for frame sampling"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Target height (must be multiple of 32)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Target width (must be multiple of 32)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help="Number of frames per output pth file"
    )
    parser.add_argument(
        "--chunk-ids",
        type=str,
        nargs='+',
        help="Specific chunk IDs to process (e.g., 000 001). If not specified, process all."
    )
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs='+',
        default=None,
        help="Optional explicit output episode indices; useful for bounded smoke runs."
    )
    parser.add_argument(
        "--episode-success-filter",
        choices=("all", "success", "failure"),
        default="all",
        help="Select outcomes from meta/episodes.jsonl; default preserves legacy all-episode behavior."
    )
    parser.add_argument(
        "--latents-dir",
        type=str,
        default=None,
        help="Optional latent-output directory; default is <dataset-root>/latents."
    )
    parser.add_argument(
        "--skip-episodes-jsonl-update",
        action="store_true",
        help="Do not rewrite action_config in meta/episodes.jsonl (for isolated smoke extraction)."
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="VAE encode batch size (chunks with same frame length are batched)"
    )
    
    args = parser.parse_args()
    
    # 初始化日志
    init_logger()
    
    # 验证尺寸
    assert args.height % 32 == 0, f"Height must be multiple of 32, got {args.height}"
    assert args.width % 32 == 0, f"Width must be multiple of 32, got {args.width}"
    
    # 获取配置
    config = VA_CONFIGS[args.config_name]
    env_local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.local_rank = env_local_rank
    device = torch.device(f"cuda:{env_local_rank}")
    logger.info(f"Distributed shard: rank={rank}, world_size={world_size}, local_rank={env_local_rank}")

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(env_local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    
    # 初始化组件
    logger.info("Initializing VAE encoder...")
    vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'vae')
    vae_encoder = WanVAEEncoder(vae_path, device)
    
    logger.info("Loading tokenizer and text encoder...")
    tokenizer = load_tokenizer(
        os.path.join(config.wan22_pretrained_model_name_or_path, 'tokenizer')
    )
    text_encoder = load_text_encoder(
        os.path.join(config.wan22_pretrained_model_name_or_path, 'text_encoder'),
        torch_dtype=config.param_dtype,
        torch_device=device,
    )
    
    # 初始化视频处理器
    video_processor = VideoProcessor(
        target_fps=args.target_fps,
        target_height=args.height,
        target_width=args.width,
        chunk_size=args.chunk_size
    )
    
    # 设置目录
    dataset_root = Path(args.dataset_root)
    videos_dir = dataset_root / 'videos'
    latents_dir = Path(args.latents_dir) if args.latents_dir else dataset_root / 'latents'
    meta_dir = dataset_root / 'meta'
    
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return

    # Refuse to run if latents/ already exists and contains files, to prevent
    # silent overwrites of a partially or fully completed extraction.
    # A shared torchrun output directory is intentionally written by all
    # ranks.  The non-empty guard is only safe in single-process mode;
    # otherwise a faster rank can write while rank 0 is still loading models.
    if world_size == 1 and latents_dir.exists() and any(latents_dir.rglob('*.pth')):
        logger.error(
            f"latents directory already contains .pth files: {latents_dir}\n"
            "  Remove or rename it before re-running extraction."
        )
        return

    # 创建latents目录
    latents_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载metadata
    meta_file = meta_dir / 'episodes.jsonl'
    episode_metadata = load_metadata(meta_file)
    selected_episode_ids = set(args.episode_ids) if args.episode_ids is not None else None
    
    # 获取所有chunk目录
    if args.chunk_ids:
        chunk_dirs = [videos_dir / f'chunk-{chunk_id}' for chunk_id in args.chunk_ids]
        chunk_dirs = [d for d in chunk_dirs if d.exists()]
    else:
        chunk_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith('chunk-')])
    
    logger.info(f"Found {len(chunk_dirs)} chunks to process")
    
    # 展平任务列表后按 rank 切分，避免多GPU重复处理同一文件
    all_jobs = []
    for chunk_dir in chunk_dirs:
        chunk_id = chunk_dir.name.replace('chunk-', '')
        for camera_key in args.camera_keys:
            video_camera_dir = chunk_dir / f'observation.images.{camera_key}'
            if not video_camera_dir.exists():
                logger.warning(f"Camera directory not found: {video_camera_dir}")
                continue
            mp4_files = sorted(video_camera_dir.glob('episode_*.mp4'))
            logger.info(f"Found {len(mp4_files)} mp4 files in {video_camera_dir}")
            for mp4_file in mp4_files:
                episode_id = int(mp4_file.stem.replace('episode_', ''))
                if selected_episode_ids is not None and episode_id not in selected_episode_ids:
                    continue
                if args.episode_success_filter != "all":
                    metadata = episode_metadata.get(str(episode_id))
                    if metadata is None or "success" not in metadata:
                        raise ValueError(
                            "episode-success filtering requires explicit success metadata: "
                            f"episode {episode_id} in {meta_file}"
                        )
                    target_success = args.episode_success_filter == "success"
                    if bool(metadata["success"]) != target_success:
                        continue
                all_jobs.append((chunk_id, camera_key, mp4_file))

    jobs = all_jobs[rank::world_size]
    logger.info(f"Total jobs: {len(all_jobs)} | Jobs for this rank: {len(jobs)}")

    total_episodes = 0
    total_chunks = 0
    for chunk_id, camera_key, mp4_file in jobs:
        episode_name = mp4_file.stem
        episode_id = int(episode_name.replace('episode_', ''))

        text = ""
        if str(episode_id) in episode_metadata:
            meta = episode_metadata[str(episode_id)]
            text = meta.get('tasks', '')
        else:
            logger.warning(f"No metadata found for episode {episode_id}")

        save_camera_dir = latents_dir / f'chunk-{chunk_id}' / f'observation.images.{camera_key}'
        save_camera_dir.mkdir(parents=True, exist_ok=True)

        num_chunks = process_episode(
            video_processor=video_processor,
            vae_encoder=vae_encoder,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            video_path=str(mp4_file),
            text=text,
            episode_id=episode_id,
            save_dir=save_camera_dir,
            device=device,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )

        if num_chunks:
            total_episodes += 1
            total_chunks += num_chunks

        torch.cuda.empty_cache()
    
    logger.info(f"Latent extraction completed! Processed {total_episodes} episodes, {total_chunks} chunks")

    # 所有rank完成后，由rank 0更新episodes.jsonl中的action_config
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if rank == 0 and not args.skip_episodes_jsonl_update:
        update_episodes_jsonl(dataset_root, meta_dir, latents_dir)

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
