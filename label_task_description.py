import os, json, cv2, base64
import requests
import h5py
import numpy as np
import re

# MCAP 支持（可选，用于 .mcap 文件）
try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    _has_mcap = True
except ImportError:
    _has_mcap = False
# ==== 代理保持你刚才的环境变量 ====
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["http_proxy"]  = "http://127.0.0.1:7890"

# OpenAI 兼容网关配置
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://kfc-api.sxxe.net").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL") or "gemini-3-pro-preview"

# 不再在导入时强制要求 OPENAI_API_KEY，允许通过请求参数传入

# 返回视频语义阶段的JSON列表
# negative 和 hard_negative 字段用于负样本标注，当前任务只需要填写 positive 描述即可
PROMPT_TEMPLATE = (
    "Below are frames from {m} robot manipulation videos (labeled Episode 0 to Episode {m_minus_1}). "
    "For EACH video, carefully observe the ENTIRE sequence and describe the SPECIFIC task performed. "
    "Focus on: "
    "1) which arm is used (left/right/both), "
    "2) what object is grasped (be specific about color, shape, type), "
    "3) what action is performed on what target (be specific about the target location/object). "
    "Pay attention to details that make EACH episode unique. "
    "Return ONLY a JSON object in this exact format (no markdown, no extra fields):\n"
    '{{"instructions": ["description for episode 0", "description for episode 1", ...]}}\n'
    "Each description string must be ≤20 words."
)


def _openai_chat_completion(
    messages,
    model: str = None,
    temperature: float = 0.2,
    timeout: int = 120,
    api_key: str = None,
    base_url: str = None,
):
    """OpenAI 兼容 /v1/chat/completions 调用。api_key/base_url/model 为 None 时使用环境变量。"""
    base = (base_url or OPENAI_BASE_URL).rstrip("/")
    key = api_key or OPENAI_API_KEY
    if not key:
        raise RuntimeError("未设置 OPENAI_API_KEY。请设置环境变量或在请求中传入 api_key。")
    url = f"{base}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or MODEL,
        "messages": messages,
        "temperature": temperature,
        # 尽量约束返回 JSON 文本；不支持该字段的兼容网关通常会忽略
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"OpenAI API error: HTTP {r.status_code} - {r.text}")
    data = r.json()
    try:
        choices = data.get("choices") or []
        if not choices:
            raise KeyError("choices")
        first = choices[0] or {}
        msg = first.get("message") or {}
        content = msg.get("content")
        # 兼容部分模型返回 content 为分段数组（如 [{"type":"text","text":"..."}]）
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
            content = "\n".join(parts).strip()
        if isinstance(content, str) and content.strip():
            return content
        # 兼容少数兼容网关返回 text 字段
        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text
        # 兼容部分网关返回 message.parts / message.output_text
        parts = msg.get("parts")
        if isinstance(parts, list):
            txts = []
            for p in parts:
                if isinstance(p, str):
                    txts.append(p)
                elif isinstance(p, dict):
                    t = p.get("text")
                    if isinstance(t, str) and t.strip():
                        txts.append(t)
            merged = "\n".join(txts).strip()
            if merged:
                return merged
        out_txt = msg.get("output_text")
        if isinstance(out_txt, str) and out_txt.strip():
            return out_txt
        raise KeyError("content")
    except Exception:
        raise RuntimeError(
            "OpenAI 响应缺少可解析文本（message.content）。"
            f" model={payload.get('model')} raw={json.dumps(data, ensure_ascii=False)[:800]}"
        )


def adaptive_sample_frames(frames, target_frames=16):
    """
    自适应采样：开头、中间、结尾均匀覆盖，确保能捕捉到不同episode的差异。
    frames: 任意可索引序列（如 base64 字符串列表）
    """
    n = len(frames)
    if n <= target_frames:
        return frames

    # 改进策略：更均匀地覆盖整个视频，重点捕捉中间动作差异
    # 前 30% 取 5 帧，中间 40% 取 6 帧，后 30% 取 5 帧（总 16）
    first_end = int(n * 0.3)
    mid_start = first_end
    mid_end = int(n * 0.7)
    
    first_part = np.linspace(0, first_end, 5, dtype=int, endpoint=False)
    mid_part = np.linspace(mid_start, mid_end, 6, dtype=int, endpoint=False)
    last_part = np.linspace(mid_end, n - 1, 5, dtype=int)

    indices = np.unique(np.concatenate([first_part, mid_part, last_part]))
    indices = indices[:target_frames]
    return [frames[i] for i in indices]


def _build_multimodal_messages(frames64):
    """统一构造多模态输入，确保不同模型使用完全一致的提示词与图像读入格式。"""
    prompt = PROMPT_TEMPLATE.format(m=1, m_minus_1=0)
    user_content = [{"type": "text", "text": prompt}]
    for f64 in frames64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f64}"},
            }
        )
    return [{"role": "user", "content": user_content}]


# ========== MCAP 相关函数 ==========

def _iter_mcap_messages(mcap_path):
    """迭代 MCAP 消息：(schema, channel, message, decoded_ros2_msg)"""
    if not _has_mcap:
        raise RuntimeError("未安装 mcap/mcap-ros2-support，请运行: pip install mcap mcap-ros2-support")
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            yield schema, channel, message, ros_msg


def list_image_topics_mcap(mcap_path):
    """列出 MCAP 中可用的图像话题（CompressedImage / Image）"""
    topics = []
    seen = set()
    for schema, channel, message, decoded in _iter_mcap_messages(mcap_path):
        topic = getattr(channel, "topic", "")
        if topic in seen:
            continue
        if decoded is None:
            continue
        # CompressedImage: format + data
        has_format = hasattr(decoded, "format") or (isinstance(decoded, dict) and "format" in decoded)
        has_data = hasattr(decoded, "data") or (isinstance(decoded, dict) and "data" in decoded)
        if has_format and has_data:
            seen.add(topic)
            topics.append(topic)
        # Image: height, width, data
        elif hasattr(decoded, "height") and hasattr(decoded, "width") and has_data:
            seen.add(topic)
            topics.append(topic)
    return topics


def _decode_compressed_image(msg):
    """从 CompressedImage 消息解码为 numpy 图像 (BGR)"""
    data = getattr(msg, "data", None) or (msg.get("data") if isinstance(msg, dict) else None)
    if data is None:
        return None
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _decode_raw_image(msg):
    """从 Image 消息解码为 numpy 图像 (BGR)"""
    data = getattr(msg, "data", None) or (msg.get("data") if isinstance(msg, dict) else None)
    height = int(getattr(msg, "height", 0) or msg.get("height", 0))
    width = int(getattr(msg, "width", 0) or msg.get("width", 0))
    encoding = str(getattr(msg, "encoding", "rgb8") or msg.get("encoding", "rgb8"))
    if not data or height <= 0 or width <= 0:
        return None
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8", "8UC3"):
        img = raw.reshape((height, width, 3))
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if encoding in ("mono8", "8UC1"):
        img = raw.reshape((height, width))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return None


def sample_frames_from_mcap(mcap_path, n=None, image_topic=None):
    """从 MCAP 文件中采样图像帧，返回 (frames64, fps)"""
    if not _has_mcap:
        raise RuntimeError("未安装 mcap/mcap-ros2-support，请运行: pip install mcap mcap-ros2-support")
    if n is None:
        try:
            n = int(os.getenv("FRAME_SAMPLE_N", "100"))
        except Exception:
            n = 100
    if n < 16:
        n = 16

    topics = list_image_topics_mcap(mcap_path)
    if not topics:
        raise RuntimeError("MCAP 中未发现图像话题（CompressedImage/Image）")

    # 排除 depth 话题（深度图无法用于视觉描述）
    non_depth = [t for t in topics if "depth" not in t.lower()]
    if not non_depth:
        raise RuntimeError(
            "该 MCAP 仅包含 depth 话题，无法用于视觉标注。"
            "请使用包含 color 相机的数据，或指定其他 MCAP 文件。"
        )
    candidates = non_depth

    if image_topic is None:
        for t in candidates:
            if "color" in t and "compressed" in t.lower():
                image_topic = t
                break
        if image_topic is None:
            image_topic = candidates[0]
    elif image_topic not in topics:
        print(f"警告: 图像话题 {image_topic} 不存在，尝试候选项")
        image_topic = candidates[0]

    frames = []
    for schema, channel, message, decoded in _iter_mcap_messages(mcap_path):
        topic = getattr(channel, "topic", "")
        if topic != image_topic:
            continue
        if decoded is None:
            continue
        img = None
        if hasattr(decoded, "format") or (isinstance(decoded, dict) and decoded.get("format")):
            img = _decode_compressed_image(decoded)
        elif hasattr(decoded, "height") and hasattr(decoded, "width"):
            img = _decode_raw_image(decoded)
        if img is not None:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(base64.b64encode(buf).decode())

    if not frames:
        raise RuntimeError(f"话题 {image_topic} 中未能解码到任何图像")

    total = len(frames)
    n = max(1, int(n))
    indices = [int(i * total / n) for i in range(n)]
    frames64 = [frames[i] for i in indices if i < total]

    return frames64, 30.0


def gen_task_description_mcap(mcap_path, image_topic=None, model=None, api_key=None, base_url=None):
    """从 MCAP 文件生成任务描述"""
    frames64, fps = sample_frames_from_mcap(mcap_path, n=None, image_topic=image_topic)
    max_imgs = 16
    frames64 = adaptive_sample_frames(frames64, target_frames=max_imgs)
    messages = _build_multimodal_messages(frames64)
    text = _openai_chat_completion(
        messages, model=model or MODEL, temperature=0.2, timeout=180,
        api_key=api_key, base_url=base_url
    ).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    try:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()

        parsed = json.loads(text)
        if isinstance(parsed, dict) and "instructions" in parsed:
            instructions = parsed["instructions"]
            if isinstance(instructions, list) and len(instructions) > 0:
                return instructions[0]
            if isinstance(instructions, str):
                return instructions
        return text
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"警告: 解析任务描述失败，返回原始文本: {e}")
    return text


# ========== HDF5 相关函数 ==========

def _group_has_image_children(grp):
    """判断组是否包含“图像状”直接子节点（Dataset 或子 Group 内含图像 Dataset）。"""
    for k in grp.keys():
        if "timestamp" in k.lower():
            continue
        obj = grp[k]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
            return True
        if isinstance(obj, h5py.Group):
            for kk in obj.keys():
                sub = obj[kk]
                if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                    return True
    return False


def _scan_tree_for_image_group(hdf, grp):
    """从给定组开始 DFS，找到第一个“直接子节点含图像”的组，返回 (path, group) 或 (None, None)。"""
    if grp is None or not isinstance(grp, h5py.Group):
        return None, None
    if _group_has_image_children(grp):
        path = grp.name if grp.name else "/"
        return path, grp
    for k in grp.keys():
        child = grp[k]
        if isinstance(child, h5py.Group):
            found_path, found_grp = _scan_tree_for_image_group(hdf, child)
            if found_grp is not None:
                return found_path, found_grp
    return None, None


def find_image_group(hdf):
    """自动定位图像所在的组，并返回(组路径字符串, 组对象)。先尝试固定路径，再全树扫描。"""
    candidate_groups = [
        "observations/images",
        "images",
        "observations",
    ]
    for g in candidate_groups:
        if g in hdf:
            grp = hdf[g]
            if isinstance(grp, h5py.Group):
                # 检查该组下是否包含图像数据集
                for k in grp.keys():
                    obj = grp[k]
                    if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
                        return g, grp
                    if isinstance(obj, h5py.Group):
                        # 检查子组中是否有图像数据
                        for kk in obj.keys():
                            sub = obj[kk]
                            if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                                return g, grp
    # 固定路径未找到：从根目录全树扫描
    return _scan_tree_for_image_group(hdf, hdf)

def list_cameras_in_group(grp):
    """列出组下可用的相机键名"""
    cameras = []
    for k in grp.keys():
        obj = grp[k]
        # 跳过时间戳数据
        if 'timestamp' in k.lower():
            continue
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 and obj.shape[-1] in (1, 3, 4):
            cameras.append(k)
        elif isinstance(obj, h5py.Group):
            # 如果子组内包含图像数据集
            for kk in obj.keys():
                sub = obj[kk]
                if isinstance(sub, h5py.Dataset) and sub.ndim >= 3 and sub.shape[-1] in (1, 3, 4):
                    cameras.append(k)
                    break
    return cameras

def get_camera_data(hdf, group_path, camera_name):
    """获取相机数据"""
    grp = hdf[group_path]
    node = grp[camera_name]
    
    # 情况1：直接是 (T,H,W,C) 的数据集
    if isinstance(node, h5py.Dataset):
        return node[:]
    
    # 情况2：是子组，内部再有 dataset
    if isinstance(node, h5py.Group):
        candidate_ds = ["data", "images", "frames"] + list(node.keys())
        for ds in candidate_ds:
            if ds in node and isinstance(node[ds], h5py.Dataset):
                ds_node = node[ds]
                if ds_node.ndim >= 3:
                    return ds_node[:]
    
    raise RuntimeError(f"无法识别相机 {camera_name} 的数据存储方式")

def sample_frames_from_hdf5(hdf5_path, n=None, camera_name=None):
    """从HDF5文件中采样帧"""
    # 采样帧数：默认 100，可用环境变量覆盖（避免请求过大时快速降级）
    # 注意：实际发送时会通过 adaptive_sample_frames 降到 16 帧
    if n is None:
        try:
            n = int(os.getenv("FRAME_SAMPLE_N", "100"))
        except Exception:
            n = 100
    # 确保至少采样 16 帧（后续会自适应降采样到 16）
    if n < 16:
        n = 16
    with h5py.File(hdf5_path, 'r') as f:
        # 查找图像组
        group_path, grp = find_image_group(f)
        if grp is None:
            raise RuntimeError("未找到图像组（尝试了 observations/images、images、observations）")
        
        # 获取相机列表
        cameras = list_cameras_in_group(grp)
        if len(cameras) == 0:
            raise RuntimeError(f"在组 {group_path} 下未发现相机数据集")
        
        # 如果没有指定相机，使用第一个相机
        if camera_name is None:
            camera_name = cameras[0]
        elif camera_name not in cameras:
            print(f"警告: 相机 {camera_name} 不存在，使用第一个相机 {cameras[0]}")
            camera_name = cameras[0]
        
        # 获取相机数据
        camera_data = get_camera_data(f, group_path, camera_name)
        total = camera_data.shape[0]
        
        # 均匀采样帧
        n = max(1, int(n))
        indices = [int(i * total / n) for i in range(n)]
        frames64 = []
        
        for idx in indices:
            if idx >= total:
                continue
            
            # 获取帧
            frame = camera_data[idx]
            
            # 处理图像格式
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            
            # 转换颜色格式
            if frame.ndim == 2:
                # 灰度图像转BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[-1] == 4:
                # RGBA转BGR
                frame_bgr = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
            elif frame.shape[-1] == 3:
                # RGB转BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # 编码为base64
            _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames64.append(base64.b64encode(buf).decode())
        
        # 估算fps（假设30fps，或根据时间戳计算）
        fps = 30.0  # 默认fps
        if 'timestamps' in f:
            timestamps = f['timestamps'][:]
            if len(timestamps) > 1:
                fps = 1.0 / (timestamps[1] - timestamps[0]) if timestamps[1] != timestamps[0] else 30.0
        
        return frames64, fps

def gen_task_description(file_path, camera_name=None, image_topic=None, model=None, api_key=None, base_url=None):
    """
    从 HDF5 或 MCAP 文件生成任务描述字符串。
    - HDF5: 使用 camera_name 指定相机
    - MCAP: 使用 image_topic 指定图像话题（如 /camera2/camera/color/image_raw/compressed）
    - model/api_key/base_url 可选，为 None 时使用环境变量
    """
    fp_lower = file_path.lower()
    if fp_lower.endswith(".mcap"):
        return gen_task_description_mcap(file_path, image_topic=image_topic, model=model, api_key=api_key, base_url=base_url)
    return _gen_task_description_hdf5(file_path, camera_name=camera_name, model=model, api_key=api_key, base_url=base_url)


def _gen_task_description_hdf5(hdf5_path, camera_name=None, model=None, api_key=None, base_url=None):
    """从 HDF5 文件生成任务描述字符串"""
    frames64, fps = sample_frames_from_hdf5(hdf5_path, n=None, camera_name=camera_name)
    # 接口限制：最多 16 张图；先自适应降采样，再发请求
    max_imgs = 16
    frames64 = adaptive_sample_frames(frames64, target_frames=max_imgs)
    messages = _build_multimodal_messages(frames64)

    text = _openai_chat_completion(
        messages, model=model or MODEL, temperature=0.2, timeout=180,
        api_key=api_key, base_url=base_url
    ).strip()
    # 移除可能的引号
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    # 解析返回的文本，提取纯文本描述
    # 可能包含 markdown 代码块：```json\n{...}\n```
    # 或直接是 JSON 字符串：{...}
    try:
        # 尝试移除 markdown 代码块
        if "```json" in text:
            # 提取 ```json 和 ``` 之间的内容
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            # 处理其他代码块格式
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        
        # 解析 JSON
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "instructions" in parsed:
            instructions = parsed["instructions"]
            if isinstance(instructions, list) and len(instructions) > 0:
                # 返回第一个 instruction 的纯文本
                return instructions[0]
            elif isinstance(instructions, str):
                return instructions
        # 如果不是期望的格式，返回原始文本
        return text
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # 如果解析失败，返回原始文本
        print(f"警告: 解析任务描述失败，返回原始文本: {e}")
    return text

def get_episode_length(hdf5_path, camera_name=None):
    """获取episode的总帧数（length）"""
    with h5py.File(hdf5_path, 'r') as f:
        group_path, grp = find_image_group(f)
        if grp is None:
            print(f"警告: {hdf5_path} 未找到图像组，返回 length=0")
            return 0
        
        cameras = list_cameras_in_group(grp)
        if not cameras:
            print(f"警告: {hdf5_path} 未找到相机，返回 length=0")
            return 0
        
        if camera_name is None:
            camera_name = cameras[0]
        elif camera_name not in cameras:
            print(f"警告: {hdf5_path} 相机 {camera_name} 不存在，使用 {cameras[0]}")
            camera_name = cameras[0]
        
        camera_data = get_camera_data(f, group_path, camera_name)
        frame_count = int(camera_data.shape[0])
        print(f"✓ {os.path.basename(hdf5_path)} | camera={camera_name} | frames={frame_count}")
        return frame_count

def extract_episode_index(hdf5_path):
    """从文件路径中提取episode索引"""
    filename = os.path.basename(hdf5_path)
    # 尝试匹配 episode_0, episode_1 等格式
    match = re.search(r'episode[_\s]*(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # 如果没匹配到，尝试从文件名中提取数字
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # 默认返回0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="对 HDF5 或 MCAP 文件进行自动标注")
    parser.add_argument(
        "data_path", nargs="?",
        default=None,
        help=".mcap 或 .hdf5 文件路径（可选，不填则使用默认路径）"
    )
    parser.add_argument(
        "--camera", "-c", default=None,
        help="HDF5 相机名称，如 camera2"
    )
    parser.add_argument(
        "--image-topic", "-t", default=None,
        help="MCAP 图像话题，如 /camera2/camera/color/image_raw/compressed"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_path:
        data_path = os.path.abspath(args.data_path)
    else:
        # 默认 MCAP 文件
        data_path = os.path.join(
            script_dir, "data", "mcap_data", "frank_data",
            "episode_1190_20260119_171114", "episode_1190_20260119_171114",
            "episode_1190_20260119_171114_0.mcap"
        )
    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, "episode_0.hdf5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据文件，请指定有效的 .mcap 或 .hdf5 路径")

    is_mcap = data_path.lower().endswith(".mcap")
    print(f"→ 处理 {'MCAP' if is_mcap else 'HDF5'} 文件: {data_path}")

    camera_name = args.camera if not is_mcap else None
    if camera_name is None and not is_mcap:
        camera_name = "camera2"
    image_topic = args.image_topic if is_mcap else None

    try:
        task_description = gen_task_description(
            data_path, camera_name=camera_name, image_topic=image_topic
        )
        episode_index = extract_episode_index(data_path)
        # length 记录任务描述文本的字符数
        length = len(task_description)
        # 输出到数据文件所在目录
        out = os.path.join(os.path.dirname(os.path.abspath(data_path)), "instruction.json")
        
        # 读取已存在的文件（如果存在）
        existing_episodes = {}  # episode_index -> instruction_text
        if os.path.exists(out):
            try:
                # 尝试读取为单个JSON对象格式
                with open(out, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict) and "instructions" in data:
                                # 新格式：单个JSON对象
                                instructions_list = data["instructions"]
                                # 重建 existing_episodes 映射（按索引顺序）
                                for idx, inst in enumerate(instructions_list):
                                    existing_episodes[idx] = inst
                            else:
                                # 旧格式：JSON Lines，尝试转换
                                f.seek(0)
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            ep = json.loads(line)
                                            if isinstance(ep, dict) and "episode_index" in ep:
                                                ep_idx = ep["episode_index"]
                                                task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
                                                task_text = task
                                                if "```json" in task or "{" in task:
                                                    try:
                                                        if "```json" in task:
                                                            start = task.find("```json") + 7
                                                            end = task.find("```", start)
                                                            if end != -1:
                                                                task = task[start:end].strip()
                                                        parsed = json.loads(task)
                                                        if isinstance(parsed, dict) and "instructions" in parsed:
                                                            inst = parsed["instructions"]
                                                            task_text = inst[0] if isinstance(inst, list) and inst else str(inst)
                                                    except:
                                                        pass
                                                existing_episodes[ep_idx] = task_text
                                        except json.JSONDecodeError:
                                            continue
                        except json.JSONDecodeError:
                            # 如果整个文件不是JSON，尝试按JSON Lines解析
                            f.seek(0)
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        ep = json.loads(line)
                                        if isinstance(ep, dict) and "episode_index" in ep:
                                            ep_idx = ep["episode_index"]
                                            task = ep.get("tasks", [""])[0] if ep.get("tasks") else ""
                                            task_text = task
                                            if "```json" in task or "{" in task:
                                                try:
                                                    if "```json" in task:
                                                        start = task.find("```json") + 7
                                                        end = task.find("```", start)
                                                        if end != -1:
                                                            task = task[start:end].strip()
                                                    parsed = json.loads(task)
                                                    if isinstance(parsed, dict) and "instructions" in parsed:
                                                        inst = parsed["instructions"]
                                                        task_text = inst[0] if isinstance(inst, list) and inst else str(inst)
                                                except:
                                                    pass
                                            existing_episodes[ep_idx] = task_text
                                    except json.JSONDecodeError:
                                        continue
            except Exception as e:
                print(f"读取现有文件时出错: {e}")
        
        # 更新或添加当前episode的描述
        existing_episodes[episode_index] = task_description
        
        # 按episode_index排序，构建instructions数组
        sorted_episodes = sorted(existing_episodes.items())
        instructions_list = [desc for _, desc in sorted_episodes]
        
        # 写入单个JSON对象格式
        output_data = {"instructions": instructions_list}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  保存 {out}")
        print(f"  Episode {episode_index}: {task_description} (length: {length})")
    except Exception as e:
        print(f"  处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()