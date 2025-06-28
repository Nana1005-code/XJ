# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for LLaVa-Onevision.
"""

import math
from collections.abc import Iterable
from typing import List, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import select_best_resolution
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)

#processor参数类
class LlavaOnevisionProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "image_kwargs": {}, #图像处理相关参数，初始化为空字典
        "videos_kwargs": {},
    }


class LlavaOnevisionProcessor(ProcessorMixin):
    r"""
    用于将文本、图像和视频输入统一处理成 LLaVa-OneVision 模型可以接受的格式
    主要职责为,文本使用tokenizer,图像使用image_processor,视频使用video_processor
    协调多模态输入,将 <image> 或 <video> 等特殊 token 替换为对应的视觉 token 序列
    Constructs a LLaVa-Onevision processor which wraps a LLaVa-Onevision video processor, LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaOnevisionVideoProcessor`], [`LlavaOnevisionImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaOnevisionVideoProcessor.__call__`], [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaOnevisionImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`LlavaOnevisionVideoProcessor`], *optional*):
            The video processor is a required input.
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"
    
    #初始化处理器
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        num_image_tokens=None,#特殊标记,例如IMG的个数
        vision_feature_select_strategy=None,
        chat_template=None,#对话模板
        image_token="<image>",#特殊标记,例如IMG
        video_token="<video>",
        vision_aspect_ratio="anyres_max_9",#控制图像分割策略
        **kwargs,
    ):
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        #下述为设置IMG占位符实际为什么
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        #利用tokenizer将IMG编码为同一个id
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_aspect_ratio = vision_aspect_ratio
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    #调用处理器时执行的方法，处理传入的text，images，videos
    #一般是直接调用__call__方法，然后得到input
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[LlavaOnevisionProcessorKwargs],
    ) -> BatchFeature:
        """
        主方法，用于为模型准备一个或多个序列和图像。该方法会将 `text` 和 `kwargs` 参数转发给 LlamaTokenizerFast 的 [`~LlamaTokenizerFast.__call__`] 方法（如果 `text` 不为 None）以对文本进行编码。若要处理图像，则会将 `images` 和 `kwargs` 参数转发给 LlavaNextImageProcessor 的 [`~LlavaNextImageProcessor.__call__`] 方法（如果 `images` 不为 None）。详细信息请参考上述两个方法的文档说明。

        参数:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
            要处理的单张或多张图像。每张图像可以是 PIL 图像、NumPy 数组或 PyTorch 张量。支持通道优先和通道最后的格式。
            text (`str`, `List[str]`, `List[List[str]]`):
            要编码的单个或批量序列。每个序列可以是字符串或字符串列表（预分词字符串）。如果序列以字符串列表（预分词）形式提供，必须设置 `is_split_into_words=True`（以避免与批量序列的歧义）。
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
            要处理的单个或批量视频。每个视频可以是 4 维 NumPy 数组或 PyTorch 张量。

        返回:
            [`BatchFeature`]: 返回一个 [`BatchFeature`]，包含以下字段：

            - **input_ids** -- 要输入模型的 token id 列表。当 `text` 不为 None 时返回。
            - **attention_mask** -- 指定模型应关注哪些 token 的索引列表（当 `return_attention_mask=True` 或 *"attention_mask"* 在 `self.model_input_names` 中且 `text` 不为 None 时返回）。
            - **pixel_values** -- 要输入模型的像素值。当 `images` 不为 None 时返回。
            - **pixel_values_videos** -- 要输入模型的视频像素值。当 `videos` 不为 None 时返回。
            - **image_sizes** -- 每张图像的尺寸，用于去除填充。当 `images` 不为 None 时返回。
        """

        #合并默认值参数和用户提供的参数,确保每一种模态都有自己的配置项
        output_kwargs = self._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        #将文本标准化成列表模式,目的是为了统一处理(方便遍历和batch)
        #在这个位置就已经得到了文本输入
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        ###############################################
        #处理图片或者视频
        image_inputs = video_inputs = {}

        #处理图片
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            #返回结果是包括pixel_values,image_sizes,batch_num_images的字典

            batch_num_images = iter(image_inputs["batch_num_images"])
            image_sizes = iter(image_inputs["image_sizes"])
            height, width = get_image_size(
                to_numpy_array(image_inputs["pixel_values"][0][0]),
                channel_dim=output_kwargs["images_kwargs"].get("data_format"),
            )
            text, num_image_tokens = self._expand_image_tokens(
                text, image_sizes, height, width, self.image_token, batch_num_images
            )

        if videos is not None:
            video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])

            one_video = video_inputs.get("pixel_values_videos")[0]
            if isinstance(video_inputs.get("pixel_values_videos")[0], (list, tuple)):
                one_video = np.array(one_video)
            else:
                one_video = to_numpy_array(one_video)
            height, width = get_image_size(one_video[0], channel_dim=output_kwargs["images_kwargs"].get("data_format"))
            num_frames = one_video.shape[0]  # frame dim is always after batch dim
            patches_height_width = int(math.sqrt(self.num_image_tokens))
            pooled_height_width = math.ceil(patches_height_width / 2)
            num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline token
            #扩展token?
            text = [sample.replace(self.video_token, self.video_token * num_video_tokens) for sample in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        
        ###得到了含有图像/视频占位符的文本嵌入 这个时候已经是embedding向量了
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        #检查特殊多模态是否存在
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        #创建一个与 input_ids 同形状的数组 mm_token_type_ids
        #所有对应位置上是 <image> token 的地方设为 1,其余为 0
        #这个字段可用于模型区分哪些 token 是图像 token,便于后续注意力机制处理
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        #整合所有模态的输出
        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=return_tensors)

    #根据图像的实际大小,动态扩展IMG token的数量
    def _expand_image_tokens(
        self,
        text: List[TextInput], #此时传入的text是一个纯文本的列表
        image_sizes: Iterable[Union[List[int], int]],
        height: int,
        width: int,#height和weight是图像分辨率,是模型需要的大小
        special_token: str,#特殊标记,IMG
        batch_num_images: Iterable[int], #每个样本包括多少图片
    ):
        prompt_strings = [] #存储替换后的文本
        max_num_vision_tokens = 0 #记录所有样本中最大视觉 token 数量，用于后续填充对齐。
        
        for sample in text:
            if special_token in sample:#判断是否有图像输入
                is_multi_image = next(batch_num_images) != 1 #判断是否为多图输入
            else:
                is_multi_image = False
            
            while special_token in sample:#循环替换文本中的所有image占位符
                if is_multi_image: #每张图使用固定的token数量+1,要有一个 newline token
                    num_image_tokens = self.num_image_tokens + 1  # one for image_newline
                
                #单图多图处理方式不同：
                else:#单图情况,利用原始图像尺寸去计算token数量
                    original_size = next(image_sizes)
                    if not isinstance(original_size, (list, tuple)):
                        # cast to list to avoid numerical precision errors when calculating unpadding
                        original_size = original_size.tolist() #为了数据安全,先将original_size转化为Python List
                    orig_height, orig_width = original_size #得到原始的图像高和宽拿来计算patch数量
                    #计算得到要将一个image占位符替换成多少个占位符
                    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                #记录所有样本中的最大token数量,为统一长度做处理
                max_num_vision_tokens = max(max_num_vision_tokens, num_image_tokens)
                #某些模型配置要求少一个 token(如不使用 CLS token),这里做相应调整。
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                
                #将第一个出现的 <image> 替换为对应数量的 <placeholder> 
                sample = sample.replace(special_token, "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        #之前用placeholder代替IMG,现在又把placeholder替换回IMG
        text = [sample.replace("<placeholder>", special_token) for sample in prompt_strings]
        return text, max_num_vision_tokens

    #批量计算每个模特需要的token数量,用于后续模型构建 attention mask、位置编码等
    #参数中,orig_height是图像的维度，height是指的是模型需要的维度
    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    #根据图像长宽比,计算实际使用的token数量,避免填充影响特征数量
    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        max_num_patches = int(self.vision_aspect_ratio.strip("anyres_max_"))
        ratio = math.sqrt(current_height * current_width / (max_num_patches * patches_height**2))
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(current_width // ratio)
            newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    #批量计算每个模特需要的token数量,用于后续模型构建 attention mask、位置编码等
    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (List[List[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (List[List[str]], *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
            audio_lengths (List[int], *optional*):
                The input length formatted as per each audio.
        Returns:
            Dict[str, List[int]]: A dictionary mapping each modality ("image", "video", "audio")
            to a list containing the number of placeholder tokens required. If the model doesn't accept
            a certain modality or no input sizes are provided, the dict value is set to an empty list.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LlavaOnevisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            size = (
                (size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"]))
            )
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)  # llava-ov doesn't batch pixels as Idefics, thus `1` patch`
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, processed_height, processed_width
                )
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                batch_num_image_tokens.append(num_image_tokens)
            vision_data.update({"num_image_tokens": batch_num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    #用于解码模型输出的 token IDs 回人类可读文本
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["LlavaOnevisionProcessor"]

