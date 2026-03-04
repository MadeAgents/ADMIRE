# Copyright 2026 OPPO
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
from openai import OpenAI
import base64
import io
from PIL import Image
from transformers import AutoProcessor

from utils import smart_resize_image

processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-72B-Instruct')

class OpenClient:
    def __init__(self, address, model_name, api_key="EMPTY"):
        openai_api_key = api_key
        openai_api_base = address
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = model_name

    def get_completion(self, user_prompt, system_prompt=None, image_paths=[], max_pixels=None, max_tokens=1024, temperature=0.0, frequency_penalty=1, stop=[], resize_method='qwen'):
        if max_pixels is not None:
            encoded_images = [self.get_resized_and_encoded_image(img_path, max_pixels=max_pixels, method=resize_method) for img_path in image_paths]
        else:
            encoded_images = [self.get_encoded_image(img_path) for img_path in image_paths]

        system_prompt = system_prompt or "You are a helpful assistant."

        messages = [
            {
                "role": "system", "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    *({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image;base64,{img}"
                        },
                    } for img in encoded_images),
                    {
                        "type": "text", 
                        "text": user_prompt
                    }
                ],
            },
        ]

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            frequency_penalty=frequency_penalty
        )
        return chat_response.choices[0].message.content

    def get_encoded_image(self, image_path):
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        return encoded_image_text

    def get_resized_and_encoded_image(self, image_path, min_pixels=None, max_pixels=None, method='qwen'):
        if not min_pixels and not max_pixels:
            return self.get_encoded_image(image_path)

        factor = processor.image_processor.patch_size * processor.image_processor.merge_size
        min_pixels = min_pixels or processor.image_processor.min_pixels
        max_pixels = max_pixels or processor.image_processor.max_pixels
        
        with Image.open(image_path) as image:
            image = smart_resize_image(image, min_pixels=min_pixels, max_pixels=max_pixels, factor=factor, method=method)
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                output.seek(0)
                encoded_image = base64.b64encode(output.read())
        encoded_image_text = encoded_image.decode("utf-8")
        return encoded_image_text