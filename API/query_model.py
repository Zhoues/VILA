from openai import OpenAI
import base64
import os
import time
import requests

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def query_gpt4o(image_paths, prompt, model_name='gpt-4o', retry=100):
    """
    Query the GPT-4 Vision model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    - retry: Integer, the number of retries.
    """
    base64_images = [encode_image(image_path) for image_path in image_paths]

    for r in range(retry):
        try:
            input_dicts = [{"type": "text", "text": prompt}]
            for i, image in enumerate(base64_images):
                input_dicts.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                    "role": "user",
                    "content": input_dicts,
                    }
                ],
                max_tokens=1024,
                n=1,
                temperature=0.0,
            )
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
    return 'Failed: Query Error'


def query_gpto1(image_paths, prompt, model_name='o1', retry=100):
    """
    Query the GPT-O1 model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    - model_name: String, the name of the model.
    - retry: Integer, the number of retries.
    """
    response = query_gpt4o(image_paths, prompt, model_name, retry)
    return response


def query_api(image_paths, prompt, model_name, retry=100):
    """
    Query the Gemini 2 Pro model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    base64_images = [encode_image(image_path) for image_path in image_paths]

    for r in range(retry):
        try:
            input_dicts = [{"type": "text", "text": prompt}]
            for i, image in enumerate(base64_images):
                input_dicts.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
                
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": input_dicts,
                    }
                ],
                "max_tokens": 1024,
                "n": 1,
                "temperature": 0.0,
                "user": "DMXAPI",
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "DMXAPI/1.0.0 (https://www.dmxapi.com/)",
            }

            response = requests.post(base_url, headers=headers, json=payload)
            print(response.json())
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            time.sleep(1)
    return 'Failed: Query Error'


def query_gemini_2_pro(image_paths, prompt, model_name='gemini-2.0-pro-exp-02-05', retry=100):
    """
    Query the Gemini 2 Pro model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    response = query_api(image_paths, prompt, model_name, retry)
    return response

def query_gemini_2_5_pro(image_paths, prompt, model_name='gemini-2.5-pro-exp-03-25', retry=100):
    """
    Query the Gemini 2.5 Pro model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    response = query_api(image_paths, prompt, model_name, retry)
    return response

def query_gemini_2_flash(image_paths, prompt, model_name='gemini-2.0-flash', retry=100):
    """
    Query the Gemini 2 Flash model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    response = query_api(image_paths, prompt, model_name, retry)
    return response
    

def query_claude_3_5_sonnet(image_paths, prompt, model_name='claude-3-5-sonnet-20240620', retry=100):
    """
    Query the Claude 3.5 Sonnet model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    response = query_api(image_paths, prompt, model_name, retry)
    return response

def query_claude_3_7_sonnet(image_paths, prompt, model_name='claude-3-7-sonnet-20250219', retry=100):
    """
    Query the Claude 3.7 Sonnet model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    response = query_api(image_paths, prompt, model_name, retry)
    return response
    

def query_qwen2_5_vl_7b(image_paths, prompt, model_name='qwen2.5-vl-7b-instruct', retry=100):
    """
    Query the Claude 3.7 Sonnet model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    response = query_api(image_paths, prompt, model_name, retry)
    return response


def query_qwen2_5_vl_72b(image_paths, prompt, model_name='qwen2.5-vl-72b-instruct', retry=100):
    """
    Query the Claude 3.7 Sonnet model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    response = query_api(image_paths, prompt, model_name, retry)
    return response

def query_server(image_paths, prompt, url="http://127.0.0.1:25547", enable_depth=0, depth_paths=None, retry=3):
    """
    Query the server with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    - url: String, the url of the server.
    - depth_path: String, the path to the depth image.
    - enable_depth: Integer, 0 or 1, whether to enable depth.
    - retry: Integer, the number of retries.
    """

    # 1. 将 image_paths 转为 base64
    image_url_list = []
    for path in image_paths:
        image_url_list.append(encode_image(path))

    # 2. 处理 depth_path（如果传入）
    depth_url_list = []
    if depth_paths is not None:
        # 如果 depth_path 是单个字符串，可以直接转换
        if isinstance(depth_paths, str):
            depth_url_list = [encode_image(depth_paths)]
        # 如果 depth_path 是多个深度图组成的列表，需要与 image_paths 一一对应
        elif isinstance(depth_paths, list):
            depth_url_list = [encode_image(dpth) for dpth in depth_paths]
        else:
            raise ValueError("depth_path 参数必须为字符串或字符串列表")

    # 3. 构造请求体
    request_data = {
        "image_url": image_url_list,       # base64 编码后的图像列表
        "depth_url": depth_url_list,       # base64 编码后的深度图列表（可能为空）
        "enable_depth": enable_depth,      # 是否开启深度图模式
        "text": prompt                     # 用户文本
    }

    # 4. 进行 HTTP 请求，并支持一定次数的重试
    for attempt in range(1, retry + 1):
        try:
            response = requests.post(url + "/query", json=request_data)
            if response.status_code == 200:
                # 解析返回的 JSON 数据
                try:
                    response_content = response.json()
                except ValueError:
                    print(f"[Error] 第 {attempt} 次请求返回的数据无法解析为 JSON。")
                    response_content = None
                print(response_content)
                return response_content["answer"]
            else:
                print(f"[Warning] 第 {attempt} 次请求返回状态码 {response.status_code}，将在一秒后重试。")
        except requests.exceptions.RequestException as e:
            print(f"[Error] 第 {attempt} 次请求出现异常: {e}")

    # 如果多次重试仍然失败，返回 None
    print("[Fatal] 请求失败，超过最大重试次数。")
    return None