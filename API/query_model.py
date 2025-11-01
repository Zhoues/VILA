from openai import OpenAI
import base64
import os
import time
import requests
import http.client
import json
base_url = "https://cnapi.kksj.org/v1/"
api_key ='sk-sZlrLladnwVvtgnURsz8LjQJXAFEgTkc7qcDg3mxC3plUcwH'
openai_client = OpenAI(api_key=api_key, base_url=base_url)
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
    
    # base64_images = [encode_image(image_path) for image_path in image_paths]

    # for r in range(retry):
    #     try:
    #         input_dicts = [{"type": "text", "text": prompt}]
    #         for i, image in enumerate(base64_images):
    #             input_dicts.append({"type": "image_url",
    #                                 "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
                
    #         payload = {
    #             "model": model_name,
    #             "messages": [
    #                 {
    #                     "role": "user",
    #                     "content": input_dicts,
    #                 }
    #             ],
    #             "max_tokens": 1024,
    #             "n": 1,
    #             "temperature": 0.0,
    #             "user": "DMXAPI",
    #         }
    #         headers = {
    #             "Content-Type": "application/json",
    #             "Authorization": f"Bearer {api_key}",
    #             "User-Agent": "DMXAPI/1.0.0 (https://www.dmxapi.com/)",
    #         }

    #         response = requests.post(base_url, headers=headers, json=payload)
    #         print(response.json())
    #         return response.json()["choices"][0]["message"]["content"]
    #     except Exception as e:
    #         print(e)
    #         time.sleep(1)
    # return 'Failed: Query Error'
    conn = http.client.HTTPSConnection("api.kksj.org")
    base64_images = [encode_image(image_path) for image_path in image_paths]
    for r in range(retry):
        try:
            input_dicts = [{"type": "text", "text": prompt}]
            for i, image in enumerate(base64_images):
                 input_dicts.append({"type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
            payload = json.dumps({
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
            })
            headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
            }
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            print(data.decode("utf-8"))
            return json.loads(data.decode("utf-8"))["choices"][0]["message"]["content"]
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

def query_gemini_2_5_pro(image_paths, prompt, model_name='gemini-2.5-pro-thinking', retry=100):
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

def query_server(image_paths, prompt, url="http://127.0.0.1:25547", enable_spatial=0, intrinsics=None, depth_z_paths=None, retry=3):
    """
    Query the server with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    - url: String, the url of the server.
    - intrinsics: List of Arrays, the intrinsics matrix.
    - depth_z_paths: List of Strings, the path to the depth image.
    - enable_spatial: Integer, 0 or 1, whether to enable spatial.
    - retry: Integer, the number of retries.
    """

    # 1. 构造请求体
    request_data = {
        "image_paths": image_paths,
        "depth_z_paths": depth_z_paths,
        "intrinsics": intrinsics,
        "enable_spatial": enable_spatial,  
        "text": prompt
    }

    # 2. 进行 HTTP 请求，并支持一定次数的重试
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