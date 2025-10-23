import os
import requests
import time
from requests.exceptions import RequestException, Timeout, ConnectionError

# 模型服务配置
REWARD_MODEL_SERVICE_URL = os.getenv("REWARD_MODEL_SERVICE_URL", "http://localhost:10086")

def model_based_verify(item_str: str, item_title: str, max_retries: int = 3) -> float:
    """使用模型服务进行答案验证，包含重试机制"""
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{REWARD_MODEL_SERVICE_URL}/score",
                json={
                    "chatbot_response": item_str,
                    "real_item_title": item_title
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # print(f"模型服务调用成功，尝试次数: {attempt + 1}")
                # 返回置信度加权的结果
                return result["normalized_score"]
            else:
                print(f"模型服务请求失败 (尝试 {attempt + 1}): {response.status_code}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                return 0.0
            
        except (ConnectionError, Timeout) as e:
            print(f"网络错误 (尝试 {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            print(f"所有 {max_retries + 1} 次尝试都失败了")
            return 0.0
        except RequestException as e:
            print(f"请求异常 (尝试 {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            return 0.0
        except Exception as e:
            print(f"模型验证过程中发生错误 (尝试 {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            return 0.0
    
    return 0.0