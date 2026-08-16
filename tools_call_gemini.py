import os
import json
import requests
import random
import time



API_POOL = {
    index: os.environ.get(f"GEMINI_API_KEY_{index}", "")
    for index in range(2, 7)
}

_FALLBACK_KEY = os.environ.get("GEMINI_API_KEY", "")
if _FALLBACK_KEY:
    for key in API_POOL:
        if not API_POOL[key]:
            API_POOL[key] = _FALLBACK_KEY

API_POOL = {key: value for key, value in API_POOL.items() if value}




def get_random_api():
    if not API_POOL:
        raise RuntimeError(
            "??? Gemini API Key???????? GEMINI_API_KEY "
            "? GEMINI_API_KEY_2 ~ GEMINI_API_KEY_6"
        )
    return random.choice(list(API_POOL.values()))


def access_gemini(text, max_retry=3, model_name="gemini-2.5-pro"):
    original_model = model_name
    
    for phase in range(2):  # 两个阶段：原始模型阶段 + 降级阶段
        for attempt in range(max_retry):
            apikey = get_random_api()
            try:
                # 动态构建模型URL
                url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={apikey}'
                headers = {'Content-Type': 'application/json'}
                data = {
                    'contents': [{
                        'parts': [{'text': text}]
                    }]
                }
                
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=300
                )
                response.raise_for_status()  # 检查HTTP错误
                
                # 尝试解析JSON响应
                response_json = response.json()
                response_txt = response_json['candidates'][0]['content']['parts'][0]['text']
                
                return response_txt
            
            except Exception as e:
                last_error = e
                print(f"当前第 {attempt+1} 请求失败, 错误模型:{model_name}, 错误原因: {str(e)}")
                if attempt < max_retry - 1:
                    time.sleep(2 ** attempt)  # 指数退避等待
                continue
        
        # 如果当前阶段失败，检查是否需要降级
        if phase == 0 and model_name == "gemini-2.5-pro":
            print(f"原始的模型：{original_model} 请求达到最大尝试次数, 降级模型为 gemini-2.0-flash 进行最后尝试")
            model_name = "gemini-2.0-flash"  # 降级模型
            continue
        
        # 如果已经是降级模型或不需要降级，则返回错误
        error_msg = (f"全部请求失败, 本次请求使用的初始模型为: {original_model}, "
                   f"最终兜底模型为: {model_name}. 最后一次报错的错误信息为: {str(last_error)}")
        raise Exception(error_msg)


if __name__ == '__main__':

    text = '我现在需要写一篇小说，是中世纪的故事背景，但是需要有日系的一些风格，类似GALGAME的剧本创作。我需要你提供三个性格各异的女主角，给出其身份背景，性格，外貌，以及相关的语言风格。'
    print(access_gemini(text))
