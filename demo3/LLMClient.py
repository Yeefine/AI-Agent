import os
from openai import OpenAI
from typing import List, Dict
import json

try:
    with open("api_params.json", "r", encoding="utf-8") as f:
        params_dict = json.load(f)  # 解析为字典
except (FileNotFoundError, json.JSONDecodeError):
    print("参数文件不存在或格式错误！")
    params_dict = {}

# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = params_dict.get("API_KEY")
BASE_URL = params_dict.get("BASE_URL")
MODEL_ID = params_dict.get("MODEL_ID")
TAVILY_API_KEY = params_dict.get("TAVILY_API_KEY")
# print(f"{API_KEY}, {BASE_URL}, {MODEL_ID}")
os.environ['API_KEY'] = params_dict.get("API_KEY")
os.environ['BASE_URL'] = params_dict.get("BASE_URL")
os.environ['MODEL_ID'] = params_dict.get("MODEL_ID")

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("MODEL_ID")
        apiKey = apiKey or os.getenv("API_KEY")
        baseUrl = baseUrl or os.getenv("BASE_URL")
        timeout = timeout or int(os.getenv("TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)


# 示例输出已移除，避免在源文件中包含非代码文本或非法字符。
