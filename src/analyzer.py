from typing import Dict, List
import os
import json
from api_client import ModelClient
from tqdm import tqdm
import json_repair

class ImageAnalyzer:
    def __init__(self, client: ModelClient):
        self.client = client
        
    def analyze_images(self, image_dir: str, show_progress=False) -> Dict[str, Dict]:
        """分析目录下所有图片"""
        image_infos = {}
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
        
        iterator = tqdm(image_files) if show_progress else image_files
        for image_file in iterator:
            if show_progress:
                iterator.set_description(f"正在分析: {image_file}")
            
            try:
                image_path = os.path.join(image_dir, image_file)
                # 读取图片数据
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                # 调用模型分析，传入完整的图片路径
                result = self.client.analyze_image(image_data, image_path=image_path)
                
                # 解析JSON响应
                if isinstance(result, str):
                    try:
                        result = json_repair.loads(result)
                    except json.JSONDecodeError:
                        result = {"devices": [], "personnel": []}

                # 添加 analysis 层级
                result = {"analysis": result}
                # 始终添加 image_path，即使是 None
                result["image_path"] = image_path
                
                image_infos[image_file] = result.get('analysis', {})
                
                if show_progress:
                    devices_count = len(result.get('analysis', {}).get('devices', []))
                    personnel_count = len(result.get('analysis', {}).get('personnel', []))
                    iterator.set_postfix(
                        status="成功",
                        devices=devices_count,
                        personnel=personnel_count
                    )
            except Exception as e:
                if show_progress:
                    iterator.set_postfix(status="失败")
                print(f"分析图片 {image_file} 时发生错误: {str(e)}")
                
        return image_infos
        
    def create_reference_pairs(self, image_infos: Dict) -> List[tuple]:
        """创建参照图对"""
        # 使用Qwen分析并配对
        pairs = self.client.find_matching_pairs(image_infos)
        return pairs 