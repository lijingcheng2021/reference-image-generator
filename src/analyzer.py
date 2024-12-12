from typing import Dict, List
import os
import json
from api_client import ModelClient
from tqdm import tqdm
import json_repair
import base64

class ImageAnalyzer:
    def __init__(self, client: ModelClient):
        self.client = client
        
    def analyze_images(self, image_dir: str, annotations: Dict = None, show_progress=False) -> Dict[str, Dict]:
        """分析目录下所有图片
        Args:
            image_dir: 图片目录路径
            annotations: 图片标注数据字典，格式为 {image_name: annotation_data}
            show_progress: 是否显示进度条
        """
        image_infos = {}
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
        
        iterator = tqdm(image_files) if show_progress else image_files
        for image_file in iterator:
            if show_progress:
                iterator.set_description(f"正在分析: {image_file}")
            
            try:
                if annotations and image_file in annotations:
                    # 获取标注数据中的 objects
                    objects = annotations[image_file].get('objects', [])
                    
                    # 读取图片数据
                    image_path = os.path.join(image_dir, image_file)
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    
                    # 使用模型生成场景描述
                    description = self._generate_scene_description(objects, image_data)
                    
                    # 保存标注数据和生成的描述
                    image_infos[image_file] = {
                        'annotation': annotations[image_file],
                        'description': description
                    }
                
                if show_progress:
                    objects_count = len(annotations.get(image_file, {}).get('objects', []))
                    iterator.set_postfix(
                        status="成功",
                        objects=objects_count
                    )
            except Exception as e:
                if show_progress:
                    iterator.set_postfix(status="失败")
                print(f"处理图片 {image_file} 时发生错误: {str(e)}")
        
        return image_infos
    
    def _generate_scene_description(self, objects: List[str], image_data: bytes) -> str:
        """根据��和标注的物体列表生成场景描述
        Args:
            objects: 标注的物体列表
            image_data: 图片二进制数据
        """
        prompt = f"""对以下标注物体描述其颜色、外观和行为，并以JSON格式返回：

        物体列表：{', '.join(objects)}

        返回格式示例：
        {{
            "塔吊": "黄色高大塔吊，正在进行钢构件吊装",
            "脚手架": "银灰色金属脚手架，已完成搭设并固定",
            "安全网": "绿色密织安全网，完整覆盖外立面"
        }}

        要求：
        1. 使用简洁的专业用语描述外观和行为
        2. 只返回JSON格式数据
        """
        
        try:
            response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=[{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': prompt,
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}",
                        }
                    }],
                }],
                temperature=0.7,
                top_p=0.7
            )
            
            return json_repair.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"生成场景描述失败: {str(e)}")
            return "场景描述生成失败"
        
    def create_reference_pairs(self, image_infos: Dict) -> List[Dict]:
        """创建参照图对并生成问答"""
        pairs_with_qa = []
        image_names = list(image_infos.keys())
        
        # 两两配对遍历
        for i in range(len(image_names)):
            for j in range(i + 1, len(image_names)):
                ref_name = image_names[i]
                test_name = image_names[j]
                
                # 判断两张图片是否适合配对
                is_match = self._check_pair_match(
                    ref_name, 
                    test_name,
                    image_infos[ref_name],
                    image_infos[test_name]
                )
                
                if is_match:
                    # 如果适合配对，则生成问答对
                    qa_pair = self._generate_qa_pair(
                        ref_name, 
                        test_name,
                        image_infos[ref_name],
                        image_infos[test_name]
                    )
                    
                    if qa_pair:  # 如果成功生成问答对
                        pairs_with_qa.append({
                            "reference": ref_name,
                            "test": test_name,
                            "qa_pair": qa_pair
                        })
        
        return pairs_with_qa

    def _check_pair_match(self, ref_name: str, test_name: str, 
                         ref_info: Dict, test_info: Dict) -> bool:
        """使用模型判断两张图片是否适合配对"""
        prompt = f"""请分析这两张图片是否适合作为参照图对进行配对：

                图1信息：
                {json.dumps(ref_info['annotation']['objects'], ensure_ascii=False, indent=2)}

                图2信息：
                {json.dumps(test_info['annotation']['objects'], ensure_ascii=False, indent=2)}

                配对原则：
                1. 两张图片应该包含相似的主要物体

                请直接返回 "是" 或 "否"，表示是否适合配对。"""
        
        try:
            response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
                temperature=0.2,  # 降低温度以获得更确定的答案
                top_p=0.1
            )
            
            result = response.choices[0].message.content.strip()
            return result == "是"
        except Exception as e:
            print(f"判断配对失败: {str(e)}")
            return False

    def _generate_qa_pair(self, ref_name: str, test_name: str, 
                         ref_info: Dict, test_info: Dict) -> Dict:
        """生成问答对"""
        prompt = f"""基于以下两张图片的分析信息，生成一个问答对：

                图1信息：
                {json.dumps(ref_info["description"], ensure_ascii=False, indent=2)}

                图2信息：
                {json.dumps(test_info["description"], ensure_ascii=False, indent=2)}

                请生成一个问答对，要求：
                对比图1和图2的信息，对图2进行提问，然后对应回答问题

                请以下面的JSON格式返回：
                {{
                    "question": "您的问题",
                    "answer": "您的回答"
                }}"""
        
        try:
            response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
                temperature=0.8,
                top_p=0.8
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"生成问答对失败: {str(e)}")
            return None