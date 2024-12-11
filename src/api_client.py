import base64
import json
from openai import OpenAI
from typing import Dict, List
import json_repair

class ModelClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.client.models.list().data[0].id
        
    def extract_image_info(self, image_data: bytes) -> Dict:
        """使用QwenVL提取图片信息"""
        # 直接使用传入的图片数据进行base64编码
        encoded_image = base64.b64encode(image_data).decode("utf-8")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': '请分析图片中的：1.设备信息 2.人员穿戴 3.人员行为',
                }, {
                    'type': 'image_url',
                    'image_url': {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    }
                }],
            }],
            temperature=0.8,
            top_p=0.8
        )
        
        return response.choices[0].message.content
    
    def find_matching_pairs(self, image_infos: List[Dict]) -> List[tuple]:
        """使用Qwen分析图片信息并找出合适的配对"""
        prompt = self._create_matching_prompt(image_infos)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }],
            }],
            temperature=0.8,
            top_p=0.8
        )
        
        return self._parse_matching_response(response.choices[0].message.content)
    
    def _encode_image(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    def _create_matching_prompt(self, image_infos: Dict[str, Dict]) -> str:
        """创建配对提示语
        Args:
            image_infos: 包含图片分析结果的字典，格式为 {image_name: {"analysis": {"devices": [...], "personnel": [...]}}
        """
        info_list = []
        for img_name, info in image_infos.items():
            analysis = info.get('analysis', {})
            devices = analysis.get('devices', [])
            personnel = analysis.get('personnel', [])
            info_text = f"图片名称：{img_name}\n设备：{', '.join(devices)}\n人员：{', '.join(personnel)}"
            info_list.append(info_text)
        
        all_info = "\n\n".join(info_list)
        
        return f"""分析以下工地场景图片，找出适合作为配对的图片：
                
                配对原则：
                   具备可比性。
                   比如 图1中有人戴安全帽，图2中有人没戴安全帽
                   比如 图1中有塔吊，图2中没有塔吊
                   
               
                
                图片信息如下：
                {all_info}
                
                请返回配对结果，格式：
                1. [图1文件名] - [图2文件名]：[配对原因，说明差异]
                2. [图1文件名] - [图2文件名]：[配对原因，说明差异]
                ...
                
                """
    
    def _parse_matching_response(self, response: str) -> List[tuple]:
        """解析配对响应"""
        pairs = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if '-' in line:  # 查找包含配对信息的行
                # 移除序号和空格
                pair_text = line.split('.', 1)[-1].strip()
                # 分割两个图片名
                img_names = pair_text.split('-')
                if len(img_names) == 2:
                    img1 = img_names[0].strip()
                    img2 = img_names[1].strip()
                    pairs.append((img1, img2))
        
        return pairs
    
    def analyze_image(self, image_data: bytes, image_path: str = None) -> Dict:
        """分析单张图片
        Args:
            image_data: 图片二进制数据
            image_path: 图片路径
        """
        structured_prompt = """请分析图片中的设备和人员情况，并以下面的JSON格式返回：
                                1. **devices** 数组：列出图片中所有可见的设备和机械，并进行简洁的描述，注意设备外观。
                                2. **personnel** 数组：描述每个人员的着装和行为，特别注意安全防护装备（如安全帽、安全绳、反光衣等）的佩戴情况。

                                返回结果格式的要求如下：
                                - devices：设备和机械的列表，每项为一条简单描述。
                                - personnel：人员的列表，每项为一条简单描述。

                                注意事项：
                                1. 仅输出根据图片生成的分析结果，勿包含示例内容。
                                2. 确保返回的 JSON 严格符合格式规范，以下为参考结构：

                                ```json
                                {
                                    "devices": ["设备或机械描述 1", "设备或机械描述 2", ...],
                                    "personnel": ["人员描述 1", "人员描述 2", ...]
                                }"""
        
        # 发送请求获取结构化输出
        structured_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': structured_prompt,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}",
                    }
                }],
            }],
            temperature=0.8,
            top_p=0.8
        )
        
        result = structured_response.choices[0].message.content
        
        # 确保返回的是字典格式
        if isinstance(result, str):
            try:
                result = json_repair.loads(result)
            except json.JSONDecodeError:
                result = {"devices": [], "personnel": []}
        
        # 构建标准的返回格式
        final_result = {
            "analysis": result,
            "image_path": image_path
        }
        
        return final_result
    
    def analyze_images(self, image_dir: str, show_progress=False) -> Dict[str, Dict]:
        image_infos = {}
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        iterator = tqdm(image_files) if show_progress else image_files
        for image_file in iterator:
            if show_progress:
                iterator.set_description(f"正在分析: {image_file}")
            
            try:
                image_path = os.path.join(image_dir, image_file)
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                result = self.client.analyze_image(image_data, image_path=image_path)
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
    
    def create_reference_pairs(self, image_infos: Dict[str, Dict]) -> List[tuple]:
        """创建参照图对"""
        pairs = self.find_matching_pairs(image_infos)
        
        # 过滤掉没有匹配的图片
        matched_pairs = []
        for img1, img2 in pairs:
            if img1 in image_infos and img2 in image_infos:
                matched_pairs.append((img1, img2))
            else:
                print(f"Skipping unmatched pair: {img1}, {img2}")
        
        return matched_pairs