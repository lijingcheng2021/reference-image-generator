from api_client import ModelClient
from analyzer import ImageAnalyzer
import os
import shutil
import json
import json_repair
def main():
    # 配置路径
    base_dir = "/data1/ljc/code/application-data-generation"
    raw_dir = os.path.join(base_dir, "data/images/v0_100")
    # 更新输出目录路径
    dataset_dir = "/data1/ljc/项目/参照图数据生成/data/reference_dataset"
    # 输入 JSONL 文件路径
    jsonl_file = os.path.join(base_dir, "data/sample_annotation/v0_100/converted_annotations.jsonl")
    # 更新输出文件路径
    multimodal_file = os.path.join(dataset_dir, "multimodal_data.jsonl")
    
    # 确保输出目录存在
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 读取注释文件
    image_annotations = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 构建完整图片路径
            full_image_path = os.path.join(base_dir, data['image'])
            image_annotations[os.path.basename(data['image'])] = {
                'objects': data['objects'],
                'scene': data['scene'],
                'anomaly': data['anomaly'],
                'image_path': full_image_path
            }

    #try:
        # 配置
    raw_dir = "/data1/ljc/code/application-data-generation/data/images/v0_100"
    dataset_dir = "/data1/ljc/项目/参照图数据生成/data/reference_dataset/v0"
    multimodal_file = os.path.join(dataset_dir, "multimodal_data.jsonl")
    
    # 初始化API客户端
    client = ModelClient(
        api_key='YOUR_API_KEY',
        base_url='http://140.207.201.5:60070/v1'
    )
    analyzer = ImageAnalyzer(client)
    
    # 分析所有图片
    print("开始分析图片...")
    image_infos = analyzer.analyze_images(raw_dir, annotations=image_annotations)
    if not image_infos:
        raise Exception("没有成功分析任何图片")
        
    # 创建参照图对
    print("开始创建参照图对...")
    pairs = analyzer.create_reference_pairs(image_infos)
    if not pairs:
        raise Exception("没有找到合适的参照图对")
        
    # 生成多模态训练数据
    print(f"开始生成多模态训练数据到 {multimodal_file}")
    generate_multimodal_data(pairs, image_infos, raw_dir, multimodal_file, client)
    
    print("处理完成！")
        
    # except Exception as e:
    #     print(f"发生错误: {str(e)}")
    #     return

def generate_multimodal_data(pairs, image_infos, raw_dir, output_file, client):
    """生成多模态训练数据"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, pair_dict in enumerate(pairs):
            # 从字典中获取参考图和测试图的名称
            ref_name = pair_dict['reference']
            test_name = pair_dict['test']
            
            # 获取两张图片的分析信息
            ref_info = image_infos[ref_name]
            test_info = image_infos[test_name]
            
            # 如果已经有生成好的问答对，直接使用
            if 'qa_pair' in pair_dict:
                qa_pair = pair_dict['qa_pair']
            else:
                # 构建提示词，包含两张图片的详细分析信息
                prompt = f"""基于以下两张工地场景图片的分析信息，生成一个专业的问答对：

                        参考图信息：
                        {json.dumps(ref_info, ensure_ascii=False, indent=2)}

                        测试图信息：
                        {json.dumps(test_info, ensure_ascii=False, indent=2)}

                        请生成一个问答对，要求：
                        1. 问题应该从参考图的场景出发，询问测试图的相关情况
                        2. 回答应该详细对比两张图片中物体的异同
                        3. 回答要突出重点，使用专业的描述方式

                        请以下面的JSON格式返回：
                        {{
                            "question": "您的问题",
                            "answer": "您的回答"
                        }}"""
                
                # 调用模型生成问答对
                response = client.client.chat.completions.create(
                    model=client.model,
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
                
                try:
                    qa_pair = json.loads(response.choices[0].message.content)
                except json.JSONDecodeError as e:
                    print(f"解析问答对失败: {str(e)}")
                    continue

            try:
                # 构建完整的数据条目
                data_item = {
                    "id": f"pair_{i}",
                    "images": [
                        os.path.join(raw_dir, ref_name),
                        os.path.join(raw_dir, test_name)
                    ],
                    "conversations": [
                        {"from": "human", "value": qa_pair["question"]},
                        {"from": "assistant", "value": qa_pair["answer"]}
                    ]
                }
                # 写入JSONL文件
                f.write(json.dumps(data_item, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"生成数据条目失败: {str(e)}")
                continue

def _generate_comparison_text(ref_info, test_info):
    """根据两张图片的分析信息生成比较文本"""
    # Access the nested 'analysis' dictionary first
    ref_devices = ref_info['analysis'].get('devices', [])
    ref_personnel = ref_info['analysis'].get('personnel', [])
    test_devices = test_info['analysis'].get('devices', [])
    test_personnel = test_info['analysis'].get('personnel', [])
    
    return f"根据参考图的标准（设备：{', '.join(ref_devices)}；人员：{', '.join(ref_personnel)}），分析测试图中存在的问题（设备：{', '.join(test_devices)}；人员：{', '.join(test_personnel)}）"

if __name__ == "__main__":
    main() 