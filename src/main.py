from api_client import ModelClient
from analyzer import ImageAnalyzer
import os
import shutil
import json
import json_repair
def main():
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
        image_infos = analyzer.analyze_images(raw_dir)
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
        for i, (pair_info) in enumerate(pairs):
            ref_name = pair_info[0].strip('[]').split(']')[0]
            test_name = pair_info[1].strip('[]').split('：')[0]
            pair_reason = pair_info[1].split('：')[1] if len(pair_info[1].split('：')) > 1 else ""

            # 使用字典格式的提示词
            prompt = f"""请根据以下场景生成一个关于两张图片对比的问答对，直接输出JSON格式：
                        {{
                            "question": "您的问题",
                            "answer": "您的回答"
                        }}

                        场景：这是两张需要进行对比分析的图片，其中存在以下情况：
                        {pair_reason}

                        注意：
                        - 问题格式：先描述图1中的情况，然后询问图2中的具体情况
                        - 问题示例："图1中，[描述图1的情况]。请分析图2中[询问具体需要对比的内容]？"
                        - 回答需要：
                            清晰指出图2与图1的差异
                        - 如果提供的场景信息不足以支持标准的问答格式，可以灵活调整，但应保持专业性和对比分析的核心目的"""
            
            # 修改 API 调用方式
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
            
            # 解析JSON响应
            generated_qa = response.choices[0].message.content.strip()
            try:
                qa_dict = json_repair.loads(generated_qa)
                question = qa_dict.get('question', '解析失败')
                answer = qa_dict.get('answer', '解析失败')
            except Exception as e:
                print(f"解析失败: {str(e)}, 原文: {generated_qa}")
                question = "解析失败"
                answer = "解析失败"

            data = {
                "id": f"pair_{i}",
                "image": [
                    os.path.join(raw_dir, ref_name),
                    os.path.join(raw_dir, test_name)
                ],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image><image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
                "pair_reason": pair_reason
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

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