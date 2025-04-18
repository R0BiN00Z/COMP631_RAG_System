import json
import sys
import re

def is_chinese(text):
    # 检查文本是否包含中文字符
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def analyze_json_file(file_path):
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据类型
        print(f"数据类型: {type(data)}")
        
        # 如果是列表，显示长度和第一个元素的结构
        if isinstance(data, list):
            print(f"数据条目数: {len(data)}")
            if data:
                print("\n第一个条目的结构:")
                first_item = data[0]
                print(json.dumps(first_item, ensure_ascii=False, indent=2))
                
                # 分析所有条目的字段
                all_keys = set()
                for item in data:
                    all_keys.update(item.keys())
                print(f"\n所有字段: {sorted(all_keys)}")
                
                # 统计每个字段的存在情况
                print("\n字段统计:")
                for key in sorted(all_keys):
                    count = sum(1 for item in data if key in item)
                    print(f"{key}: {count} 条数据 ({count/len(data)*100:.2f}%)")
            
            # 分析语言分布
            chinese_titles = 0
            chinese_contents = 0
            mixed_titles = 0
            mixed_contents = 0
            
            for item in data:
                title = item.get('title', '')
                content = item.get('content', '')
                
                # 分析标题
                if is_chinese(title):
                    chinese_titles += 1
                elif any(is_chinese(c) for c in title):
                    mixed_titles += 1
                
                # 分析内容
                if is_chinese(content):
                    chinese_contents += 1
                elif any(is_chinese(c) for c in content):
                    mixed_contents += 1
            
            # 打印统计结果
            print("\n语言分布统计:")
            print(f"纯中文标题: {chinese_titles} ({chinese_titles/len(data)*100:.2f}%)")
            print(f"混合语言标题: {mixed_titles} ({mixed_titles/len(data)*100:.2f}%)")
            print(f"纯中文内容: {chinese_contents} ({chinese_contents/len(data)*100:.2f}%)")
            print(f"混合语言内容: {mixed_contents} ({mixed_contents/len(data)*100:.2f}%)")
            
            # 显示一些示例
            print("\n示例数据:")
            for i, item in enumerate(data[:3]):
                print(f"\n示例 {i+1}:")
                print(f"标题: {item.get('title', '')[:100]}...")
                print(f"内容: {item.get('content', '')[:200]}...")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "merged_data.json"
    analyze_json_file(file_path) 