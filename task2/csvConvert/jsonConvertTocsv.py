import json
import csv
import re
from json.decoder import JSONDecodeError


def fix_broken_json(json_str):
    # Change non english character to english
    json_str = (json_str
                .replace('“', '"').replace('”', '"')
                .replace('‘', "'").replace('’', "'")
                .replace('，', ',').replace('：', ':'))
    
    # Fix missing start or stop character
    json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
    json_str = re.sub(r':\s*([^"\s]+)(\s*[,}])', r': "\1"\2', json_str)
    json_str = re.sub(r'}\s*{', '},{', json_str)
    
    # Fix URL issue
    json_str = re.sub(r'(https?://[^\s"]+)(\s+)', 
                    lambda m: m.group(1).replace(' ', '%20'), json_str)
    
    json_str = re.sub(r'}\s*([^\]\s])', r'},\1', json_str)
    
    return json_str


def validate_structure(data):
    # Valid the structure and clean the data in json
    valid_records = []
    required_keys = {'content', 'url'}
    
    for item in data:
        if isinstance(item, dict) and all(key in item for key in required_keys):
            item['url'] = re.sub(r'\?.*$', '', item['url'].strip())
            valid_records.append({
                'content': str(item.get('content', '')).strip(),
                'url': item['url']
            })
    return valid_records

def json_to_csv(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        for attempt in range(3):
            try:
                data = json.loads(raw_data)
                break
            except JSONDecodeError as e:
                raw_data = fix_broken_json(raw_data)
        else:
            raise ValueError("Cannot fix Json file in 3 attempt.")

        cleaned_data = validate_structure(data)
        
        # Write cleaned data to csv file
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['content', 'url'])
            writer.writeheader()
            writer.writerows(cleaned_data)
            
        print(f"Conversion Successful：{len(cleaned_data)} data stored into {output_file}")

    except Exception as e:
        print(f"Error Detail：{str(e)}")
        print("Manual Fix Recommend：")
        print("1. Check the comma near line 10")
        print("2. Check quotation mark")
        print("3. Check all data were separate use comma")
        print("4. Check space in URL")

# Run example
if __name__ == "__main__":
    input_file = "xiaohongshu_data.json"
    output_file = "xiaohongshu_data.csv"
    json_to_csv(input_file, output_file)
