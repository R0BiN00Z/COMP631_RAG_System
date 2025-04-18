import json
import sys
import re

# Check whether the text contain Chinese character or not
def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def analyze_json_file(file_path):
    try:
        # Read JSOM File
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Data Type: {type(data)}")        # Check the data Type
        
        # If the data type is any sort of list, show the list length and the first element in that list
        if isinstance(data, list):
            print(f"Data Num Count: {len(data)}")
            if data:
                print("\nThe first element structure:")
                first_item = data[0]
                print(json.dumps(first_item, ensure_ascii=False, indent=2))
                
                # Analyze all element
                all_keys = set()
                for item in data:
                    all_keys.update(item.keys())
                print(f"\nAll Contents: {sorted(all_keys)}")
                
                # Summarize the status of element
                print("\nElement Analyze:")
                for key in sorted(all_keys):
                    count = sum(1 for item in data if key in item)
                    print(f"{key}: {count} of Elements ({count/len(data)*100:.2f}%)")
            
            # Analyze Language Distribution
            chinese_titles = 0
            chinese_contents = 0
            mixed_titles = 0
            mixed_contents = 0
            
            for item in data:
                title = item.get('title', '')
                content = item.get('content', '')
                
                # Analyze Title
                if is_chinese(title):
                    chinese_titles += 1
                elif any(is_chinese(c) for c in title):
                    mixed_titles += 1
                
                # Analyze content
                if is_chinese(content):
                    chinese_contents += 1
                elif any(is_chinese(c) for c in content):
                    mixed_contents += 1
            
            # Print the Analyze Result
            print("\nContent Language Distribution Analyze:")
            print(f"Full Chinese Title: {chinese_titles} ({chinese_titles/len(data)*100:.2f}%)")
            print(f"Mixed Language Title: {mixed_titles} ({mixed_titles/len(data)*100:.2f}%)")
            print(f"Full Chinese Content: {chinese_contents} ({chinese_contents/len(data)*100:.2f}%)")
            print(f"Mixed Language Content: {mixed_contents} ({mixed_contents/len(data)*100:.2f}%)")
            
            # Show some sample data
            print("\nSample Data:")
            for i, item in enumerate(data[:3]):
                print(f"\nSample {i+1}:")
                print(f"Title: {item.get('title', '')[:100]}...")
                print(f"Content: {item.get('content', '')[:200]}...")
        
    except Exception as e:
        print(f"Error during Analyze: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "merged_data.json"
    analyze_json_file(file_path) 
