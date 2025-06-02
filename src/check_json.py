import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple
import argparse

def check_single_json(filepath: str) -> Tuple[str, str]:
    """
    检查单个JSON文件的有效性
    
    参数:
        filepath: 文件完整路径
        
    返回:
        Tuple[str, str]: (文件路径, 错误信息) 如果文件正常则返回(None, None)
    """
    try:
        # 检查文件是否为空
        if os.path.getsize(filepath) == 0:
            return (filepath, "空文件")
        
        # 尝试解析JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
            
        return (None, None)
    
    except json.JSONDecodeError as e:
        return (filepath, f"无效JSON - 错误: {str(e)}")
    except UnicodeDecodeError as e:
        return (filepath, f"编码错误 - 错误: {str(e)}")
    except Exception as e:
        return (filepath, f"读取失败 - 错误: {str(e)}")

def check_json_files(directory: str, max_workers: int = 8) -> List[str]:
    """
    使用多线程检查目录下的所有JSON文件
    
    参数:
        directory: 要检查的目录路径
        max_workers: 最大线程数
        
    返回:
        List[str]: 有问题的文件路径及错误信息列表
    """
    problematic_files = []
    
    # 确保目录存在
    if not os.path.isdir(directory):
        raise ValueError(f"目录不存在: {directory}")
    
    # 收集所有JSON文件路径
    json_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.json'):
                json_files.append(os.path.join(root, filename))
    
    # 使用线程池并行检查
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(check_single_json, json_files)
        
        for filepath, error in results:
            if filepath is not None:
                problematic_files.append(f"{error}: {filepath}")
    
    return problematic_files

def main():
    parser = argparse.ArgumentParser(description='多线程检查JSON文件有效性')
    parser.add_argument('directory', type=str, help='要检查的目录路径')
    parser.add_argument('--workers', type=int, default=8, 
                       help='最大线程数 (默认: 8)')
    args = parser.parse_args()
    
    try:
        problems = check_json_files(args.directory, args.workers)
        
        if problems:
            print("发现以下有问题的文件:")
            for problem in problems:
                print(problem)
            print(f"\n总计: {len(problems)} 个问题")
        else:
            print("所有JSON文件检查通过，没有发现问题")
            
        return problems
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return []

if __name__ == '__main__':
    main()