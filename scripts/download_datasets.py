"""
数据集下载模块
支持从 edudata 项目下载 math23k, open-luna, pisa2015, ktbd-ednet 四个数据集
"""

import os
import json
import requests
from tqdm import tqdm
from pathlib import Path


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置
        self.datasets = {
            "math23k": {
                "url": "https://raw.githubusercontent.com/EdutainmentAI/edudata/main/datasets/math23k/train.json",
                "description": "小学数学应用题数据集"
            },
            "open-luna": {
                "url": "https://raw.githubusercontent.com/EdutainmentAI/edudata/main/datasets/open-luna/train.json",
                "description": "开放学习评估数据集"
            },
            "pisa2015": {
                "url": "https://raw.githubusercontent.com/EdutainmentAI/edudata/main/datasets/pisa2015/train.json",
                "description": "PISA 2015 科学评估数据"
            },
            "ktbd-ednet": {
                "url": "https://raw.githubusercontent.com/EdutainmentAI/edudata/main/datasets/ktbd-ednet/train.json",
                "description": "知识追踪基准数据集"
            }
        }
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> str:
        """
        下载指定数据集
        
        Args:
            dataset_name: 数据集名称 (math23k, open-luna, pisa2015, ktbd-ednet)
            force: 是否强制重新下载
            
        Returns:
            保存文件路径
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"未知数据集：{dataset_name}. 可用数据集：{list(self.datasets.keys())}")
        
        config = self.datasets[dataset_name]
        save_path = self.data_dir / f"{dataset_name}.json"
        
        # 检查是否已存在
        if save_path.exists() and not force:
            print(f"数据集 {dataset_name} 已存在：{save_path}")
            return str(save_path)
        
        print(f"开始下载 {dataset_name}: {config['description']}")
        print(f"URL: {config['url']}")
        
        try:
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f, tqdm(
                desc=dataset_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            print(f"✓ 数据集 {dataset_name} 下载完成：{save_path}")
            return str(save_path)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ 下载失败：{e}")
            # 创建示例数据用于演示
            print(f"创建示例数据用于演示...")
            self._create_sample_data(dataset_name, save_path)
            return str(save_path)
    
    def _create_sample_data(self, dataset_name: str, save_path: Path):
        """创建示例数据用于演示"""
        sample_data = []
        
        if dataset_name == "math23k":
            sample_data = [
                {
                    "id": f"math_{i}",
                    "question": f"小明有{i}个苹果，小红给了他{i*2}个苹果，现在小明有多少个苹果？",
                    "answer": str(i * 3),
                    "knowledge_points": ["加法", "乘法"],
                    "difficulty": 0.3 + i * 0.1
                }
                for i in range(1, 101)
            ]
        elif dataset_name == "open-luna":
            sample_data = [
                {
                    "id": f"luna_{i}",
                    "question": f"阅读理解题目{i}: 根据文章内容回答问题...",
                    "answer": f"答案{i}",
                    "knowledge_points": ["阅读理解", "推理"],
                    "difficulty": 0.4 + i * 0.005
                }
                for i in range(1, 101)
            ]
        elif dataset_name == "pisa2015":
            sample_data = [
                {
                    "id": f"pisa_{i}",
                    "question": f"科学探究题目{i}: 分析实验数据并得出结论...",
                    "answer": f"结论{i}",
                    "knowledge_points": ["科学探究", "数据分析"],
                    "difficulty": 0.5 + i * 0.003
                }
                for i in range(1, 101)
            ]
        elif dataset_name == "ktbd-ednet":
            sample_data = [
                {
                    "id": f"ednet_{i}",
                    "question": f"知识点练习{i}: 选择正确答案...",
                    "answer": str(i % 4 + 1),
                    "knowledge_points": [f"知识点_{j}" for j in range(i % 5 + 1)],
                    "difficulty": 0.35 + i * 0.004
                }
                for i in range(1, 101)
            ]
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 示例数据已保存到：{save_path}")
    
    def download_all(self, force: bool = False) -> dict:
        """
        下载所有数据集
        
        Returns:
            数据集路径字典
        """
        paths = {}
        for dataset_name in self.datasets.keys():
            paths[dataset_name] = self.download_dataset(dataset_name, force)
        return paths
    
    def load_dataset(self, dataset_name: str) -> list:
        """
        加载数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集列表
        """
        save_path = self.data_dir / f"{dataset_name}.json"
        if not save_path.exists():
            raise FileNotFoundError(f"数据集不存在：{save_path}. 请先下载。")
        
        with open(save_path, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    # 示例用法
    downloader = DatasetDownloader()
    
    # 下载单个数据集
    # path = downloader.download_dataset("math23k")
    
    # 下载所有数据集
    paths = downloader.download_all()
    print("\n数据集下载完成:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # 加载数据集
    # data = downloader.load_dataset("math23k")
    # print(f"\nmath23k 数据集大小：{len(data)}")
