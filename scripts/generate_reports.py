"""
诊断报告生成模块
调用本地 ollama 大模型 (gemma3:4b) 处理题干信息并生成文本诊断报告
"""

import json
import requests
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm


class DiagnosticReportGenerator:
    """诊断报告生成器"""
    
    def __init__(self, model_name: str = "gemma3:4b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_generate = f"{ollama_url}/api/generate"
        self.api_chat = f"{ollama_url}/api/chat"
    
    def check_model_availability(self) -> bool:
        """检查模型是否可用"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model_name in model_names:
                    print(f"✓ 模型 {self.model_name} 可用")
                    return True
                else:
                    print(f"✗ 模型 {self.model_name} 未找到。可用模型：{model_names}")
                    return False
            return False
        except requests.exceptions.RequestException as e:
            print(f"✗ 无法连接到 Ollama 服务：{e}")
            print("请确保 Ollama 服务正在运行，并且已拉取 gemma3:4b 模型")
            return False
    
    def generate_report(self, question: str, answer: str = None, 
                       knowledge_points: List[str] = None) -> str:
        """
        为单个题目生成诊断报告
        
        Args:
            question: 题目标干
            answer: 学生答案（可选）
            knowledge_points: 知识点列表（可选）
            
        Returns:
            生成的诊断报告文本
        """
        # 构建提示词
        prompt = self._build_prompt(question, answer, knowledge_points)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        }
        
        try:
            response = requests.post(self.api_generate, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"生成报告失败：{e}")
            return self._generate_fallback_report(question, answer, knowledge_points)
    
    def _build_prompt(self, question: str, answer: str = None, 
                     knowledge_points: List[str] = None) -> str:
        """构建提示词"""
        prompt = """你是一位专业的教育诊断专家。请根据以下题目信息生成一份详细的诊断报告。

"""
        prompt += f"题目：{question}\n"
        
        if answer:
            prompt += f"学生答案：{answer}\n"
        
        if knowledge_points:
            prompt += f"涉及知识点：{', '.join(knowledge_points)}\n"
        
        prompt += """
请生成一份诊断报告，包含以下内容：
1. 题目分析：题目的考查目标和难度分析
2. 知识点掌握情况：学生对各个知识点的掌握程度评估
3. 错误分析：如果提供了学生答案，分析可能的错误原因
4. 学习建议：针对性的学习建议和后续练习方向

请以结构化的方式输出报告内容。
"""
        return prompt
    
    def _generate_fallback_report(self, question: str, answer: str = None,
                                  knowledge_points: List[str] = None) -> str:
        """生成 fallback 报告（当模型不可用时）"""
        report = "【诊断报告】\n\n"
        report += f"题目：{question[:100]}...\n\n"
        
        if knowledge_points:
            report += f"涉及知识点：{', '.join(knowledge_points)}\n\n"
        
        report += "题目分析：\n"
        report += "- 本题考查学生对基础概念的理解和应用能力\n"
        report += "- 难度等级：中等\n\n"
        
        if answer:
            report += f"学生答案分析：\n"
            report += f"- 提供的答案：{answer}\n"
            report += "- 需要进一步评估答案的准确性\n\n"
        
        report += "学习建议：\n"
        report += "- 建议加强相关知识点的练习\n"
        report += "- 推荐类似题目进行巩固训练\n"
        
        return report
    
    def generate_batch_reports(self, dataset: List[Dict], 
                               output_path: str = None,
                               batch_size: int = 10) -> List[Dict]:
        """
        批量生成诊断报告
        
        Args:
            dataset: 数据集列表
            output_path: 输出文件路径（可选）
            batch_size: 批处理大小
            
        Returns:
            包含诊断报告的数据集
        """
        results = []
        
        # 检查模型可用性
        model_available = self.check_model_availability()
        if not model_available:
            print("⚠ 使用 fallback 模式生成报告")
        
        print(f"开始为 {len(dataset)} 条数据生成诊断报告...")
        
        for i, item in enumerate(tqdm(dataset, desc="生成报告")):
            question = item.get("question", "")
            answer = item.get("answer", None)
            knowledge_points = item.get("knowledge_points", None)
            
            report = self.generate_report(question, answer, knowledge_points)
            
            result_item = item.copy()
            result_item["diagnostic_report"] = report
            results.append(result_item)
            
            # 定期保存进度
            if output_path and (i + 1) % batch_size == 0:
                self._save_results(results, output_path)
                print(f"已保存 {i + 1} 条报告到 {output_path}")
        
        # 最终保存
        if output_path:
            self._save_results(results, output_path)
            print(f"✓ 所有报告已保存到：{output_path}")
        
        return results
    
    def _save_results(self, results: List[Dict], output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def load_dataset(dataset_path: str) -> List[Dict]:
    """加载数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    from download_datasets import DatasetDownloader
    
    # 初始化
    generator = DiagnosticReportGenerator(model_name="gemma3:4b")
    downloader = DatasetDownloader()
    
    # 加载数据集
    print("加载 math23k 数据集...")
    data = downloader.load_dataset("math23k")
    print(f"数据集大小：{len(data)}")
    
    # 生成诊断报告（前 5 条用于测试）
    sample_data = data[:5]
    reports = generator.generate_batch_reports(
        sample_data,
        output_path="./data/math23k_with_reports.json",
        batch_size=5
    )
    
    print("\n示例报告:")
    print(reports[0]["diagnostic_report"])
