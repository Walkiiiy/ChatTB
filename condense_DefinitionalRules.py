#!/usr/bin/env python3
"""
规则凝缩脚本 - 使用本地LLM凝缩definational rules

该脚本读取rules_res_type.json数据集，对每条数据中的definational rules进行凝缩，
合并相似的规则，在保证不丢失信息的前提下减少规则数量。
"""

import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import os
import sys

# 添加Process_model路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Process_model'))

from LLMClient import LLMClient


class RuleCondenser:
    """规则凝缩器类"""
    
    def __init__(self, model_path: str, max_new_tokens: int = 1024, temperature: float = 0.1):
        """
        初始化规则凝缩器
        
        Args:
            model_path: 本地LLM模型路径
            max_new_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化LLM客户端
        self.llm_client = LLMClient(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        # 系统提示词
        self.system_prompt = """你是一个专业的SQL规则分析专家。你的任务是将多个相似的definational rules合并成一个更简洁、更通用的规则，同时确保不丢失任何重要信息。

definational rules是定义性的规则，描述了在特定条件下应该使用什么列名、表名或操作。你需要：
1. 识别相似的规则（例如都涉及相同的列名或表名）
2. 将它们合并成一个更通用的规则
3. 确保合并后的规则覆盖所有原始规则的情况
4. 保持规则的准确性和完整性

输出格式：返回一个JSON数组，包含合并后的规则，每个规则包含condition和operation字段。"""
    
    def extract_definational_rules(self, rules: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        从规则列表中提取definational rules
        
        Args:
            rules: 规则列表
            
        Returns:
            definational rules列表
        """
        return [rule for rule in rules if rule.get("type") == "definational"]
    
    def create_condensation_prompt(self, definational_rules: List[Dict[str, str]], 
                                 question: str, db_id: str) -> str:
        """
        创建规则凝缩的提示词
        
        Args:
            definational_rules: definational rules列表
            question: 问题文本
            db_id: 数据库ID
            
        Returns:
            格式化的提示词
        """
        rules_text = ""
        for i, rule in enumerate(definational_rules, 1):
            rules_text += f"{i}. Condition: {rule['condition']}\n"
            rules_text += f"   Operation: {rule['operation']}\n\n"
        
        prompt = f"""
        Database: {db_id}
Question: {question}

The following are the definitional rules that need to be consolidated:

{rules_text}

Your task: Analyze these rules and merge similar ones into more concise and general rules.  

Strict requirements:
1. The merged rules must cover **all cases** from the original rules.  
2. **No important information may be lost** — preserve every detail, but merge and rewrite similar or redundant operations into a **single unified statement**.  
3. If multiple rules have the same condition, produce **one rule** whose operation is expressed as **a single string** (not an array).  
   - This string should merge overlapping details into one concise, clear description.  
   - If multiple operations differ, combine them into a single paragraph or long sentence that expresses priorities, fallbacks, or variations.  
4. If conditions are semantically equivalent, merge them into one condition.  
5. Redundancy is acceptable if needed for clarity, but information loss is not.  
6. Output must be **only valid JSON**, formatted as an array.  
7. Each rule in the array must include:  
   - `"condition"`: the merged condition text.  
   - `"operation"`: a **single string** containing the fully merged operation.  
   - `"type"`: always `"definitional"`.  
   - `"merged_from"`: a list of the original input indices that were merged.  
   - `"note"`: optional, only if needed to explain priorities, fallbacks, or conflict resolution.  

Return only the consolidated rules in JSON format.
"""
        
        return prompt
    
    def parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        解析LLM响应，提取合并后的规则
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的规则列表
        """
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('['):
                rules = json.loads(response.strip())
            else:
                # 尝试从响应中提取JSON
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    rules = json.loads(json_match.group())
                else:
                    self.logger.warning(f"无法从响应中提取JSON: {response[:200]}...")
                    return []
            
            # 验证规则格式
            valid_rules = []
            for rule in rules:
                if isinstance(rule, dict) and 'condition' in rule and 'operation' in rule:
                    valid_rules.append({
                        'condition': rule['condition'],
                        'operation': rule['operation'],
                        'type': 'definational'
                    })
                else:
                    self.logger.warning(f"无效的规则格式: {rule}")
            
            return valid_rules
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {e}")
            self.logger.error(f"响应内容: {response[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"解析响应时出错: {e}")
            return []
    
    def condense_rules_for_sample(self, sample: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """
        对单个样本进行规则凝缩
        
        Args:
            sample: 包含rules的样本数据
            sample_id: 样本ID
            
        Returns:
            凝缩后的样本数据
        """
        # 提取definational rules
        definational_rules = self.extract_definational_rules(sample.get("rules", []))
        
        # 输出原始规则
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"样本 {sample_id} - 原始 definational rules ({len(definational_rules)} 条):")
        for i, rule in enumerate(definational_rules, 1):
            self.logger.info(f"  {i}. Condition: {rule['condition']}")
            self.logger.info(f"     Operation: {rule['operation']}")
        
        if len(definational_rules) <= 1:
            # 如果只有一个或没有definational rules，直接返回原样
            self.logger.info(f"样本 {sample_id} - 只有一个或没有definational rules，跳过凝缩")
            return sample
        
        # 创建凝缩提示词
        prompt = self.create_condensation_prompt(
            definational_rules, 
            sample.get("question", ""), 
            sample.get("db_id", "")
        )
        
        try:
            # 调用LLM进行规则凝缩
            self.logger.info(f"样本 {sample_id} - 开始规则凝缩...")
            response = self.llm_client.chat(prompt, self.system_prompt)
            
            # 解析响应
            condensed_rules = self.parse_llm_response(response)
            
            if condensed_rules:
                # 输出凝缩后的规则
                self.logger.info(f"样本 {sample_id} - 凝缩后的 definational rules ({len(condensed_rules)} 条):")
                for i, rule in enumerate(condensed_rules, 1):
                    self.logger.info(f"  {i}. Condition: {rule['condition']}")
                    self.logger.info(f"     Operation: {rule['operation']}")
                
                # 计算压缩比
                compression_ratio = len(condensed_rules) / len(definational_rules)
                self.logger.info(f"样本 {sample_id} - 压缩比: {compression_ratio:.3f} ({len(condensed_rules)}/{len(definational_rules)})")
                
                # 创建新的样本数据
                new_sample = sample.copy()
                new_sample["rules"] = condensed_rules
                new_sample["original_definational_count"] = len(definational_rules)
                new_sample["condensed_definational_count"] = len(condensed_rules)
                new_sample["condensation_ratio"] = compression_ratio
                return new_sample
            else:
                self.logger.warning(f"样本 {sample_id} - 规则凝缩失败，使用原始规则")
                return sample
                
        except Exception as e:
            self.logger.error(f"样本 {sample_id} - 规则凝缩过程中出错: {e}")
            return sample
    
    def check_resume_status(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        检查断点恢复状态
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            
        Returns:
            包含恢复状态信息的字典
        """
        status = {
            'can_resume': False,
            'processed_count': 0,
            'total_count': 0,
            'remaining_count': 0,
            'stats': {
                'successful_condensations': 0,
                'failed_condensations': 0,
                'total_original_rules': 0,
                'total_condensed_rules': 0
            }
        }
        
        # 读取输入数据获取总数
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        status['total_count'] = len(data)
        
        # 检查输出文件是否存在
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                status['processed_count'] = len(processed_data)
                status['remaining_count'] = status['total_count'] - status['processed_count']
                status['can_resume'] = status['remaining_count'] > 0
                
                # 计算统计信息
                for sample_id, sample in processed_data.items():
                    original_count = sample.get('original_definational_count', 0)
                    condensed_count = sample.get('condensed_definational_count', 0)
                    status['stats']['total_original_rules'] += original_count
                    status['stats']['total_condensed_rules'] += condensed_count
                    
                    if condensed_count < original_count:
                        status['stats']['successful_condensations'] += 1
                    else:
                        status['stats']['failed_condensations'] += 1
                        
            except Exception as e:
                self.logger.warning(f"读取输出文件失败: {e}")
                status['can_resume'] = False
        
        return status
    
    def process_dataset(self, input_file: str, output_file: str, 
                       max_samples: Optional[int] = None) -> None:
        """
        处理整个数据集，支持从断点继续处理
        
        Args:
            input_file: 输入JSON文件路径
            output_file: 输出JSON文件路径
            max_samples: 最大处理样本数（用于测试）
        """
        self.logger.info(f"开始处理数据集: {input_file}")
        
        # 读取输入数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确定处理范围
        sample_ids = list(data.keys())
        if max_samples:
            sample_ids = sample_ids[:max_samples]
        
        # 检查输出文件是否存在，如果存在则从断点继续
        processed_data = {}
        start_index = 0
        condensation_stats = {
            'total_samples': len(sample_ids),
            'successful_condensations': 0,
            'failed_condensations': 0,
            'total_original_rules': 0,
            'total_condensed_rules': 0
        }
        
        if os.path.exists(output_file):
            self.logger.info(f"发现已存在的输出文件: {output_file}")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                # 计算已处理的样本数量
                processed_count = len(processed_data)
                start_index = processed_count
                
                # 重新计算统计信息
                for sample_id, sample in processed_data.items():
                    original_count = sample.get('original_definational_count', 0)
                    condensed_count = sample.get('condensed_definational_count', 0)
                    condensation_stats['total_original_rules'] += original_count
                    condensation_stats['total_condensed_rules'] += condensed_count
                    
                    if condensed_count < original_count:
                        condensation_stats['successful_condensations'] += 1
                    else:
                        condensation_stats['failed_condensations'] += 1
                
                self.logger.info(f"从断点继续处理: 已处理 {processed_count} 个样本，从第 {start_index + 1} 个样本开始")
                self.logger.info(f"当前统计: 成功凝缩 {condensation_stats['successful_condensations']} 个，失败 {condensation_stats['failed_condensations']} 个")
                
            except Exception as e:
                self.logger.warning(f"读取输出文件失败: {e}，将重新开始处理")
                processed_data = {}
                start_index = 0
        else:
            self.logger.info(f"输出文件不存在，将从头开始处理")
        
        self.logger.info(f"总共需要处理 {len(sample_ids)} 个样本")
        
        # 创建临时保存文件
        temp_output_file = output_file.replace('.json', '_temp.json')
        
        # 从断点开始处理
        remaining_samples = sample_ids[start_index:]
        if not remaining_samples:
            self.logger.info("所有样本已处理完成！")
            return
        
        self.logger.info(f"剩余需要处理的样本: {len(remaining_samples)} 个")
        
        for i, sample_id in enumerate(remaining_samples, start=start_index):
            sample = data[sample_id]
            
            # 统计原始规则数量
            original_definational_count = len(self.extract_definational_rules(sample.get("rules", [])))
            condensation_stats['total_original_rules'] += original_definational_count
            
            # 进行规则凝缩
            condensed_sample = self.condense_rules_for_sample(sample, sample_id)
            
            # 统计凝缩后规则数量
            condensed_definational_count = len(self.extract_definational_rules(condensed_sample.get("rules", [])))
            condensation_stats['total_condensed_rules'] += condensed_definational_count
            
            # 判断是否成功凝缩
            if condensed_definational_count < original_definational_count:
                condensation_stats['successful_condensations'] += 1
            else:
                condensation_stats['failed_condensations'] += 1
            
            processed_data[sample_id] = condensed_sample
            
            # 每10条数据保存一次
            if (i + 1) % 10 == 0:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"已处理 {i + 1} 个样本，保存中间结果...")
                
                # 保存当前进度到输出文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                # 输出当前统计信息
                self.logger.info(f"当前统计:")
                self.logger.info(f"  成功凝缩的样本: {condensation_stats['successful_condensations']}")
                self.logger.info(f"  凝缩失败的样本: {condensation_stats['failed_condensations']}")
                self.logger.info(f"  原始definational rules总数: {condensation_stats['total_original_rules']}")
                self.logger.info(f"  凝缩后definational rules总数: {condensation_stats['total_condensed_rules']}")
                
                if condensation_stats['total_original_rules'] > 0:
                    compression_ratio = condensation_stats['total_condensed_rules'] / condensation_stats['total_original_rules']
                    self.logger.info(f"  当前压缩比: {compression_ratio:.3f}")
                
                self.logger.info(f"  中间结果已保存到: {output_file}")
                self.logger.info(f"{'='*80}\n")
        
        # 最终保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # 输出最终统计信息
        self.logger.info(f"\n{'='*80}")
        self.logger.info("处理完成！")
        self.logger.info(f"成功凝缩的样本: {condensation_stats['successful_condensations']}")
        self.logger.info(f"凝缩失败的样本: {condensation_stats['failed_condensations']}")
        self.logger.info(f"原始definational rules总数: {condensation_stats['total_original_rules']}")
        self.logger.info(f"凝缩后definational rules总数: {condensation_stats['total_condensed_rules']}")
        
        if condensation_stats['total_original_rules'] > 0:
            compression_ratio = condensation_stats['total_condensed_rules'] / condensation_stats['total_original_rules']
            self.logger.info(f"最终压缩比: {compression_ratio:.3f}")
        
        self.logger.info(f"结果已保存到: {output_file}")
        self.logger.info(f"{'='*80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="规则凝缩脚本")
    parser.add_argument("--input", "-i", required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSON文件路径")
    parser.add_argument("--model_path", "-m", 
                       default="/home/ubuntu/walkiiiy/ChatTB/Process_model/models--Qwen3-8B",
                       help="本地LLM模型路径")
    parser.add_argument("--max_samples", "-n", type=int, 
                       help="最大处理样本数（用于测试）")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="生成温度")
    parser.add_argument("--log_level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--check_status", action="store_true",
                       help="检查断点恢复状态，不进行实际处理")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return
    
    # 如果只是检查状态，不需要模型
    if args.check_status:
        # 创建规则凝缩器（不需要加载模型）
        condenser = RuleCondenser(
            model_path=args.model_path,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # 检查断点状态
        status = condenser.check_resume_status(args.input, args.output)
        
        print(f"\n{'='*80}")
        print("断点恢复状态检查")
        print(f"{'='*80}")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"总样本数: {status['total_count']}")
        print(f"已处理样本数: {status['processed_count']}")
        print(f"剩余样本数: {status['remaining_count']}")
        print(f"可以恢复: {'是' if status['can_resume'] else '否'}")
        
        if status['processed_count'] > 0:
            print(f"\n当前统计:")
            print(f"  成功凝缩的样本: {status['stats']['successful_condensations']}")
            print(f"  凝缩失败的样本: {status['stats']['failed_condensations']}")
            print(f"  原始definational rules总数: {status['stats']['total_original_rules']}")
            print(f"  凝缩后definational rules总数: {status['stats']['total_condensed_rules']}")
            
            if status['stats']['total_original_rules'] > 0:
                compression_ratio = status['stats']['total_condensed_rules'] / status['stats']['total_original_rules']
                print(f"  当前压缩比: {compression_ratio:.3f}")
        
        print(f"{'='*80}")
        return
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        logging.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 创建规则凝缩器
    condenser = RuleCondenser(
        model_path=args.model_path,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # 处理数据集
    condenser.process_dataset(
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
