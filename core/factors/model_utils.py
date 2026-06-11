"""
模型工具函数 - 用于解析和管理模型文件夹命名
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def parse_model_folder_name(folder_name: str) -> Dict[str, any]:
    """
    解析模型文件夹名称，提取训练配置信息
    
    格式: {模型缩写}_{预测天数}d_{训练年数}y_{股票数}s_{配置标志}_{任务缩写}_{时间戳}
    
    参数:
        folder_name: 文件夹名称，如 "xl_15d_16y_6000s_GF_hy_0529_1512"
        
    返回:
        包含解析信息的字典
    """
    pattern = r'^([a-z]+)_(\d+)d_(\d+)y_(\d+)s_([A-Z]+)_([a-z]+)_(\d{4}_\d{4})$'
    match = re.match(pattern, folder_name)
    
    if not match:
        # 尝试解析旧格式
        return _parse_old_folder_name(folder_name)
    
    model_abbr, forward_days, years, stocks, config_str, task_abbr, timestamp = match.groups()
    
    # 解析模型类型
    model_types = []
    if model_abbr == 'xl':
        model_types = ['xgboost', 'lightgbm']
    elif model_abbr == 'xg':
        model_types = ['xgboost']
    elif model_abbr == 'lgb':
        model_types = ['lightgbm']
    else:
        model_types = [model_abbr]
    
    # 解析配置标志
    config_flags = {
        'use_gpu': 'G' in config_str,
        'use_sample_weight': 'W' in config_str,
        'include_fundamentals': 'F' in config_str,
        'include_candle_pattern': 'C' in config_str,
        'no_special_config': config_str == 'N'
    }
    
    # 解析任务类型
    task_map = {
        'hy': 'hybrid',
        'rg': 'regression',
        'rk': 'ranking'
    }
    task = task_map.get(task_abbr, task_abbr)
    
    # 解析时间戳
    try:
        train_time = datetime.strptime(timestamp, '%m%d_%H%M')
        timestamp_str = train_time.strftime('%Y-%m-%d %H:%M')
    except:
        timestamp_str = timestamp
    
    return {
        'folder_name': folder_name,
        'model_abbr': model_abbr,
        'model_types': model_types,
        'forward_days': int(forward_days),
        'training_years': int(years),
        'stock_count': int(stocks),
        'config_str': config_str,
        'config_flags': config_flags,
        'task_abbr': task_abbr,
        'task': task,
        'timestamp': timestamp,
        'train_time': timestamp_str,
        'format': 'new'
    }


def _parse_old_folder_name(folder_name: str) -> Dict[str, any]:
    """
    解析旧格式的文件夹名称
    
    旧格式: train_{task}_{forward_days}d_{years}y_{stocks}s_{timestamp}
    示例: train_hybrid_20d_16y_5472s_0526_1643
    """
    pattern = r'^train_([a-z]+)_(\d+)d_(\d+)y_(\d+)s_(\d{4}_\d{4})$'
    match = re.match(pattern, folder_name)
    
    if not match:
        return {
            'folder_name': folder_name,
            'format': 'unknown',
            'error': '无法解析文件夹名称格式'
        }
    
    task, forward_days, years, stocks, timestamp = match.groups()
    
    try:
        train_time = datetime.strptime(timestamp, '%m%d_%H%M')
        timestamp_str = train_time.strftime('%Y-%m-%d %H:%M')
    except:
        timestamp_str = timestamp
    
    return {
        'folder_name': folder_name,
        'task': task,
        'forward_days': int(forward_days),
        'training_years': int(years),
        'stock_count': int(stocks),
        'timestamp': timestamp,
        'train_time': timestamp_str,
        'format': 'old'
    }


def get_latest_model_dir(models_root: str = 'models') -> Optional[str]:
    """
    获取最新的模型目录
    
    参数:
        models_root: 模型根目录
        
    返回:
        最新的模型目录路径，如果没有则返回None
    """
    if not os.path.exists(models_root):
        return None
    
    # 首先检查是否有latest目录
    latest_dir = os.path.join(models_root, 'latest')
    if os.path.exists(latest_dir):
        return latest_dir
    
    # 查找所有模型目录
    model_dirs = []
    for item in os.listdir(models_root):
        item_path = os.path.join(models_root, item)
        if os.path.isdir(item_path) and item not in ['mark', 'latest']:
            model_dirs.append(item_path)
    
    if not model_dirs:
        return None
    
    # 按修改时间排序，取最新的
    model_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return model_dirs[0]


def find_model_files(model_dir: str) -> Dict[str, str]:
    """
    在模型目录中查找模型文件
    
    参数:
        model_dir: 模型目录
        
    返回:
        字典，键为模型类型，值为模型文件路径
    """
    model_files = {}
    
    if not os.path.exists(model_dir):
        return model_files
    
    # 查找所有模型文件
    for filename in os.listdir(model_dir):
        if filename.endswith('_factor_model.pkl'):
            model_type = filename.split('_')[0]
            model_files[model_type] = os.path.join(model_dir, filename)
    
    return model_files


def get_model_info(model_path: str) -> Dict[str, any]:
    """
    获取模型文件的完整信息
    
    参数:
        model_path: 模型文件路径
        
    返回:
        包含模型信息的字典
    """
    if not os.path.exists(model_path):
        return {'error': f'模型文件不存在: {model_path}'}
    
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # 解析文件夹名称
    folder_info = parse_model_folder_name(os.path.basename(model_dir))
    
    # 解析模型文件名称
    model_type = model_name.split('_')[0]
    
    # 尝试读取训练配置
    config_path = os.path.join(model_dir, 'training_config.json')
    config_info = {}
    if os.path.exists(config_path):
        import json
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_info = json.load(f)
        except:
            pass
    
    return {
        'model_path': model_path,
        'model_dir': model_dir,
        'model_name': model_name,
        'model_type': model_type,
        'folder_info': folder_info,
        'config_info': config_info,
        'exists': True,
        'file_size': os.path.getsize(model_path) if os.path.exists(model_path) else 0
    }


def list_available_models(models_root: str = 'models') -> List[Dict[str, any]]:
    """
    列出所有可用的模型
    
    参数:
        models_root: 模型根目录
        
    返回:
        模型信息列表
    """
    if not os.path.exists(models_root):
        return []
    
    models = []
    
    # 遍历所有目录
    for item in os.listdir(models_root):
        item_path = os.path.join(models_root, item)
        if not os.path.isdir(item_path):
            continue
        
        # 跳过特殊目录
        if item in ['mark', 'latest']:
            continue
        
        # 解析文件夹信息
        folder_info = parse_model_folder_name(item)
        
        # 查找模型文件
        model_files = find_model_files(item_path)
        
        if model_files:
            # 获取目录信息
            mtime = os.path.getmtime(item_path)
            size = sum(os.path.getsize(f) for f in model_files.values() if os.path.exists(f))
            
            models.append({
                'name': item,
                'path': item_path,
                'folder_info': folder_info,
                'model_files': model_files,
                'model_count': len(model_files),
                'mtime': mtime,
                'size': size,
                'formatted_time': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    # 按修改时间排序
    models.sort(key=lambda x: x['mtime'], reverse=True)
    
    return models


def format_model_info_for_display(model_info: Dict[str, any]) -> str:
    """
    格式化模型信息用于显示
    
    参数:
        model_info: 模型信息字典
        
    返回:
        格式化的字符串
    """
    if 'error' in model_info:
        return f"错误: {model_info['error']}"
    
    lines = []
    folder_info = model_info.get('folder_info', {})
    
    if folder_info.get('format') == 'new':
        lines.append(f"模型目录: {model_info.get('model_dir', '')}")
        lines.append(f"模型文件: {model_info.get('model_name', '')}")
        lines.append(f"模型类型: {', '.join(folder_info.get('model_types', []))}")
        lines.append(f"预测天数: {folder_info.get('forward_days', 0)}天")
        lines.append(f"训练年数: {folder_info.get('training_years', 0)}年")
        lines.append(f"股票数量: {folder_info.get('stock_count', 0):,}只")
        
        config_flags = folder_info.get('config_flags', {})
        config_desc = []
        if config_flags.get('use_gpu'):
            config_desc.append("GPU加速")
        if config_flags.get('use_sample_weight'):
            config_desc.append("样本加权")
        if config_flags.get('include_fundamentals'):
            config_desc.append("基本面因子")
        if config_flags.get('include_candle_pattern'):
            config_desc.append("K线形态")
        if config_desc:
            lines.append(f"配置: {', '.join(config_desc)}")
        
        lines.append(f"任务类型: {folder_info.get('task', '')}")
        lines.append(f"训练时间: {folder_info.get('train_time', '')}")
        
    elif folder_info.get('format') == 'old':
        lines.append(f"模型目录: {model_info.get('model_dir', '')}")
        lines.append(f"模型文件: {model_info.get('model_name', '')}")
        lines.append(f"任务类型: {folder_info.get('task', '')}")
        lines.append(f"预测天数: {folder_info.get('forward_days', 0)}天")
        lines.append(f"训练年数: {folder_info.get('training_years', 0)}年")
        lines.append(f"股票数量: {folder_info.get('stock_count', 0):,}只")
        lines.append(f"训练时间: {folder_info.get('train_time', '')}")
        lines.append("格式: 旧格式")
    
    else:
        lines.append(f"模型路径: {model_info.get('model_path', '')}")
        lines.append(f"格式: {folder_info.get('format', 'unknown')}")
    
    return '\n'.join(lines)