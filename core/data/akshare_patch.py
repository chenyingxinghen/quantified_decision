#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
akshare请求方法补丁模块
用于在请求头中添加nid和create_time参数
"""
import random
import time
from datetime import datetime
from typing import Optional, Dict, Any
import requests


def generate_nid():
    """生成随机的nid"""
    return ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyz', k=32))


def patch_akshare_requests():
    """补丁akshare的请求方法，在请求头中添加nid和create_time"""
    try:
        import akshare.request
        
        # 保存原始函数
        original_make_request_with_retry_json = akshare.request.make_request_with_retry_json
        original_make_request_with_retry_text = akshare.request.make_request_with_retry_text
        
        def patched_make_request_with_retry_json(
            url, params=None, headers=None, proxies=None, max_retries=3, retry_delay=1
        ):
            """
            修补版的JSON请求函数，添加nid和create_time到请求头
            """
            # 确保headers存在
            if headers is None:
                headers = {}
            
            # 添加nid和create_time到请求头
            headers['nid'] = generate_nid()
            headers['create_time'] = str(int(time.time() * 1000))  # 当前时间戳（毫秒）
            
            # 调用原始函数
            return original_make_request_with_retry_json(
                url, params=params, headers=headers, proxies=proxies, 
                max_retries=max_retries, retry_delay=retry_delay
            )
        
        def patched_make_request_with_retry_text(
            url, params=None, headers=None, proxies=None, max_retries=3, retry_delay=1
        ):
            """
            修补版的文本请求函数，添加nid和create_time到请求头
            """
            # 确保headers存在
            if headers is None:
                headers = {}
            
            # 添加nid和create_time到请求头
            headers['nid'] = generate_nid()
            headers['create_time'] = str(int(time.time() * 1000))  # 当前时间戳（毫秒）
            
            # 调用原始函数
            return original_make_request_with_retry_text(
                url, params=params, headers=headers, proxies=proxies, 
                max_retries=max_retries, retry_delay=retry_delay
            )
        
        # 替换akshare中的函数
        akshare.request.make_request_with_retry_json = patched_make_request_with_retry_json
        akshare.request.make_request_with_retry_text = patched_make_request_with_retry_text
        
        # print("✓ akshare请求方法已成功打补丁")
        # print("  - 已添加nid和create_time到请求头")
        # print("  - 原始功能保持不变")
        
        return True
        
    except ImportError:
        print("✗ 无法导入akshare模块")
        return False
    except Exception as e:
        print(f"✗ 打补丁失败: {e}")
        return False


def unpatch_akshare_requests():
    """取消补丁（如有需要）"""
    try:
        import importlib
        importlib.reload(__import__('akshare.request'))
        print("✓ 已取消akshare请求方法补丁")
        return True
    except Exception as e:
        print(f"✗ 取消补丁失败: {e}")
        return False


if __name__ == "__main__":
    # 测试补丁
    print("测试akshare请求补丁...")
    success = patch_akshare_requests()
    if success:
        print("补丁应用成功！")
    else:
        print("补丁应用失败！")