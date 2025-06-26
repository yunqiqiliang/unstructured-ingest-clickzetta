#!/usr/bin/env python3
"""
工具函数模块
"""

from typing import List, Dict, Any, Optional


def confirm_action(message: str, default: bool = False) -> bool:
    """
    显示确认提示并获取用户输入
    
    Args:
        message: 确认消息
        default: 默认值（True表示默认yes，False表示默认no）
    
    Returns:
        bool: 用户是否确认
    """
    if default:
        prompt = f"{message} (Y/n): "
        default_answer = 'y'
    else:
        prompt = f"{message} (y/N): "
        default_answer = 'n'
    
    answer = input(prompt).strip().lower()
    
    if not answer:
        answer = default_answer
    
    return answer in ['y', 'yes', '是', 'ok']


def display_connections(connections: List[Dict[str, Any]], title: str = "连接列表") -> None:
    """
    显示连接列表
    
    Args:
        connections: 连接配置列表
        title: 显示标题
    """
    if not connections:
        print(f"\n❌ {title}为空")
        return
    
    print(f"\n{title} ({len(connections)} 个):")
    for i, conn in enumerate(connections, 1):
        conn_name = conn.get('connection_name', 'unnamed')
        service = conn.get('service', 'N/A')
        instance = conn.get('instance', 'N/A')
        workspace = conn.get('workspace', 'default')
        
        print(f"\n{i}. {conn_name}")
        print(f"   服务: {service}")
        print(f"   实例: {instance}")
        print(f"   工作空间: {workspace}")


def get_matched_connections(connections: List[Dict[str, Any]], 
                          pattern: str) -> List[Dict[str, Any]]:
    """
    获取匹配特定模式的连接
    
    Args:
        connections: 所有连接列表
        pattern: 匹配模式
    
    Returns:
        匹配的连接列表
    """
    matched = []
    for conn in connections:
        conn_name = conn.get('connection_name', 'unnamed')
        if pattern in conn_name:
            matched.append(conn)
    return matched


def format_duration(seconds: float) -> str:
    """
    格式化时间持续长度
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def format_number(num: int) -> str:
    """
    格式化数字，添加千位分隔符
    
    Args:
        num: 数字
    
    Returns:
        格式化的数字字符串
    """
    return f"{num:,}"