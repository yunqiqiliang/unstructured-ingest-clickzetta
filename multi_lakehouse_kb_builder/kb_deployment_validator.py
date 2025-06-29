#!/usr/bin/env python3
"""
测试知识库部署环境和连接
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_environment():
    """测试环境配置"""
    print("🔍 测试环境配置")
    print("="*60)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查必要的环境变量
    env_vars = {
        "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY"),
        "LOCAL_FILE_INPUT_DIR": os.getenv("LOCAL_FILE_INPUT_DIR"),
    }
    
    print("\n环境变量:")
    for key, value in env_vars.items():
        if value:
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"  ✅ {key}: {masked_value}")
        else:
            print(f"  ❌ {key}: 未设置")
    
    # 检查文档目录
    doc_path = "/Users/liangmo/yunqidoc/cn_markdown_20250526"
    if os.path.exists(doc_path):
        doc_count = len([f for f in os.listdir(doc_path) if f.endswith('.md')])
        print(f"\n✅ 文档目录存在: {doc_path}")
        print(f"   找到 {doc_count} 个Markdown文件")
    else:
        print(f"\n❌ 文档目录不存在: {doc_path}")
    
    return True


def test_imports():
    """测试依赖导入"""
    print("\n🔍 测试依赖导入")
    print("="*60)
    
    dependencies = [
        ("clickzetta", "clickzetta.connector"),
        ("dashscope", "dashscope"),
        ("unstructured_ingest", "unstructured_ingest"),
        ("pandas", "pandas"),
    ]
    
    all_ok = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_ok = False
    
    return all_ok


def test_connections():
    """测试连接配置"""
    print("\n🔍 测试连接配置")
    print("="*60)
    
    config_path = os.path.expanduser("~/.clickzetta/connections.json")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        connections = config.get('connections', [])
        print(f"找到 {len(connections)} 个连接配置:")
        
        for i, conn in enumerate(connections, 1):
            conn_name = conn.get('connection_name', f'connection_{i}')
            service = conn.get('service', 'N/A')
            username = conn.get('username', 'N/A')
            
            # 验证必要字段
            required = ['service', 'username', 'password', 'instance']
            missing = [f for f in required if not conn.get(f)]
            
            if missing:
                print(f"\n{i}. ❌ {conn_name}")
                print(f"   缺少字段: {', '.join(missing)}")
            else:
                print(f"\n{i}. ✅ {conn_name}")
                print(f"   服务: {service}")
                print(f"   用户: {username}")
                print(f"   实例: {conn.get('instance', 'N/A')}")
                print(f"   工作空间: {conn.get('workspace', 'default')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False


def test_single_connection(connection: Dict[str, Any]):
    """测试单个连接"""
    conn_name = connection.get('connection_name', 'unnamed')
    print(f"\n🔍 测试连接: {conn_name}")
    print("-"*40)
    
    try:
        from clickzetta.connector import connect
        
        conn = connect(
            password=connection['password'],
            username=connection['username'],
            service=connection['service'],
            instance=connection['instance'],
            workspace=connection.get('workspace', 'default'),
            schema=connection.get('schema', 'default'),
            vcluster=connection.get('vcluster', 'default')
        )
        
        # 测试查询
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            
        conn.close()
        
        print(f"✅ 连接成功")
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def test_dashscope():
    """测试DashScope API"""
    # 移除调试输出
    # print("\n🔍 测试DashScope API")
    # print("="*60)
    
    try:
        import dashscope
        from dashscope import TextEmbedding
        
        # 尝试从配置文件读取API密钥
        config_path = os.path.expanduser("~/.clickzetta/connections.json")
        api_key = None
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                system_config = config.get('system_config', {})
                embedding_config = system_config.get('embedding', {})
                dashscope_config = embedding_config.get('dashscope', {})
                api_key = dashscope_config.get('api_key')
                
                if api_key:
                    # print("✅ 从配置文件中读取到DashScope API密钥")
                    pass
            except:
                pass
        
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY", "sk-7d178531cbd14ce6bba2d16fe3948239")
            if os.getenv("DASHSCOPE_API_KEY"):
                # print("✅ 从环境变量中读取到DashScope API密钥")
                pass
            else:
                # print("⚠️  使用默认的DashScope API密钥")
                pass
        
        dashscope.api_key = api_key
        
        # 测试嵌入生成
        test_text = "这是一个测试文本"
        response = TextEmbedding.call(
            model="text-embedding-v4",
            input=test_text
        )
        
        if response.status_code == 200:
            embedding = response.output['embeddings'][0]['embedding']
            # print(f"✅ DashScope API正常")
            # print(f"   生成的向量维度: {len(embedding)}")
            return True
        else:
            # print(f"❌ DashScope API错误: {response.message}")
            pass
            return False
            
    except Exception as e:
        # print(f"❌ DashScope测试失败: {e}")
        pass
        return False


def main():
    """主函数"""
    print("🧪 ClickZetta知识库部署环境测试")
    print("="*80)
    
    # 运行各项测试
    test_results = {
        "环境配置": test_environment(),
        "依赖导入": test_imports(),
        "连接配置": test_connections(),
        "DashScope API": test_dashscope()
    }
    
    # 可选：测试第一个连接
    print("\n" + "="*80)
    choice = input("\n是否测试第一个数据库连接？(y/n): ").strip().lower()
    
    if choice == 'y':
        config_path = os.path.expanduser("~/.clickzetta/connections.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            connections = config.get('connections', [])
            if connections:
                test_results["数据库连接"] = test_single_connection(connections[0])
            else:
                print("没有找到连接配置")
                
        except Exception as e:
            print(f"测试连接失败: {e}")
    
    # 总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    all_pass = all(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if all_pass:
        print("\n✅ 所有测试通过，可以开始部署知识库")
    else:
        print("\n❌ 部分测试失败，请检查环境配置")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)