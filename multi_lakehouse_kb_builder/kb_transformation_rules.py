"""
数据转换规则引擎
处理从raw表到silver表的数据清洗和转换逻辑
"""
from typing import List, Dict, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class TransformationRuleEngine:
    """数据转换规则引擎"""
    
    def __init__(self):
        """初始化规则引擎"""
        # 预定义的转换操作模板 (基于ClickZetta支持的函数)
        self.operation_templates = {
            # 文本清理操作 (使用ClickZetta标准语法)
            "trim": "TRIM({column})",  # 去除两端空格
            "trim_left": "LTRIM({column})",  # 去除左侧空格
            "trim_right": "RTRIM({column})",  # 去除右侧空格
            "lowercase": "LOWER({column})",  # 转换为小写
            "uppercase": "UPPER({column})",  # 转换为大写
            
            # HTML和标记清理 (使用re2正则表达式引擎)
            "remove_html": "REGEXP_REPLACE({column}, '<[^>]+>', '')",
            "remove_urls": "REGEXP_REPLACE({column}, 'https?://[^\\s]+', '')",
            "remove_emails": "REGEXP_REPLACE({column}, '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}', '')",
            
            # 空白字符处理
            "normalize_spaces": "REGEXP_REPLACE({column}, '\\s+', ' ')",
            "remove_newlines": "REGEXP_REPLACE({column}, '[\\r\\n]+', ' ')",
            "remove_tabs": "REGEXP_REPLACE({column}, '\\t+', ' ')",
            
            # 特殊字符处理
            "keep_alphanumeric": "REGEXP_REPLACE({column}, '[^a-zA-Z0-9\\s]', '')",
            "keep_alphanumeric_chinese": "REGEXP_REPLACE({column}, '[^a-zA-Z0-9\\u4e00-\\u9fa5\\s]', '')",
            "remove_special_chars": "REGEXP_REPLACE({column}, '[!@#$%^&*()_+=\\[\\]{{}};\",<>?/\\\\|`~]', '')",
            
            # 数字处理
            "remove_numbers": "REGEXP_REPLACE({column}, '[0-9]+', '')",
            "extract_numbers": "REGEXP_EXTRACT({column}, '[0-9]+', 1)",  # 提取数字
            
            # 标点符号处理 (简化为单个REGEXP_REPLACE调用)
            "normalize_chinese_punctuation": "REGEXP_REPLACE({column}, '[，。！？；：""''（）【】《》]', ' ')",
            "keep_basic_punctuation": "REGEXP_REPLACE({column}, '[^a-zA-Z0-9\\u4e00-\\u9fa5\\s,.!?;:]', '')",
            
            # 字符串替换 (使用REPLACE函数)
            "remove_quotes": "REPLACE(REPLACE({column}, '\"', ''), '''', '')",  # 移除引号
            "replace_underscores": "REPLACE({column}, '_', ' ')",  # 下划线替换为空格
            
            # 自定义文本替换
            "replace_custom": "REPLACE({column}, '{old_text}', '{new_text}')",  # 自定义替换
            "replace_tabs_with_spaces": "REPLACE({column}, '\\t', '    ')",  # 制表符转4空格
            "replace_double_quotes": "REPLACE({column}, '\"', '\\'')",  # 双引号转单引号
            "replace_newlines_with_space": "REPLACE({column}, '\\n', ' ')",  # 换行符转空格
            
            # 常见标准化替换
            "normalize_quotes": "REPLACE(REPLACE({column}, '\u201c', '\"'), '\u201d', '\"')",  # 标准化引号
            "normalize_dashes": "REPLACE(REPLACE(REPLACE({column}, '—', '-'), '–', '-'), '‐', '-')",  # 标准化破折号
            "normalize_ellipsis": "REPLACE({column}, '…', '...')",  # 标准化省略号
            "remove_zero_width": "REPLACE(REPLACE(REPLACE({column}, '\\u200B', ''), '\\u200C', ''), '\\u200D', '')",  # 移除零宽字符
            
            # 敏感信息脱敏
            "mask_email": "REGEXP_REPLACE({column}, '([^@]+)@([^.]+)(\\..+)', '***@\\2\\3')",  # 邮箱前缀脱敏
            "mask_phone": "REGEXP_REPLACE({column}, '(\\d{3})\\d{4}(\\d{4})', '\\1****\\2')",  # 手机号脱敏
            
            # 中文特定处理
            "full_to_half_space": "REPLACE({column}, '　', ' ')",  # 全角空格转半角
            "remove_chinese_punctuation": "REGEXP_REPLACE({column}, '[，。！？；：、""''（）【】《》—…]', '')"  # 移除中文标点
        }
        
        # 预定义的过滤条件模板
        self.filter_templates = {
            "not_null": "{column} IS NOT NULL",
            "not_empty": "LENGTH({column}) > 0",
            "min_length": "LENGTH({column}) >= {value}",
            "max_length": "LENGTH({column}) <= {value}",
            "between_length": "LENGTH({column}) BETWEEN {min} AND {max}",
            "contains": "{column} LIKE '%{value}%'",
            "not_contains": "{column} NOT LIKE '%{value}%'",
            "starts_with": "{column} LIKE '{value}%'",
            "ends_with": "{column} LIKE '%{value}'",
            "regex_match": "{column} RLIKE '{pattern}'",
            "has_chinese": "{column} RLIKE '[\\u4e00-\\u9fa5]'",
            "has_english": "{column} RLIKE '[a-zA-Z]'",
            "is_numeric": "{column} RLIKE '^[0-9]+$'"
        }
    
    def generate_transformation_sql(self, 
                                  schema_name: str,
                                  raw_table: str,
                                  silver_table: str,
                                  rules: List[Dict],
                                  additional_columns: Optional[List[str]] = None,
                                  workspace: Optional[str] = None) -> str:
        """生成完整的转换SQL语句
        
        Args:
            schema_name: Schema名称
            raw_table: 原始表名
            silver_table: 目标表名
            rules: 转换规则列表
            additional_columns: 额外需要选择的列
            
        Returns:
            生成的SQL语句
        """
        # 移除调试代码
        
        # 不再需要自动添加embeddings的CAST转换
        # 因为Raw表和Silver表都使用VECTOR类型
        
        # 解析规则
        where_conditions = []
        
        # 处理转换规则，记录哪些列被转换
        transformed_columns = {}
        column_base_expr = {}  # 记录每个列的基础表达式（用于链式转换）
        
        for rule in rules:
            if rule.get('type') == 'transform':
                col_name = rule.get('column', '')
                if not col_name:
                    continue
                    
                # 获取当前列的基础表达式
                base_expr = column_base_expr.get(col_name, col_name)
                
                # 生成转换表达式，传入基础表达式
                expr = self._generate_transform_expression_with_base(rule, base_expr)
                if expr:
                    # 更新基础表达式（去掉 AS 部分）
                    if ' AS ' in expr:
                        new_base = expr.split(' AS ')[0]
                        column_base_expr[col_name] = new_base
                        transformed_columns[col_name] = expr
                    else:
                        column_base_expr[col_name] = expr
                        transformed_columns[col_name] = f"{expr} AS {col_name}"
            elif rule.get('type') == 'filter':
                condition = self._generate_filter_condition(rule)
                if condition:
                    where_conditions.append(condition)
            elif rule.get('type') == 'filter_group':
                # 处理过滤组
                group_condition = self._generate_filter_group_condition(rule)
                if group_condition:
                    where_conditions.append(group_condition)
        
        # 定义所有列的正确顺序（与表结构一致）
        all_columns_ordered = [
            'id', 
            'record_locator',
            'type', 
            'record_id', 
            'element_id', 
            'filetype', 
            'file_directory',
            'filename',
            'last_modified', 
            'languages',
            'page_number',
            'text',
            'embeddings',
            'parent_id',
            'is_continuation',
            'orig_elements',
            'element_type',
            'coordinates',
            'link_texts',
            'link_urls',
            'email_message_id',
            'sent_from',
            'sent_to',
            'subject',
            'url',
            'version',
            'date_created', 
            'date_modified', 
            'date_processed',
            'text_as_html',
            'emphasized_text_contents',
            'emphasized_text_tags',
            'documents_source'
        ]
        
        # 按照正确的顺序构建SELECT列表
        select_expressions = []
        for col in all_columns_ordered:
            if col in transformed_columns:
                # 使用转换后的表达式
                select_expressions.append(transformed_columns[col])
            else:
                # 使用原始列
                select_expressions.append(col)
        
        # 构建SQL - 使用workspace前缀（如果提供）
        if workspace:
            silver_table_full = f"{workspace}.{schema_name}.{silver_table}"
            raw_table_full = f"{workspace}.{schema_name}.{raw_table}"
        else:
            silver_table_full = f"{schema_name}.{silver_table}"
            raw_table_full = f"{schema_name}.{raw_table}"
        
        sql_parts = [
            f"-- Generated by kb_transformation_rules.py TransformationRuleEngine.generate_transformation_sql()",
            f"-- Total columns: {len(select_expressions)}, Transformed columns: {len(transformed_columns)}",
            f"-- UNIQUE MARKER: KB_TRANS_RULES_2024",
            f"INSERT OVERWRITE {silver_table_full}",
            "SELECT",
            "    " + ",\n    ".join(select_expressions),
            f"FROM {raw_table_full}"
        ]
        
        # 添加WHERE条件
        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))
        
        result_sql = "\n".join(sql_parts)
        
        # 分析生成的SQL列数
        sql_lines = result_sql.split('\n')
        select_lines = []
        in_select = False
        for line in sql_lines:
            if 'SELECT' in line:
                in_select = True
                continue
            if 'FROM' in line:
                break
            if in_select and line.strip() and not line.strip().startswith('--'):
                select_lines.append(line.strip().rstrip(','))
        
        # 移除调试日志 - 列数验证
        
        return result_sql
    
    def _generate_transform_expression_with_base(self, rule: Dict, base_expr: str) -> Optional[str]:
        """生成转换表达式（支持链式转换）
        
        Args:
            rule: 转换规则
            base_expr: 基础表达式（可能是列名或之前的转换结果）
            
        Returns:
            SQL表达式
        """
        column = rule.get('column', '')
        operation = rule.get('operation', '')
        params = rule.get('params', {})
        
        if not column:
            return None
        
        # 如果是预定义的操作
        if operation in self.operation_templates:
            template = self.operation_templates[operation]
            
            # 处理带参数的模板
            if '{old_text}' in template or '{new_text}' in template:
                # 自定义替换操作
                if 'old_text' in params and 'new_text' in params:
                    expr = template.format(
                        column=base_expr,  # 使用基础表达式而不是列名
                        old_text=params['old_text'],
                        new_text=params['new_text']
                    )
                    return f"{expr} AS {column}"
                else:
                    logger.warning(f"缺少必需的参数: old_text 或 new_text")
                    return None
            else:
                # 不需要额外参数的操作
                expr = template.format(column=base_expr)  # 使用基础表达式
                return f"{expr} AS {column}"
        
        # 如果是自定义SQL表达式
        if operation:
            # 简单验证，避免SQL注入
            if self._is_safe_sql_expression(operation):
                # 替换{column}占位符
                expr = operation.replace('{column}', base_expr)  # 使用基础表达式
                # 替换其他参数
                for key, value in params.items():
                    expr = expr.replace(f'{{{key}}}', str(value))
                return f"{expr} AS {column}"
        
        return None
    
    def _generate_transform_expression(self, rule: Dict) -> Optional[str]:
        """生成转换表达式
        
        Args:
            rule: 转换规则
            
        Returns:
            SQL表达式
        """
        column = rule.get('column', '')
        operation = rule.get('operation', '')
        params = rule.get('params', {})
        
        if not column:
            return None
        
        # 如果是预定义的操作
        if operation in self.operation_templates:
            template = self.operation_templates[operation]
            
            # 处理带参数的模板
            if '{old_text}' in template or '{new_text}' in template:
                # 自定义替换操作
                if 'old_text' in params and 'new_text' in params:
                    expr = template.format(
                        column=column,
                        old_text=params['old_text'],
                        new_text=params['new_text']
                    )
                    return f"{expr} AS {column}"
                else:
                    logger.warning(f"缺少必需的参数: old_text 或 new_text")
                    return None
            else:
                # 不需要额外参数的操作
                expr = template.format(column=column)
                return f"{expr} AS {column}"
        
        # 如果是自定义SQL表达式
        if operation:
            # 简单验证，避免SQL注入
            if self._is_safe_sql_expression(operation):
                # 替换{column}占位符
                expr = operation.replace('{column}', column)
                # 替换其他参数
                for key, value in params.items():
                    expr = expr.replace(f'{{{key}}}', str(value))
                return f"{expr} AS {column}"
        
        return None
    
    def _generate_filter_condition(self, rule: Dict) -> Optional[str]:
        """生成过滤条件
        
        Args:
            rule: 过滤规则
            
        Returns:
            SQL条件表达式
        """
        condition_type = rule.get('condition_type', '')
        condition = rule.get('condition', '')
        
        # 如果是预定义的条件类型
        if condition_type in self.filter_templates:
            template = self.filter_templates[condition_type]
            params = rule.get('params', {})
            
            # 替换参数
            try:
                condition = template.format(**params)
                return condition
            except KeyError as e:
                logger.error(f"过滤条件参数错误: {e}")
                return None
        
        # 如果是自定义条件
        if condition:
            # 简单验证
            if self._is_safe_sql_expression(condition):
                return condition
        
        return None
    
    def _generate_filter_group_condition(self, rule: Dict) -> Optional[str]:
        """生成过滤组条件
        
        Args:
            rule: 过滤组规则
            
        Returns:
            SQL条件表达式
        """
        operator = rule.get('operator', 'AND')
        conditions = rule.get('conditions', [])
        
        if not conditions:
            return None
        
        # 处理每个子条件
        condition_parts = []
        for condition in conditions:
            if condition.get('type') == 'condition':
                # 单个条件
                sub_condition = self._generate_filter_condition(condition)
                if sub_condition:
                    condition_parts.append(sub_condition)
            elif condition.get('type') == 'filter_group':
                # 嵌套的过滤组
                sub_group = self._generate_filter_group_condition(condition)
                if sub_group:
                    condition_parts.append(f"({sub_group})")
        
        if not condition_parts:
            return None
        
        # 用指定的操作符连接条件
        return f" {operator} ".join(condition_parts)
    
    def _is_safe_sql_expression(self, expression: str) -> bool:
        """简单的SQL表达式安全检查
        
        Args:
            expression: SQL表达式
            
        Returns:
            是否安全
        """
        # 禁止的关键字
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 
            'INSERT', 'UPDATE', 'GRANT', 'REVOKE', 'EXEC', 
            'EXECUTE', ';', '--', '/*', '*/'
        ]
        
        expression_upper = expression.upper()
        for keyword in dangerous_keywords:
            if keyword in expression_upper:
                logger.warning(f"检测到危险SQL关键字: {keyword}")
                return False
        
        return True
    
    def validate_rules(self, rules: List[Dict]) -> Tuple[bool, List[str]]:
        """验证转换规则
        
        Args:
            rules: 规则列表
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        if not isinstance(rules, list):
            errors.append("规则必须是列表类型")
            return False, errors
        
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"规则[{i}]必须是字典类型")
                continue
            
            # 检查必需字段
            if 'type' not in rule:
                errors.append(f"规则[{i}]缺少type字段")
            elif rule['type'] not in ['transform', 'filter']:
                errors.append(f"规则[{i}]的type必须是transform或filter")
            
            if 'name' not in rule:
                errors.append(f"规则[{i}]缺少name字段")
            
            # 检查转换规则
            if rule.get('type') == 'transform':
                if 'column' not in rule:
                    errors.append(f"转换规则[{i}]缺少column字段")
                if 'operation' not in rule:
                    errors.append(f"转换规则[{i}]缺少operation字段")
            
            # 检查过滤规则
            elif rule.get('type') == 'filter':
                if 'condition' not in rule and 'condition_type' not in rule:
                    errors.append(f"过滤规则[{i}]必须包含condition或condition_type")
        
        return len(errors) == 0, errors
    
    def get_operation_list(self) -> List[Dict[str, str]]:
        """获取所有可用的转换操作列表
        
        Returns:
            操作列表
        """
        operations = []
        for key, template in self.operation_templates.items():
            operations.append({
                'key': key,
                'name': self._get_operation_name(key),
                'description': self._get_operation_description(key),
                'template': template
            })
        return operations
    
    def get_operations_by_category(self) -> Dict[str, List[Dict[str, str]]]:
        """按分类获取转换操作
        
        Returns:
            分类操作字典
        """
        categories = {
            '基础文本处理': ['trim', 'trim_left', 'trim_right', 'lowercase', 'uppercase'],
            'HTML和标记清理': ['remove_html', 'remove_urls', 'remove_emails'],
            '空白字符处理': ['normalize_spaces', 'remove_newlines', 'remove_tabs'],
            '特殊字符处理': ['keep_alphanumeric', 'keep_alphanumeric_chinese', 'remove_special_chars'],
            '数字处理': ['remove_numbers', 'extract_numbers'],
            '标点符号处理': ['normalize_chinese_punctuation', 'keep_basic_punctuation'],
            '字符串替换': ['remove_quotes', 'replace_underscores']
        }
        
        result = {}
        for category, keys in categories.items():
            result[category] = []
            for key in keys:
                if key in self.operation_templates:
                    result[category].append({
                        'key': key,
                        'name': self._get_operation_name(key),
                        'description': self._get_operation_description(key),
                        'template': self.operation_templates[key]
                    })
        
        return result
    
    def get_filter_list(self) -> List[Dict[str, str]]:
        """获取所有可用的过滤条件列表
        
        Returns:
            过滤条件列表
        """
        filters = []
        for key, template in self.filter_templates.items():
            filters.append({
                'key': key,
                'name': self._get_filter_name(key),
                'description': self._get_filter_description(key),
                'template': template
            })
        return filters
    
    def _get_operation_name(self, key: str) -> str:
        """获取操作的友好名称"""
        names = {
            # 基础文本处理
            'trim': '去除首尾空格',
            'trim_left': '去除左侧空格', 
            'trim_right': '去除右侧空格',
            'lowercase': '转换为小写',
            'uppercase': '转换为大写',
            
            # HTML和标记清理
            'remove_html': '移除HTML标签',
            'remove_urls': '移除URL链接',
            'remove_emails': '移除邮箱地址',
            
            # 空白字符处理
            'normalize_spaces': '标准化空格',
            'remove_newlines': '移除换行符',
            'remove_tabs': '移除制表符',
            
            # 特殊字符处理
            'keep_alphanumeric': '仅保留字母数字',
            'keep_alphanumeric_chinese': '仅保留中英文和数字',
            'remove_special_chars': '移除特殊字符',
            
            # 数字处理
            'remove_numbers': '移除数字',
            'extract_numbers': '提取数字',
            
            # 标点符号处理
            'normalize_chinese_punctuation': '标准化中文标点',
            'keep_basic_punctuation': '保留基础标点',
            
            # 字符串替换
            'remove_quotes': '移除引号',
            'replace_underscores': '下划线转空格'
        }
        return names.get(key, key)
    
    def _get_operation_description(self, key: str) -> str:
        """获取操作的描述"""
        descriptions = {
            # 基础文本处理
            'trim': '去除文本首尾的空格字符',
            'trim_left': '去除文本左侧的空格字符',
            'trim_right': '去除文本右侧的空格字符',
            'lowercase': '将所有字母转换为小写',
            'uppercase': '将所有字母转换为大写',
            
            # HTML和标记清理
            'remove_html': '移除文本中的HTML标签',
            'remove_urls': '移除文本中的HTTP/HTTPS链接',
            'remove_emails': '移除文本中的邮箱地址',
            
            # 空白字符处理
            'normalize_spaces': '将连续的空白字符替换为单个空格',
            'remove_newlines': '移除文本中的换行符',
            'remove_tabs': '移除文本中的制表符',
            
            # 特殊字符处理
            'keep_alphanumeric': '只保留英文字母和数字，移除其他字符',
            'keep_alphanumeric_chinese': '只保留中文、英文字母和数字',
            'remove_special_chars': '移除特殊符号字符',
            
            # 数字处理
            'remove_numbers': '移除文本中的所有数字',
            'extract_numbers': '提取文本中的第一个数字',
            
            # 标点符号处理
            'normalize_chinese_punctuation': '将中文标点符号替换为空格',
            'keep_basic_punctuation': '只保留基础标点符号',
            
            # 字符串替换
            'remove_quotes': '移除文本中的单引号和双引号',
            'replace_underscores': '将下划线替换为空格'
        }
        return descriptions.get(key, '')
    
    def _get_filter_name(self, key: str) -> str:
        """获取过滤条件的友好名称"""
        names = {
            'not_null': '非空',
            'not_empty': '非空字符串',
            'min_length': '最小长度',
            'max_length': '最大长度',
            'between_length': '长度范围',
            'has_chinese': '包含中文',
            'has_english': '包含英文'
        }
        return names.get(key, key)
    
    def _get_filter_description(self, key: str) -> str:
        """获取过滤条件的描述"""
        descriptions = {
            'not_null': '过滤掉NULL值',
            'not_empty': '过滤掉空字符串',
            'min_length': '文本长度必须大于等于指定值',
            'max_length': '文本长度必须小于等于指定值',
            'between_length': '文本长度必须在指定范围内',
            'has_chinese': '文本必须包含中文字符',
            'has_english': '文本必须包含英文字符'
        }
        return descriptions.get(key, '')
    
    def preview_transformation(self, text: str, rules: List[Dict]) -> str:
        """预览转换效果（仅用于演示，不执行实际SQL）
        
        Args:
            text: 示例文本
            rules: 转换规则
            
        Returns:
            转换后的文本
        """
        result = text
        
        for rule in rules:
            if rule.get('type') != 'transform':
                continue
            
            operation = rule.get('operation', '')
            
            # 模拟一些简单的转换
            if operation == 'trim':
                result = result.strip()
            elif operation == 'lowercase':
                result = result.lower()
            elif operation == 'uppercase':
                result = result.upper()
            elif operation == 'normalize_spaces':
                result = ' '.join(result.split())
            elif operation == 'remove_html':
                result = re.sub(r'<[^>]+>', '', result)
            elif operation == 'remove_urls':
                result = re.sub(r'https?://[^\s]+', '', result)
        
        return result