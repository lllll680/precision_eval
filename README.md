# precision_eval
71. Name: node_check
Description: 检查节点状态和配置项
Parameters: {"properties": {"node": {"type": "string", "title": "Node", "description": "节点名称或标识"}, "check_item": {"type": "string", "title": "Check Item", "description": "检查项目名称"}}, "required": ["node", "check_item"], "type": "object"}
Output: {"properties": {"状态": {"type": "string", "description": "节点当前状态"}, "原因": {"type": "string", "description": "状态原因说明"}, "修复建议": {"type": "string", "description": "针对当前状态的修复建议"}}, "type": "object"}
