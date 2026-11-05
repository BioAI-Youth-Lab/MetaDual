"""Markdown parser and function blocks extracted from the original prompt evaluator."""
import re
from typing import List, Dict, Union

def parse_fcgs_markdown(markdown_text: str) -> List[Dict[str, Union[List[str], str]]]:
    """
    输入：LLM 返回的 Markdown 文本（包含 "### Functional Collaboration Groups" 段）
    输出：列表，每个元素 {"OTUs": [...], "Function": "..."}；顺序即出现顺序
    """
    text = (markdown_text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    sec_match = re.search('(?is)###\\s*Functional Collaboration Groups(.*?)(?:\\n###|\\Z)', text)
    seg = sec_match.group(1) if sec_match else text
    groups = []
    pat_format1 = re.compile('(?:^|\\n)(?:-\\s*)?\\*\\*Group\\s*(\\d+):\\s*([^*]+)\\*\\*\\s*\\n\\s*\\*\\*Function\\*\\*:\\s*([\\s\\S]+?)(?=\\n(?:-\\s*)?\\*\\*Group\\s*\\d+:|$)', re.IGNORECASE | re.MULTILINE)
    for m in pat_format1.finditer(seg):
        group_num, otus_text, func_block = m.groups()
        otus = re.split('[,;]+', otus_text)
        otus = [o.strip() for o in otus if o.strip()]
        function = re.sub('\\s+', ' ', func_block).strip()
        if otus and function:
            groups.append({'OTUs': otus, 'Function': function})
    if not groups:
        pat_format2_numbered = re.compile('(?:^|\\n)\\d+\\.\\s*\\*\\*Group\\s*\\d+\\*\\*:\\s*([^\\n]+?)\\s*\\n\\s*\\*\\*Function\\*\\*:\\s*([\\s\\S]+?)(?=\\n\\d+\\.\\s*\\*\\*Group\\s*\\d+|$)', re.IGNORECASE | re.MULTILINE)
        for m in pat_format2_numbered.finditer(seg):
            otus_line, func_block = m.groups()
            otus = re.split('[,;]+', otus_line)
            otus = [o.strip() for o in otus if o.strip()]
            function = re.sub('\\s+', ' ', func_block).strip()
            if otus and function:
                groups.append({'OTUs': otus, 'Function': function})
    if not groups:
        pat_format3 = re.compile('(?:^|\\n)(?:-\\s*)?\\*\\*Group\\s*\\d+\\*\\*:\\s*(.+?)\\n+.*?\\*\\*Function\\*\\*:\\s*([\\s\\S]+?)(?=\\n+(?:-\\s*)?\\*\\*Group|\\Z)', re.IGNORECASE | re.DOTALL)
        for m in pat_format3.finditer(seg):
            otus_line, func_block = m.groups()
            otus = re.split('[,;\\n]+', otus_line)
            otus = [o.strip().strip('*') for o in otus if o.strip()]
            function = re.sub('\\s+', ' ', func_block).strip()
            if otus and function:
                groups.append({'OTUs': otus, 'Function': function})
    if not groups:
        pat_format4 = re.compile('(?:^|\\n)(?:-\\s*)?\\*\\*Group\\s*\\d+(?::[^*]*)?\\*\\*(?:[^\\n]*\\n)?(?:[^\\n]*\\n)?(?:-\\s*)?\\*\\*Members?\\*\\*:\\s*([^\\n]+(?:\\n(?!\\*\\*)[^\\n]+)*?)(?:\\n(?:-\\s*)?\\*\\*Function\\*\\*:|\\n\\*\\*Function\\*\\*:)\\s*([\\s\\S]+?)(?=\\n(?:-\\s*)?\\*\\*Group\\s*\\d+|$)', re.IGNORECASE | re.MULTILINE)
        for m in pat_format4.finditer(seg):
            members_text, func_text = m.groups()
            otus = re.findall('\\*([^*]+)\\*', members_text)
            if not otus:
                otus = [o.strip() for o in re.split('[,;]', members_text) if o.strip()]
            function = re.sub('\\s+', ' ', func_text).strip()
            function = re.sub('\\*\\*Evidence\\*\\*:.*$', '', function, flags=re.IGNORECASE).strip()
            if otus and function:
                groups.append({'OTUs': otus, 'Function': function})
    return groups

def build_embedding_blocks(groups: List[Dict[str, Union[List[str], str]]], batch_size: int=5) -> List[str]:
    """
    将 10 个 FCG 的 Function 文本按每 batch_size=5 合并成若干段落字符串（默认两段）。
    每段用换行连接（更利于模型学到“句子边界”），也可改为空格连接。
    """
    blocks = []
    for i in range(0, len(groups), batch_size):
        batch = groups[i:i + batch_size]
        lines = [g['Function'] for g in batch]
        blocks.append('\n'.join(lines))
    return blocks
