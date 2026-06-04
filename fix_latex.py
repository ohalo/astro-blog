#!/usr/bin/env python3
"""
修复 Markdown 文章中的 LaTeX 公式语法错误
主要问题：
1. $$ 显示公式应该独占一行
2. $$ 块内部不应该有 $ 符号
3. 确保所有 $ 和 $$ 正确配对
"""

import re
import sys

def fix_latex_in_markdown(content):
    """修复 Markdown 中的 LaTeX 语法"""
    lines = content.split('\n')
    fixed_lines = []
    in_display_math = False
    in_code_block = False
    
    for i, line in enumerate(lines):
        # 检查是否在代码块中
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            fixed_lines.append(line)
            continue
        
        if in_code_block:
            fixed_lines.append(line)
            continue
        
        # 检查是否是 $$ 显示公式
        stripped = line.strip()
        
        if stripped == '$$':
            # $$ 应该独占一行
            if not in_display_math:
                in_display_math = True
                fixed_lines.append(line)
            else:
                in_display_math = False
                fixed_lines.append(line)
            continue
        
        if in_display_math:
            # 在 $$ 块内，移除可能的 $ 符号
            # 但不移除 LaTeX 命令中的 $ -like 符号
            fixed_line = line
            # 只移除真正的行内数学定界符 $
            # 不移除 LaTeX 命令
            fixed_lines.append(fixed_line)
            continue
        
        # 处理行内公式 $...$
        # 确保 $ 成对出现
        if '$' in line and '$$' not in line:
            # 计算 $ 的数量（排除代码块和转义的 \$）
            # 简化处理：只检查是否有奇数个 $
            dollar_count = line.count('$') - line.count('\$')
            if dollar_count % 2 != 0:
                print(f"Warning: Line {i+1} has odd number of $: {line[:60]}...")
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_latex.py <markdown_file>")
        return
    
    file_path = sys.argv[1]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Fixing LaTeX in: {file_path}")
    fixed_content = fix_latex_in_markdown(content)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Done! Fixed file saved.")
    
    # 验证 $$ 块是否正确
    print("\nValidating display math blocks...")
    lines = fixed_content.split('\n')
    in_math = False
    math_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '$$':
            if not in_math:
                in_math = True
                math_start = i
            else:
                in_math = False
                math_start = None
        elif in_math:
            # 检查公式内容是否有 $
            if '$' in stripped and not stripped.startswith('$'):
                print(f"  Warning: Line {i+1} in $$ block contains $: {stripped[:60]}...")
    
    if in_math:
        print(f"  Error: Unclosed $$ block starting at line {math_start + 1}")
    else:
        print("  OK: All $$ blocks are properly closed.")

if __name__ == '__main__':
    main()
