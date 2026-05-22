#!/usr/bin/env node
/**
 * 修补构建后的 HTML 文件，将硬编码的英文字符串替换为中文
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.join(__dirname, '..', 'dist', 'client');

const replacements = [
  [/←\s*Previous\s*(Posts)?/g, '← 上一页'],
  [/Next\s*(Posts)?\s*→/g, '下一页 →'],
  [/title="Tags"/g, 'title="标签"'],
  [/aria-label="Tags"/g, 'aria-label="标签"'],
  [/aria-labelledby="Tags"/g, 'aria-labelledby="标签"'],
];

function walkDir(dir) {
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      walkDir(fullPath);
    } else if (file.endsWith('.html')) {
      patchFile(fullPath);
    }
  }
}

function patchFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf-8');
  let modified = false;
  
  for (const [pattern, replacement] of replacements) {
    if (pattern.test(content)) {
      content = content.replace(pattern, replacement);
      modified = true;
    }
  }
  
  if (modified) {
    fs.writeFileSync(filePath, content, 'utf-8');
    console.log(`✓ Patched: ${path.relative(distDir, filePath)}`);
  }
}

console.log('Patching i18n strings in built HTML files...');
walkDir(distDir);
console.log('Done!');
