#!/usr/bin/env node
/**
 * 修补构建后的 HTML 文件，将硬编码的英文字符串替换为中文
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.join(__dirname, '..', 'dist', 'client');
const vercelDir = path.join(__dirname, '..', '.vercel', 'output', 'static');

const replacements = [
  // 分页
  [/←\s*Previous\s*(Posts)?/g, '← 上一页'],
  [/Next\s*(Posts)?\s*→/g, '下一页 →'],
  // Tags
  [/title="Tags"/g, 'title="标签"'],
  [/aria-label="Tags"/g, 'aria-label="标签"'],
  [/aria-labelledby="Tags"/g, 'aria-labelledby="标签"'],
  // Read more
  [/Read\s*more/gi, '阅读更多'],
  // Posted on
  [/Posted\s*on/gi, '发布于'],
  // Search
  [/placeholder="Search"/gi, 'placeholder="搜索"'],
  [/Search\s*…/g, '搜索…'],
  // RSS
  [/Subscribe to RSS/gi, '订阅 RSS'],
  // Powered by
  [/Powered\s*by/gi, '由'],
  // Archive
  [/Archive/g, '归档'],
];

function walkDir(dir) {
  if (!fs.existsSync(dir)) {
    console.log(`  ⚠️  Directory not found: ${path.relative(__dirname, dir)}`);
    return;
  }
  
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
    console.log(`  ✓ Patched: ${path.relative(path.join(__dirname, '..'), filePath)}`);
  }
}

console.log('Patching i18n strings in built HTML files...\n');

console.log('Patching dist/client/...');
walkDir(distDir);

if (fs.existsSync(vercelDir)) {
  console.log('\nPatching .vercel/output/static/...');
  walkDir(vercelDir);
} else {
  console.log('\n⚠️  Skipping .vercel/output/static/ (not found)');
}

console.log('\nDone!');
