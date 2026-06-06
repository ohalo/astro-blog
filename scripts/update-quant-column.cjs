#!/usr/bin/env node
/**
 * 自动更新量化专栏数据文件
 * 扫描 src/content/blog/ 下的所有文章，提取量化相关的文章信息
 */

const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

const BLOG_DIR = path.join(__dirname, '..', 'src', 'content', 'blog');
const OUTPUT_FILE = path.join(__dirname, '..', 'src', 'data', 'quant-column.ts');

// 判断是否是量化相关的文章
function isQuantArticle(frontmatter) {
  const tags = frontmatter.tags || [];
  const title = frontmatter.title || '';
  const desc = frontmatter.description || '';
  
  // 检查 tags
  if (tags.some(tag => tag.includes('量化'))) {
    return true;
  }
  
  // 检查标题和描述中的关键词
  const keywords = ['量化', '因子', '策略', '回测', '风险', '投资组合', '期权', '波动率', '统计套利', '配对交易', '机器学习', 'LSTM', '随机森林', '高频', '订单簿', '算法交易'];
  const text = title + desc;
  
  return keywords.some(keyword => text.includes(keyword));
}

// 根据标题和内容判断难度级别
function estimateDifficulty(title, desc, content) {
  const text = (title + desc + content).toLowerCase();
  
  // 高级关键词
  const advancedKeywords = ['lstm', 'transformer', '深度学习', '强化学习', '高频交易', '订单流', '限价订单簿', '小波变换', '实盘交易系统', '执行算法', '滑点控制'];
  if (advancedKeywords.some(kw => text.includes(kw.toLowerCase()))) {
    return 'advanced';
  }
  
  // 中级关键词
  const intermediateKeywords = ['统计套利', '配对交易', '协整', '风险平价', 'black-litterman', '期权', '波动率', '另类数据', '因子衰减', '行为金融'];
  if (intermediateKeywords.some(kw => text.includes(kw.toLowerCase()))) {
    return 'intermediate';
  }
  
  // 默认为入门级
  return 'beginner';
}

// 估算阅读时间（分钟）
function estimateReadTime(content) {
  const words = content.split(/\s+/).length;
  // 中文阅读速度约 300 字/分钟，英文约 200 词/分钟
  // 技术文章打 7 折
  const minutes = Math.ceil(words / 300 * 0.7);
  return Math.max(minutes, 10); // 最少 10 分钟
}

// 主函数
async function main() {
  console.log('开始扫描量化文章...');
  
  // 读取所有文章文件夹
  const dirs = fs.readdirSync(BLOG_DIR).filter(dir => {
    return fs.statSync(path.join(BLOG_DIR, dir)).isDirectory();
  });
  
  console.log(`找到 ${dirs.length} 个文章文件夹`);
  
  const articles = [];
  
  for (const dir of dirs) {
    const mdPath = path.join(BLOG_DIR, dir, 'index.md');
    
    if (!fs.existsSync(mdPath)) {
      continue;
    }
    
    try {
      const fileContent = fs.readFileSync(mdPath, 'utf-8');
      const { data: frontmatter, content } = matter(fileContent);
      
      // 检查是否是量化相关
      if (!isQuantArticle(frontmatter)) {
        continue;
      }
      
      const title = frontmatter.title || dir;
      const publishDate = frontmatter.publishDate || '2026-01-01';
      const description = frontmatter.description || title;
      const difficulty = estimateDifficulty(title, description, content);
      const difficultyLabel = difficulty === 'beginner' ? '入门' : difficulty === 'intermediate' ? '中级' : '高级';
      const estimatedReadTime = estimateReadTime(content);
      
      articles.push({
        slug: dir,
        title,
        difficulty,
        difficultyLabel,
        description: description.replace(/ - halo的技术博客$/, ''),
        publishDate: publishDate.substring(0, 10),
        estimatedReadTime
      });
      
      console.log(`  ✓ ${dir}`);
    } catch (err) {
      console.error(`  ✗ 错误: ${dir}`, err.message);
    }
  }
  
  // 按发布日期倒序排序
  articles.sort((a, b) => b.publishDate.localeCompare(a.publishDate));
  
  console.log(`\n共找到 ${articles.length} 篇量化相关文章`);
  
  // 生成 TypeScript 文件内容
  const tsContent = `// 量化交易专栏 - 文章难度分类与目录结构
// 按难度从易到难排序
// 自动生成于 ${new Date().toISOString()}

export interface ColumnArticle {
  slug: string;
  title: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  difficultyLabel: string;
  description: string;
  publishDate: string;
  estimatedReadTime: number; // 分钟
}

export const quantColumnArticles: ColumnArticle[] = ${JSON.stringify(articles, null, 2)
    .replace(/"([^"]+)":/g, '$1:') // 去掉 key 的引号
    .replace(/"/g, "'") // 把双引号改为单引号
    .replace(/'slug':/g, 'slug:')
    .replace(/'title':/g, 'title:')
    .replace(/'difficulty':/g, 'difficulty:')
    .replace(/'difficultyLabel':/g, 'difficultyLabel:')
    .replace(/'description':/g, 'description:')
    .replace(/'publishDate':/g, 'publishDate:')
    .replace(/'estimatedReadTime':/g, 'estimatedReadTime:')
  };

// 按难度分组
export function getArticlesByDifficulty() {
  const beginner = quantColumnArticles.filter(a => a.difficulty === 'beginner');
  const intermediate = quantColumnArticles.filter(a => a.difficulty === 'intermediate');
  const advanced = quantColumnArticles.filter(a => a.difficulty === 'advanced');
  
  return {
    beginner,
    intermediate,
    advanced,
    total: quantColumnArticles.length
  };
}

// 获取总阅读时间
export function getTotalReadTime() {
  return quantColumnArticles.reduce((sum, article) => sum + article.estimatedReadTime, 0);
}
`;
  
  // 写入文件
  fs.writeFileSync(OUTPUT_FILE, tsContent, 'utf-8');
  
  console.log(`\n✅ 成功更新 ${OUTPUT_FILE}`);
  console.log(`   - 入门级: ${articles.filter(a => a.difficulty === 'beginner').length} 篇`);
  console.log(`   - 中级: ${articles.filter(a => a.difficulty === 'intermediate').length} 篇`);
  console.log(`   - 高级: ${articles.filter(a => a.difficulty === 'advanced').length} 篇`);
}

main().catch(console.error);
