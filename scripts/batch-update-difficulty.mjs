import matter from 'gray-matter';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 分类规则（基于标题和标签关键词）
const RULES = {
  beginner: [
    '基础', '入门', '介绍', '初学', '概览', '工具链', '工具', '框架',
    'α', '阿尔法', '贝塔', '量化交易盈利', 'Python量化工具',
    '股息', '指数基金', '茅台', '可转债'
  ],
  intermediate: [
    '策略', '回测', '因子', '多因子', '选股',
    '风险管理', '止损', '仓位', '最大回撤', 'VaR', 'CVaR',
    '统计套利', '配对交易', '协整',
    '行为金融', '另类数据',
    'Black-Litterman', '风险平价', '投资组合', '资产配置',
    '期权', '波动率', 'Delta', '备兑'
  ],
  advanced: [
    '机器学习', '深度学习', 'LSTM', 'Transformer', '强化学习', 'DQN', 'PPO',
    '高频交易', '微结构', '订单流', '限价订单簿',
    '实盘', '执行算法', 'VWAP', 'TWAP', '滑点',
    '小波变换', '状态切换', '因子衰减', '因子拥挤',
    '深度强化', '随机森林', '因子正交'
  ]
};

// 判断文章难度
function classifyArticle(title, tags = []) {
  const text = title + ' ' + tags.join(' ');
  
  // 检查高级关键词
  if (RULES.advanced.some(keyword => text.includes(keyword))) {
    return 'advanced';
  }
  
  // 检查进阶级关键词
  if (RULES.intermediate.some(keyword => text.includes(keyword))) {
    return 'intermediate';
  }
  
  // 检查入门级关键词
  if (RULES.beginner.some(keyword => text.includes(keyword))) {
    return 'beginner';
  }
  
  // 默认：无法判断
  return null;
}

// 主函数
async function main() {
  const blogDir = path.join(__dirname, '../src/content/blog');
  const articles = fs.readdirSync(blogDir);
  
  let updated = 0;
  let skipped = 0;
  let errors = 0;
  const results = [];
  
  console.log(`开始扫描 ${articles.length} 篇文章...\n`);
  
  for (const article of articles) {
    const articlePath = path.join(blogDir, article, 'index.md');
    
    // 跳过不存在 index.md 的文章
    if (!fs.existsSync(articlePath)) {
      skipped++;
      continue;
    }
    
    try {
      const file = fs.readFileSync(articlePath, 'utf-8');
      const { data, content } = matter(file);
      
      // 只处理量化相关文章
      const isQuant = data.tags?.some(tag => 
        tag.includes('量化') || 
        tag.includes('quant') ||
        tag.includes('交易') ||
        tag.includes('投资')
      );
      
      if (!isQuant) {
        skipped++;
        continue;
      }
      
      // 如果已经有 difficulty 字段，跳过
      if (data.difficulty) {
        console.log(`⏭️  跳过（已有分类）: ${data.title} [${data.difficulty}]`);
        skipped++;
        continue;
      }
      
      // 分类
      const difficulty = classifyArticle(data.title, data.tags);
      
      if (!difficulty) {
        console.log(`❓ 无法分类: ${data.title}`);
        skipped++;
        continue;
      }
      
      // 更新 frontmatter
      data.difficulty = difficulty;
      
      // 写回文件
      const newFile = matter.stringify(content, data);
      fs.writeFileSync(articlePath, newFile, 'utf-8');
      
      console.log(`✅ 已更新: ${data.title} [${difficulty}]`);
      updated++;
      results.push({ title: data.title, difficulty, slug: article });
      
    } catch (err) {
      console.error(`❌ 错误: ${article}`, err.message);
      errors++;
    }
  }
  
  console.log('\n=== 完成 ===');
  console.log(`✅ 已更新: ${updated}`);
  console.log(`⏭️  已跳过: ${skipped}`);
  console.log(`❌ 错误: ${errors}`);
  
  // 保存结果到 JSON
  fs.writeFileSync(
    path.join(__dirname, 'difficulty-update-results.json'),
    JSON.stringify(results, null, 2)
  );
  console.log('\n结果已保存到: scripts/difficulty-update-results.json');
}

main();
