# astro-blog 项目长期备忘

## 站点与部署
- 线上域名 https://blog.halo26812.eu.org（astro.config site），Vercel 部署约 2-4 分钟。
- 文章页 URL 返回 308 是尾斜杠重定向，curl 必须加 -L；quant-column 有 CDN 缓存，验证时加 ?cb= 查询串。

## 内容 schema 硬约束
- frontmatter `description` ≤ 600 字符（content collection schema，超长直接 build 失败）。
- 每篇文章标配：5 张真实计算 PNG 于 public/images/{slug}/，结果 JSON 与图同目录，正文数字必须与 JSON 一致。

## 本机沙箱坑
- `.vercel/output` 有旧产物时 build 会触发 safe-delete 拦截（count>50）；先 `mv .vercel/output` 挪开再 build。
