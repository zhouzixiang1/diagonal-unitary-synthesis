# 杭州电子科技大学本科毕业设计（初步文稿）

基于 `HDU-Bachelor-Thesis` 模板。研究方案见仓库根目录 `开题报告.md`。

## 编译

要求：XeLaTeX + biber。编译产物统一输出到 `build/`。

```bash
cd LaTeX_bishe
latexmk -xelatex main.tex
```

生成 PDF：`build/main.pdf`。

清理：

```bash
latexmk -c
# 或删除整个 build 目录
```

## 主要文件

| 文件 | 说明 |
|------|------|
| `main.tex` | 毕业论文初稿（开题阶段骨架） |
| `ref.bib` | 参考文献 |
| `latexmkrc` | 将输出目录设为 `build/` |
| `example.tex` | 原模板样例（可参考排版用法） |
| `HDU-Bachelor-Thesis.cls` | 校级模板类文件 |

## 封面信息

请在 `main.tex` 中按实际填写姓名、学号、班级、专业、导师等 TODO 字段。
