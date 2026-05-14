from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def register_chinese_font() -> str:
    """注册中文字体，避免 PDF 中中文标题、正文和表格显示为方块。

    服务器环境不一定安装 Noto CJK 或 Windows 字体；ReportLab 内置的
    STSong-Light 是可用的中文 CID 字体，适合作为跨平台兜底方案。
    如果系统安装了更完整的 TrueType/OpenType 中文字体，则优先嵌入这些字体；
    否则使用 CID 字体，保证中文文本至少可以正常渲染。
    """
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont("ReportChinese", str(font_path)))
                return "ReportChinese"
            except Exception:
                continue

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        pass
    return "Helvetica"


def build_story(results_dir: Path) -> list:
    """把训练结果目录组织成实验报告内容。"""
    story = []
    font_name = register_chinese_font()
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = font_name
    styles.add(
        ParagraphStyle(
            name="CenterTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName=font_name,
        )
    )
    story.append(Paragraph("Stanford Background Dataset 语义分割实验报告", styles["CenterTitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "本报告汇总 U-Net 在 Cross-Entropy、Dice Loss 与组合损失三种配置下的训练和测试结果。",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("数据集与模型", styles["Heading2"]))
    story.append(
        Paragraph(
            "数据集使用 Stanford Background Dataset，包含 sky、tree、road、grass、water、building、mountain 和 foreground object 共 8 类；负数或越界标签作为 unknown 像素，在损失和评估中忽略。模型为从零训练的经典 U-Net，包含编码器、解码器和同尺度 skip connection 拼接。",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    summary_rows = [["实验", "损失函数", "Batch", "学习率", "Epoch", "验证 mIoU", "测试 Acc", "测试 mIoU"]]
    for run_dir in sorted(results_dir.glob("unet_*")):
        summary = run_dir / "summary.json"
        config_path = run_dir / "config.yaml"
        if not summary.exists() or not config_path.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        test_metrics = run_dir / "test" / "metrics.json"
        test_data = json.loads(test_metrics.read_text(encoding="utf-8")) if test_metrics.exists() else {}
        summary_rows.append(
            [
                run_dir.name,
                config["train"]["loss"],
                config["data"]["batch_size"],
                config["train"]["learning_rate"],
                config["train"]["epochs"],
                f"{data.get('best_val_mean_iou', 0):.4f}",
                f"{test_data.get('pixel_accuracy', 0):.4f}" if test_data else "未评估",
                f"{test_data.get('mean_iou', 0):.4f}" if test_data else "未评估",
            ]
        )
    table = Table(summary_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.5 * cm))

    for run_dir in sorted(results_dir.glob("unet_*")):
        story.append(Paragraph(f"{run_dir.name} 实验曲线", styles["Heading2"]))
        for filename in ["loss.png", "accuracy.png", "miou.png"]:
            image_path = run_dir / filename
            if image_path.exists():
                story.append(Image(str(image_path), width=16 * cm, height=9 * cm))
                story.append(Spacer(1, 0.2 * cm))
        test_dir = run_dir / "test"
        if test_dir.exists():
            story.append(Paragraph("测试集可视化", styles["Heading3"]))
            vis = sorted(test_dir.glob("prediction_*.png"))[:3]
            for image_path in vis:
                story.append(Image(str(image_path), width=5 * cm, height=5 * cm))
        story.append(PageBreak())

    story.append(Paragraph("结论", styles["Heading2"]))
    story.append(
        Paragraph(
            "Cross-Entropy 通常收敛稳定，Dice 更适合缓解类别不均衡，组合损失往往在精度与区域重叠之间取得更均衡的表现。",
            styles["BodyText"],
        )
    )
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description="根据训练结果生成 PDF 实验报告。")
    parser.add_argument("--results", default="runs", help="训练结果目录。")
    parser.add_argument("--output", default="reports/experiment_report.pdf", help="PDF 输出路径。")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm)
    story = build_story(results_dir)
    doc.build(story)
    print(f"报告已生成：{output}")


if __name__ == "__main__":
    main()
