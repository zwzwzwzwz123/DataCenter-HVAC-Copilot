"""Generate the GPU SFT training manual as a polished PDF (Chinese).

One-off utility: builds distill/SFT_训练操作手册_v2.pdf using reportlab with
Microsoft YaHei (body) + Consolas (code). Kept in-repo so the manual can be
regenerated if the steps change.

Layout notes:
- Body/headings use MSYH/MSYHBD (real Chinese glyphs, no tofu).
- Code blocks use Consolas and MUST stay pure ASCII — any Chinese explanation
  goes in surrounding paragraphs, never inside a code block. This is the rule
  that prevents garbled boxes.

    python distill/make_manual_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---- fonts ----
FONTS = {
    "body": ("MSYH", "C:/Windows/Fonts/msyh.ttc", 0),
    "bold": ("MSYHBD", "C:/Windows/Fonts/msyhbd.ttc", 0),
    "code": ("Consolas", "C:/Windows/Fonts/consola.ttf", None),
}

# ---- palette ----
INK = colors.HexColor("#1f2933")
BLUE = colors.HexColor("#1d4ed8")
BLUE_DK = colors.HexColor("#1e3a8a")
SLATE = colors.HexColor("#52606d")
ACCENT = colors.HexColor("#0e7490")
CODE_BG = colors.HexColor("#0f172a")
CODE_FG = colors.HexColor("#e2e8f0")
NOTE_BG = colors.HexColor("#fef6e7")
NOTE_BAR = colors.HexColor("#f59e0b")
CARD_BG = colors.HexColor("#f8fafc")
CARD_LINE = colors.HexColor("#e2e8f0")
GREEN = colors.HexColor("#15803d")


def _register():
    for name, path, idx in FONTS.values():
        if idx is None:
            pdfmetrics.registerFont(TTFont(name, path))
        else:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))


class HBar(Flowable):
    """A thin colored rule used as a section divider."""

    def __init__(self, width, thickness=1.2, color=CARD_LINE):
        super().__init__()
        self.width, self.thickness, self.color = width, thickness, color

    def wrap(self, *_):
        return self.width, self.thickness

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def _styles():
    return {
        "cover_title": ParagraphStyle(
            "ct", fontName="MSYHBD", fontSize=30, leading=40,
            textColor=colors.white, alignment=TA_CENTER),
        "cover_sub": ParagraphStyle(
            "cs", fontName="MSYH", fontSize=13, leading=22,
            textColor=colors.HexColor("#dbeafe"), alignment=TA_CENTER),
        "cover_meta": ParagraphStyle(
            "cm", fontName="MSYH", fontSize=10.5, leading=18,
            textColor=colors.HexColor("#bfdbfe"), alignment=TA_CENTER),
        "h1": ParagraphStyle(
            "h1", fontName="MSYHBD", fontSize=16, leading=22,
            textColor=colors.white),
        "h2": ParagraphStyle(
            "h2", fontName="MSYHBD", fontSize=12.5, leading=18,
            textColor=BLUE_DK, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle(
            "body", fontName="MSYH", fontSize=10.5, leading=17,
            textColor=INK, alignment=TA_LEFT, spaceAfter=4),
        "bullet": ParagraphStyle(
            "bullet", fontName="MSYH", fontSize=10.5, leading=16.5,
            textColor=INK, leftIndent=16, spaceAfter=3,
            bulletIndent=4, bulletFontName="MSYH", bulletColor=BLUE),
        "note": ParagraphStyle(
            "note", fontName="MSYH", fontSize=9.8, leading=15.5,
            textColor=colors.HexColor("#7c4a03")),
        "faq_q": ParagraphStyle(
            "fq", fontName="MSYHBD", fontSize=10.5, leading=16,
            textColor=ACCENT, spaceBefore=6, spaceAfter=1),
        "faq_a": ParagraphStyle(
            "fa", fontName="MSYH", fontSize=10, leading=15.5,
            textColor=INK, spaceAfter=2),
        "step_no": ParagraphStyle(
            "sn", fontName="MSYHBD", fontSize=15, leading=18,
            textColor=colors.white, alignment=TA_CENTER),
    }


def build(out_path: str) -> None:
    _register()
    S = _styles()
    PAGE_W = A4[0]
    content_w = PAGE_W - 40 * mm  # matches margins below
    E: list = []

    # ---------- helpers ----------
    def para(text, style="body"):
        E.append(Paragraph(text, S[style]))

    def bullets(items):
        for t in items:
            E.append(Paragraph(t, S["bullet"], bulletText="•"))
        E.append(Spacer(1, 4))

    def code(text):
        # dark rounded code panel; text is pure ASCII by construction
        pre = Preformatted(
            text,
            ParagraphStyle("codetxt", fontName="Consolas", fontSize=9.3,
                           leading=13.5, textColor=CODE_FG),
        )
        tbl = Table([[pre]], colWidths=[content_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("ROUNDEDCORNERS", [5, 5, 5, 5]),
        ]))
        E.append(Spacer(1, 2))
        E.append(tbl)
        E.append(Spacer(1, 7))

    def note(text):
        cell = Paragraph("💡 " + text, S["note"])
        tbl = Table([[cell]], colWidths=[content_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
            ("LINEBEFORE", (0, 0), (0, -1), 3, NOTE_BAR),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        E.append(Spacer(1, 2))
        E.append(tbl)
        E.append(Spacer(1, 8))

    def section(num, title):
        """Colored header band for a numbered step."""
        no = Paragraph(str(num), S["step_no"])
        no_tbl = Table([[no]], colWidths=[11 * mm], rowHeights=[11 * mm])
        no_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BLUE_DK),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        ttl = Paragraph(title, S["h1"])
        band = Table([[no_tbl, ttl]], colWidths=[11 * mm, content_w - 11 * mm],
                     rowHeights=[11 * mm])
        band.setStyle(TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), BLUE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (1, 0), (1, 0), 10),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        E.append(Spacer(1, 10))
        E.append(band)
        E.append(Spacer(1, 8))

    # ================= COVER =================
    def cover(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BLUE_DK)
        canvas.rect(0, 0, PAGE_W, A4[1], fill=1, stroke=0)
        # decorative band
        canvas.setFillColor(BLUE)
        canvas.rect(0, A4[1] - 300, PAGE_W, 150, fill=1, stroke=0)
        canvas.restoreState()

    cover_block = [
        Spacer(1, 55 * mm),
        Paragraph("SFT 训练操作手册", S["cover_title"]),
        Spacer(1, 6 * mm),
        Paragraph("Planner 决策蒸馏 · 阶段 2", S["cover_sub"]),
        Paragraph("面向零经验的一步步指南", S["cover_sub"]),
        Spacer(1, 40 * mm),
        Paragraph("基座模型　Qwen2.5-1.5B-Instruct", S["cover_meta"]),
        Paragraph("训练方法　QLoRA（4-bit 量化 + LoRA）", S["cover_meta"]),
        Paragraph("预计耗时　约 1 小时　·　预计花费　￥2–4", S["cover_meta"]),
    ]
    E.extend(cover_block)
    E.append(PageBreak())

    # ================= 0 概览 =================
    para("开始之前", "h2")
    para("这份手册假设你<b>没有</b>租过 GPU、没跑过模型训练。每一步都有可直接复制的命令。"
         "遇到红色报错先别慌——翻到最后的「常见问题」，多数都能自己解决。")
    note("全程真正在训练的时间只有十几分钟，大部分时间花在装依赖和下载模型上。整套跑完约 1 小时、花费 ￥2–4。")

    para("六步总览", "h2")
    steps_overview = [
        ["1", "租一张 RTX 4090（24G），选带 PyTorch 的镜像"],
        ["2", "把项目代码传到服务器"],
        ["3", "装训练依赖（一条命令）"],
        ["4", "设置模型下载加速（国内镜像）"],
        ["5", "运行训练（10–25 分钟）"],
        ["6", "看结果、下载产物、关机"],
    ]
    rows = []
    for n, t in steps_overview:
        rows.append([
            Paragraph(n, ParagraphStyle("on", fontName="MSYHBD", fontSize=12,
                                        textColor=colors.white, alignment=TA_CENTER)),
            Paragraph(t, S["body"]),
        ])
    ov = Table(rows, colWidths=[10 * mm, content_w - 10 * mm])
    ov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), ACCENT),
        ("BACKGROUND", (1, 0), (1, -1), CARD_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (1, 0), (1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (1, 0), (1, -2), 0.5, colors.white),
        ("GRID", (0, 0), (0, -1), 1, colors.white),
    ]))
    E.append(ov)

    # ================= 1 租卡 =================
    section(1, "租一张 GPU 卡")
    para("1.1　注册与充值", "h2")
    para("打开 AutoDL（autodl.com），注册账号并充值 ￥10（足够跑很多轮）。")
    para("1.2　挑选机器", "h2")
    bullets([
        "显卡：<b>RTX 4090（24G）</b> 优先；没有就选 RTX 3090 或任意 ≥16G 的卡。",
        "<b>不要</b>租 A100 / A800——1.5B 模型用不上，纯属浪费钱。",
        "计费方式：选「按量计费」，用完就关，4090 约 ￥1.5–2 / 小时。",
    ])
    para("1.3　选镜像（很重要，能省一半配置时间）", "h2")
    bullets([
        "在「基础镜像」里选 <b>PyTorch</b> 版本，例如 PyTorch 2.3 / Python 3.10 / CUDA 12.x。",
        "选了它，torch 就已经装好，后面只需再装几个包。",
    ])
    para("1.4　进入终端", "h2")
    para("创建成功后点「JupyterLab」进入，再打开里面的「Terminal / 终端」。"
         "后面所有命令都在这个终端里输入。")

    # ================= 2 传代码 =================
    section(2, "把项目代码传到服务器")
    para("方式 A —— 已 push 到 GitHub（推荐）", "h2")
    para("在服务器终端里依次执行：")
    code("cd ~/autodl-tmp\n"
         "git clone https://github.com/zwzwzwzwz123/DataCenter-HVAC-Copilot.git\n"
         "cd DataCenter-HVAC-Copilot")
    para("方式 B —— 没 push 成功", "h2")
    para("用 JupyterLab 左侧的「上传」按钮，把项目文件夹压缩成 zip 上传，再解压：")
    code("cd ~/autodl-tmp\n"
         "unzip DataCenter-HVAC-Copilot.zip\n"
         "cd DataCenter-HVAC-Copilot")
    note("训练数据 distill/data/ 下的 jsonl 文件必须一起传上去。用 git clone 的话它们已在仓库里（前提是已 push）。")

    # ================= 3 装依赖 =================
    section(3, "安装训练依赖")
    para("在项目根目录（能看到 pyproject.toml 的位置）执行这一条：")
    code("pip install -e '.[train]'")
    para("它会自动装好全部训练依赖：")
    bullets([
        "transformers（模型加载）、trl（SFT 训练器）、peft（LoRA）",
        "datasets、accelerate、bitsandbytes（4-bit 量化）",
        "torch 镜像里已带，会自动跳过。",
    ])
    note("如果上面因引号报错（个别终端不认单引号），改用双引号版本： pip install -e \".[train]\"")
    note("你不需要装 faiss 或 sentence-transformers——那是项目检索模块用的，与训练无关；相关测试报错可直接无视。")

    # ================= 4 加速 =================
    section(4, "设置模型下载加速")
    para("Qwen 模型权重约 3GB，从官方源下载可能很慢。执行下面两条走国内镜像：")
    code("export HF_ENDPOINT=https://hf-mirror.com\n"
         "export HF_HOME=~/autodl-tmp/hf_cache")
    note("这两条只对当前终端有效。若中途关闭终端重开，训练前需再执行一次。")

    # ================= 5 训练 =================
    section(5, "运行训练")
    para("在项目根目录执行（一整条，可直接复制）：")
    code("python -m distill.train_sft \\\n"
         "    --train distill/data/gold_sft_train.jsonl \\\n"
         "    --val   distill/data/gold_sft_val.jsonl \\\n"
         "    --output distill/checkpoints/sft-qwen1.5b")
    para("接下来会发生什么（都是正常现象，不要中断）", "h2")
    bullets([
        "先下载 Qwen2.5-1.5B 权重（约 3GB，几分钟）。",
        "然后开始训练，屏幕滚动显示 loss 数字，loss 应总体下降。",
        "训练 3 轮（epoch），4090 上约 10–25 分钟。",
        "结束时打印一行：<b>[eval] val legality=XX% exact_match=XX%</b>。",
    ])
    note("这行 val legality（合法率）就是验收指标：微调后小模型能输出多少比例「格式正确、可被系统解析」的计划。越高越好。")
    para("如果显存不够（报 out of memory）", "h2")
    para("把 batch 调小、序列调短再跑（1.5B 在 24G 卡上基本不会 OOM，此为保险）：")
    code("python -m distill.train_sft \\\n"
         "    --batch-size 2 --grad-accum 8 --max-seq-len 768 \\\n"
         "    --output distill/checkpoints/sft-qwen1.5b")

    # ================= 6 结果 =================
    section(6, "看结果 · 下载产物 · 关机")
    para("6.1　查看训练报告", "h2")
    code("cat distill/checkpoints/sft-qwen1.5b/sft_train_card.json")
    para("其中 <b>legal_rate</b>（合法率）和 <b>exact_match_rate</b>（与标准答案完全一致率）"
         "是核心数字。合法率明显高于未微调基座，阶段 2 即成功。")
    para("6.2　训练产物位置", "h2")
    bullets([
        "adapter_model.safetensors —— LoRA 权重（很小，几十 MB）",
        "sft_train_card.json —— 训练报告",
        "用 JupyterLab 左侧文件树，右键该文件夹「下载」回本地保存。",
    ])
    note("用完立刻在 AutoDL 控制台点「关机」，否则按小时继续扣费。")

    # ================= 7 FAQ =================
    section(7, "常见问题")
    faqs = [
        ("pip install 报错找不到某个包？",
         "先执行 python -m pip install --upgrade pip，再重试第 3 步。"),
        ("报 missing training dependency (xxx)？",
         "说明第 3 步没装成功。回项目根目录重新执行 pip install -e '.[train]'。"),
        ("训练报 TRL 参数错误（如 max_seq_length 不认）？",
         "不同 TRL 版本参数名略有差异。把报错整段发我，我给你对应版本改法——这是脚本唯一未在真机验证过的地方，已在交接文档注明。"),
        ("模型下载卡住不动？",
         "确认第 4 步 HF_ENDPOINT 已设置；仍慢就 Ctrl+C 中断后重跑训练命令，已下载部分会续传。"),
        ("找不到训练数据文件？",
         "确认第 2 步把 distill/data/ 也传上来了。执行 ls distill/data/ 应能看到 gold_sft_train.jsonl。"),
    ]
    for q, a in faqs:
        E.append(KeepTogether([
            Paragraph("Q　" + q, S["faq_q"]),
            Paragraph("A　" + a, S["faq_a"]),
        ]))
    E.append(Spacer(1, 8))
    E.append(HBar(content_w))
    E.append(Spacer(1, 6))
    para("完整技术背景见项目内 distill/HANDOFF.md。遇到任何报错，把整段红色文字复制给我即可。", "note")

    # ---------- page decoration ----------
    def later_pages(canvas, doc):
        canvas.saveState()
        # top hairline
        canvas.setStrokeColor(CARD_LINE)
        canvas.setLineWidth(0.8)
        canvas.line(20 * mm, A4[1] - 12 * mm, PAGE_W - 18 * mm, A4[1] - 12 * mm)
        canvas.setFont("MSYH", 8)
        canvas.setFillColor(SLATE)
        canvas.drawString(20 * mm, A4[1] - 11 * mm, "SFT 训练操作手册 · Planner 蒸馏")
        # footer page number
        canvas.setFont("Consolas", 8)
        canvas.drawRightString(PAGE_W - 18 * mm, 10 * mm, f"{doc.page - 1}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="SFT 训练操作手册", author="ZW",
    )
    doc.build(E, onFirstPage=cover, onLaterPages=later_pages)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    out = str(Path(__file__).parent / "SFT_训练操作手册_v2.pdf")
    build(out)
