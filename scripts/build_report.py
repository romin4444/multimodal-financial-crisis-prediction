#!/usr/bin/env python3
"""
Build a professional PDF report (outputs/FCPS_Report.pdf) summarising the project:
objectives, leakage-free methodology, results (tables + the 8 figures), the
capstone-vs-rebuild comparison, and the honest conclusion.

Pure-Python (reportlab); no LaTeX/pandoc needed.  Author: Romin Patel.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PDF = OUT / "FCPS_Report.pdf"

NAVY = colors.HexColor("#1F3A5F")
STEEL = colors.HexColor("#2C3E50")
ACCENT = colors.HexColor("#C0392B")
LIGHT = colors.HexColor("#ECF0F1")
USABLE_W = letter[0] - 1.6 * inch  # 0.8in margins

# ─── styles ──────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=15, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=STEEL, fontSize=12, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("Cap", parent=ss["BodyText"], fontSize=8, leading=10, textColor=colors.grey,
                     alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
CELL = ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=7.6, leading=9)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")


def P(t):
    return Paragraph(t, BODY)


def fig(name, caption, width=USABLE_W):
    path = OUT / name
    if not path.exists():
        return [P(f"<i>[figure {name} not found]</i>")]
    iw, ih = ImageReader(str(path)).getSize()
    w = min(width, USABLE_W)
    h = w * ih / iw
    if h > 4.2 * inch:  # cap height so it fits with caption
        h = 4.2 * inch
        w = h * iw / ih
    return [Image(str(path), width=w, height=h), Paragraph(caption, CAP)]


def table(data, col_w=None, header=True, font=8, body_font=7.6):
    rows = [[Paragraph(str(c), CELLB if (header and r == 0) else CELL) for c in row]
            for r, row in enumerate(data)]
    t = Table(rows, colWidths=col_w, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B8C0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def _page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.5)
    canvas.line(0.8 * inch, 0.65 * inch, letter[0] - 0.8 * inch, 0.65 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(0.8 * inch, 0.5 * inch, "FCPS — Financial Crisis Prediction System · Romin Patel")
    canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    story = []

    # ── Title page ──
    story += [Spacer(1, 1.4 * inch)]
    story.append(Paragraph("Financial Crisis Prediction System (FCPS)",
                           ParagraphStyle("T", parent=ss["Title"], textColor=NAVY, fontSize=26, leading=30)))
    story.append(Paragraph("A Leakage-Free Re-Evaluation of a Multimodal Crisis-Prediction Capstone",
                           ParagraphStyle("ST", parent=ss["Title"], textColor=STEEL, fontSize=13, leading=18)))
    story += [Spacer(1, 0.5 * inch)]
    meta = [["Author", "Romin Patel"],
            ["Course", "MBAI 5600G — Group 13"],
            ["Baseline paper", "Wang et al. (2025), PLOS ONE — HMM regime early-warning"],
            ["Data", "S&P 500 + VIX (1990–2024), FRED macro, FinBERT news"],
            ["Repository", "github.com/romin4444/multimodal-financial-crisis-prediction"],
            ["Date", date.today().isoformat()]]
    story.append(table([["Field", "Value"]] + meta, col_w=[1.5 * inch, USABLE_W - 1.5 * inch]))
    story += [Spacer(1, 0.4 * inch)]
    story.append(Paragraph(
        "<b>One-line summary.</b> Under a rigorous, leakage-free evaluation (combinatorial "
        "purged cross-validation + Probability of Backtest Overfitting + Deflated Sharpe), no "
        "model — including 2025-frontier methods and real FinBERT news sentiment — beats a "
        "one-line VIX threshold out-of-sample. The one robustly positive, deployable result is a "
        "daily Financial Stress Index that reconstructs the Federal Reserve's STLFSI at r ≈ 0.8.",
        BODY))
    story.append(PageBreak())

    # ── 1. Executive summary ──
    story.append(Paragraph("1. Executive Summary", H1))
    story.append(P(
        "This report documents the rebuild and honest re-evaluation of a multimodal financial "
        "crisis-prediction capstone. The original system reported a near-perfect in-sample F1 of "
        "0.99. We show that figure was an artefact of evaluation leakage (a circular target, "
        "future-peeking regime labels, and a holdout that pre-selected crisis dates). Rebuilt as a "
        "modular, tested, CI-backed package and evaluated under a leakage-free harness, the honest "
        "out-of-sample skill is modest and does not exceed a VIX baseline. We report this null "
        "transparently — it is the credible result, consistent with weak-form market efficiency and "
        "the 2024–25 literature on backtest overfitting."))
    story.append(Paragraph("Headline results", H2))
    head = [["Metric", "Result", "Status"],
            ["FSI vs Fed STLFSI (Pearson r)", "≈ 0.76 – 0.82", "PASS (target > 0.60)"],
            ["FSI as NBER-recession classifier (AUC)", "0.86", "PASS (> 0.80)"],
            ["Crisis PR-AUC, honest walk-forward (best ML)", "≈ 0.10 – 0.27", "Below VIX baseline"],
            ["VIX baseline PR-AUC (best target)", "0.35", "Benchmark — unbeaten"],
            ["Real-news sentiment marginal value (ΔPR-AUC)", "−0.018", "No value (hurts)"],
            ["Sentiment → stress Granger p-value", "0.14", "Not significant"],
            ["Discrete-time hazard concordance (C-index)", "≈ 0.84", "Genuine ranking skill"],
            ["Probability calibration (ECE: before → after)", "0.26 → 0.05", "Fixed"],
            ["Stock-direction edge over 'always-up'", "−0.05", "No edge (EMH)"],
            ["Test suite", "80 passing, CI green", "Reproducible"]]
    story.append(table(head, col_w=[3.0 * inch, 1.7 * inch, USABLE_W - 4.7 * inch]))
    story.append(PageBreak())

    # ── 2. Objectives ──
    story.append(Paragraph("2. Project Objectives (original capstone)", H1))
    story.append(P(
        "Building on Wang et al. (2025), which detects equity-market regime switches early using an "
        "HMM/GARCH network, the capstone added a <b>multimodal</b> dimension: combine market-math "
        "signals (ARMA-GARCH volatility, a Hidden Markov regime model, and a composite Financial "
        "Stress Index) with <b>FinBERT news-sentiment</b>, and test the central thesis that "
        "<b>news sentiment leads price</b> — providing earlier warning than price alone. The system "
        "was to be validated on three crises (GFC 2008, COVID 2020, Inflation 2022) against explicit "
        "targets: FSI–STLFSI correlation &gt; 0.60 and fusion F1 ≥ 0.70 on held-out crisis windows."))
    story.append(Paragraph("3. Data & Methods", H1))
    story.append(Paragraph("3.1 Data", H2))
    story.append(P(
        "S&amp;P 500 and VIX daily series (1990–2024, 8,816 trading days) via Yahoo Finance; "
        "VIX-orthogonal macro from FRED — credit spread (Moody's Baa–10Y, BAA10Y, full history), "
        "yield-curve slope (T10Y2Y/T10Y3M), short rate (DGS3MO), oil (DCOILWTICO); cross-asset "
        "correlation from AAPL/JPM/XOM/GS; and real financial-news headlines scored with FinBERT "
        "(ProsusAI/finbert)."))
    story.append(Paragraph("3.2 Models", H2))
    story.append(P(
        "ARMA-GARCH / GJR / EGARCH (BIC-selected); a 3-state Gaussian HMM regime detector; a "
        "4-component Financial Stress Index; FinBERT and VADER sentiment; topological "
        "persistent-homology features (TDA); a calibrated logistic/RF fusion classifier; and a "
        "discrete-time hazard (survival) model."))
    story.append(Paragraph("3.3 The leakage-free evaluation harness (the core contribution)", H2))
    story.append(P(
        "<b>Exogenous target</b> — forward N-day drawdown ≤ −X% (not a model's own label). "
        "<b>Causal regimes</b> — forward-only filtered HMM posteriors P(state_t | x_1..x_t), never "
        "the smoothed forward–backward posterior that peeks at the future. <b>Combinatorial Purged "
        "Cross-Validation</b> with a 21-day embargo (López de Prado). <b>Probability of Backtest "
        "Overfitting (PBO)</b> and the <b>Deflated Sharpe Ratio</b> to correct for multiple testing. "
        "<b>Honest baselines</b> (VIX threshold, persistence, base-rate) and probability "
        "<b>calibration</b> (isotonic). A configuration only 'wins' if it clears all three gates: "
        "PR-AUC &gt; VIX, PBO &lt; 0.5, and Deflated Sharpe &gt; 0.95."))
    story.append(PageBreak())

    # ── 4. Results with figures ──
    story.append(Paragraph("4. Results", H1))

    story.append(Paragraph("4.1 Financial Stress Index — the one result that survives", H2))
    story.append(P(
        "The daily FSI reproduces the St. Louis Fed's published STLFSI at r ≈ 0.76–0.82 and "
        "discriminates NBER recessions at AUC ≈ 0.86 — using only public daily inputs. This is a "
        "genuine, deployable signal and the project's strongest result."))
    story += fig("02_sentiment_vs_fsi.png",
                 "Figure 1. VIX, FinBERT fear index, and the composite Financial Stress Index across 1990–2024; shaded bands mark the validation crises.")

    story.append(Paragraph("4.2 Regime detection (HMM) and volatility (GARCH)", H2))
    story += fig("01_regime_timeline.png",
                 "Figure 2. S&P 500 with HMM regime labels (stable / volatile / crisis) and the FSI track beneath.")
    story += fig("05_hmm_selection.png",
                 "Figure 3. HMM model selection — BIC and log-likelihood across 2–4 states; n=3 retained for interpretability.")
    story += fig("06_garch_all.png",
                 "Figure 4. GARCH-family conditional volatility vs realised returns and VIX; EGARCH(1,1) selected by BIC.")

    story.append(Paragraph("4.3 Q1 — Does anything beat a VIX threshold? (Honest CPCV)", H2))
    story.append(P(
        "Across six targets (horizon × drawdown threshold), the VIX threshold beats every feature "
        "set; adding macro, regime, and TDA features each <i>lowers</i> PR-AUC. No configuration "
        "cleared even the first gate, so PBO/DSR were moot. Result confirmed in two independent "
        "environments with real full-history credit data."))
    q1 = [["Target (H / thr)", "Base", "VIX", "Price", "+Macro", "+Regime", "+TDA", "+Sent"],
          ["10d / 7%", "0.034", "0.180", "0.128", "0.084", "0.074", "0.067", "0.039"],
          ["10d / 10%", "0.015", "0.160", "0.132", "0.101", "0.044", "0.024", "0.014"],
          ["21d / 7%", "0.092", "0.229", "0.151", "0.102", "0.114", "0.112", "0.111"],
          ["21d / 10%", "0.039", "0.178", "0.103", "0.079", "0.072", "0.063", "0.044"],
          ["63d / 7%", "0.211", "0.346", "0.271", "0.230", "0.225", "0.217", "0.200"],
          ["63d / 10%", "0.136", "0.258", "0.176", "0.140", "0.139", "0.140", "0.104"]]
    cw = [1.25 * inch] + [(USABLE_W - 1.25 * inch) / 7] * 7
    story.append(table(q1, col_w=cw))
    story.append(Paragraph("Table 1. Out-of-sample PR-AUC by target and feature set (Kaggle GPU, real FRED credit + TDA). VIX column is the benchmark; no ML column exceeds it.", CAP))
    story += fig("07_fusion_eval.png",
                 "Figure 5. Fusion-model evaluation by crisis period (in-sample/event-holdout view from the original pipeline, retained for comparison).")

    story.append(Paragraph("4.4 Q2 — Does real news sentiment add value?", H2))
    story.append(P(
        "Using real FinBERT sentiment (3,142 headlines, 39% trading-day coverage), at the best base "
        "configuration the marginal value of sentiment was <b>ΔPR-AUC = −0.018</b> (it hurt), and "
        "sentiment did <b>not</b> Granger-cause forward stress (min p = 0.14). The multimodal "
        "thesis — the capstone's central novel claim — is not supported out-of-sample."))
    story += fig("03_lead_lag.png",
                 "Figure 6. Lead-lag cross-correlation (FSI vs fear index). The original 'sentiment leads price' reading was largely a VIX-with-itself artefact under synthetic sentiment.")
    story += fig("04_shap_by_crisis.png",
                 "Figure 7. SHAP feature attribution per crisis (Random Forest). Useful for interpretability, but it explains a model that does not beat VIX.")
    story += fig("08_research_comparison.png",
                 "Figure 8. Positioning vs prior literature (price signals, sentiment, explainability, lead-lag).")
    story.append(PageBreak())

    # ── 5. Capstone vs rebuild ──
    story.append(Paragraph("5. Original Capstone vs Rigorous Rebuild", H1))
    story.append(Paragraph("5.1 Claim-by-claim verdict", H2))
    claims = [["Capstone claim / target", "Original", "Leakage-free finding"],
              ["FSI vs STLFSI r > 0.60", "met (~0.82)", "CONFIRMED (0.76–0.82) — survives"],
              ["Fusion F1 ≥ 0.70 on crisis windows", "met (up to 0.99)", "Leakage; honest PR-AUC ≈ 0.10, loses to VIX"],
              ["News sentiment leads price (lead-lag)", "headline result", "Largely VIX↔VIX; real FinBERT adds no value"],
              ["Panic signal fires before HMM crisis", "claimed (COVID)", "HMM used smoothed (future) posteriors; not robust"],
              ["HMM detects regime switch in advance", "~37 td early", "Artefact of smoothing; causal version modest"],
              ["FinBERT > VADER", "claimed", "True but moot — neither beats price+VIX"],
              ["Multimodal > single-modality (thesis)", "implicit", "RETIRED: price+VIX alone beats the full model"]]
    story.append(table(claims, col_w=[2.5 * inch, 1.1 * inch, USABLE_W - 3.6 * inch]))
    story.append(Paragraph("Table 2. Of seven substantive claims, only the FSI correlation survives leakage-free testing.", CAP))

    story.append(Paragraph("5.2 Engineering & evaluation", H2))
    eng = [["Dimension", "Original (final.py)", "Rebuild"],
           ["Structure", "2,016-line monolith (Colab/Kaggle)", "Modular src/ package, FastAPI, MIT, public repo"],
           ["Evaluation", "In-sample / event-holdout (leaky)", "CPCV + embargo + PBO + Deflated Sharpe"],
           ["Baselines", "None", "VIX / persistence / base-rate"],
           ["Probabilities", "Uncalibrated, served as-is", "Calibrated (ECE 0.26→0.05) + honesty flag"],
           ["Tests / CI", "None", "80 tests, GitHub Actions green (3.10 & 3.12)"],
           ["Reproducibility", "Notebook + secrets", "pip install, lockfile, fixed seeds"]]
    story.append(table(eng, col_w=[1.2 * inch, 2.3 * inch, USABLE_W - 3.5 * inch]))
    story.append(Paragraph("Table 3. The rebuild adds the evaluation rigor and engineering the original lacked.", CAP))
    story.append(PageBreak())

    # ── 6. Conclusion ──
    story.append(Paragraph("6. Conclusion", H1))
    story.append(P(
        "The capstone was ambitious and well-structured, but its headline results were products of "
        "evaluation leakage — precisely the failure mode the modern backtest-overfitting literature "
        "exists to catch. Rebuilt and evaluated honestly, the central multimodal early-warning thesis "
        "does not hold: no method beats a VIX threshold out-of-sample, and real news sentiment adds no "
        "robust value. This is not a downgrade but a stronger, defensible contribution: a leakage-free "
        "benchmark that explains why prior numbers do not replicate, plus one genuinely deployable "
        "signal — a daily Financial Stress Index that reconstructs a Federal Reserve index at r ≈ 0.8."))
    story.append(Paragraph("Recommended framing", H2))
    story.append(P(
        "(1) Lead with the FSI ≈ STLFSI result (real, positive). (2) Present the leakage-free harness "
        "(CPCV / PBO / Deflated Sharpe) as the methodological contribution. (3) Report the honest "
        "null — multimodal sentiment and regimes do not beat VIX, consistent with weak-form EMH. "
        "(4) Quantify the leakage gap (in-sample F1 0.99 → honest PR-AUC ≈ 0.10). Future work that "
        "could change the answer requires changing the problem — longer horizons, cross-sectional "
        "ranking, or alternative data — not more model complexity."))
    story.append(Paragraph("7. Reproducibility", H1))
    story.append(P(
        "All code, tests, notebooks, and run logs are public and CI-verified. Key commands: "
        "<font face='Courier'>pip install -e .</font>; <font face='Courier'>make demo</font> "
        "(30-second run, no API key); <font face='Courier'>python scripts/v3_run.py</font> "
        "(honest crisis backtest); the Kaggle GPU notebooks reproduce the Q1/Q2 results. "
        "Repository: github.com/romin4444/multimodal-financial-crisis-prediction."))
    story.append(Paragraph("References", H2))
    refs = ("Wang et al. (2025), PLOS ONE · Hamilton (1989) · Ang &amp; Timmermann (2012) · "
            "Bollen et al. (2011) · Tetlock (2007) · Nelson (1991) · Ardia et al. (2020) · "
            "Bussmann et al. (2020) · Hatzius et al. (2010) · Bailey &amp; López de Prado (2014, "
            "Deflated Sharpe) · López de Prado (2018, Advances in Financial ML).")
    story.append(Paragraph(refs, ParagraphStyle("R", parent=BODY, fontSize=8, textColor=colors.grey)))

    doc = SimpleDocTemplate(str(PDF), pagesize=letter,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="FCPS Report — Romin Patel", author="Romin Patel")
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    print(f"Wrote {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
