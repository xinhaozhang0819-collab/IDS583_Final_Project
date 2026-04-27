"use strict";

const fs = require("fs");
const path = require("path");

function requireRuntimePackage(name) {
  try {
    return require(name);
  } catch (_err) {
    return require(path.join(
      "/Users/minleihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules",
      name,
    ));
  }
}

const PptxGenJS = requireRuntimePackage("pptxgenjs");
const imageSize = requireRuntimePackage("image-size");

const ROOT = path.resolve(__dirname, "..");
const PROCESSED = path.join(ROOT, "data", "processed");
const FIGURES = path.join(ROOT, "reports", "figures", "loss_reserve");
const OUT_DIR = path.join(ROOT, "output", "slides");
const OUT_PATH = path.join(OUT_DIR, "credit_risk_ch4_6_academic_presentation.pptx");

const W = 13.333;
const H = 7.5;
let pptx;

const COLORS = {
  white: "FFFFFF",
  ink: "000000",
  muted: "3F3F46",
  light: "F2F2F2",
  line: "BFBFBF",
  blue: "1F4E79",
  blue2: "5F7892",
  orange: "8A4B1F",
  red: "8B3A3A",
  green: "3C6B4F",
  softBlue: "F3F6FA",
  softOrange: "F8F3EE",
  softGray: "F8F8F8",
};

const FONT_HEAD = "Times New Roman";
const FONT_BODY = "Times New Roman";

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"' && quoted && line[i + 1] === '"') {
      cur += '"';
      i += 1;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (ch === "," && !quoted) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function readCsv(filePath) {
  const text = fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, "");
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const fields = parseCsvLine(line);
    const row = {};
    headers.forEach((h, i) => {
      row[h] = fields[i] ?? "";
    });
    return row;
  });
}

function num(value) {
  if (value === null || value === undefined || value === "" || value === "NA") return NaN;
  return Number(value);
}

function fmtInt(value) {
  return Math.round(Number(value)).toLocaleString("en-US");
}

function fmtPct(value, digits = 2) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtMoneyShort(value) {
  const v = Number(value);
  if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(3)}B`;
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `$${(v / 1e3).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

function rowLookup(rows, scope, measure) {
  const row = rows.find((r) => r.analysis_scope === scope && r.pd_measure === measure);
  if (!row) throw new Error(`Missing portfolio row: ${scope} ${measure}`);
  return row;
}

function topSegments(rows, scope, measure, segmentType, n) {
  return rows
    .filter(
      (r) =>
        r.analysis_scope === scope &&
        r.pd_measure === measure &&
        r.segment_type === segmentType,
    )
    .sort((a, b) => num(b.portfolio_el_share) - num(a.portfolio_el_share))
    .slice(0, n);
}

function titleCaseSegment(value) {
  const raw = String(value ?? "");
  if (raw === "debt_consolidation") return "Debt consolidation";
  if (raw === "credit_card") return "Credit card";
  if (raw === "home_improvement") return "Home improvement";
  if (raw === "major_purchase") return "Major purchase";
  if (raw === "50-100k") return "$50k-$100k";
  if (raw === "100-150k") return "$100k-$150k";
  if (raw === "150k+") return "$150k+";
  return raw;
}

function loadMetrics() {
  const portfolio = readCsv(path.join(PROCESSED, "portfolio_expected_loss_summary.csv"));
  const segment = readCsv(path.join(PROCESSED, "segment_expected_loss_summary.csv"));
  const directSearch = readCsv(path.join(PROCESSED, "direct_active_12m_search_results.csv"));
  const stage2 = JSON.parse(
    fs.readFileSync(path.join(PROCESSED, "stage2_champion_config.json"), "utf8"),
  );
  const selectedDirect = directSearch
    .filter((r) => r.model_name === "XGBoost" && String(r.champion_eligible).toLowerCase() === "true")
    .sort((a, b) => num(b.validation_auc) - num(a.validation_auc))[0];
  const full12 = rowLookup(portfolio, "observable_12m_test", "12m");
  const resolved12 = rowLookup(portfolio, "resolved_test", "12m");
  const resolvedLife = rowLookup(portfolio, "resolved_test", "lifetime");
  const active12 = rowLookup(portfolio, "active_snapshot", "12m");
  const activeLife = rowLookup(portfolio, "active_snapshot", "lifetime");

  const activeGrade = topSegments(segment, "active_snapshot", "lifetime", "grade", 3);
  const resolvedGrade = topSegments(segment, "resolved_test", "lifetime", "grade", 3);
  const activeTerm = topSegments(segment, "active_snapshot", "lifetime", "term_months", 2);
  const resolvedTerm = topSegments(segment, "resolved_test", "lifetime", "term_months", 2);
  const activePurpose = topSegments(segment, "active_snapshot", "lifetime", "purpose_group", 1)[0];
  const resolvedPurpose = topSegments(segment, "resolved_test", "lifetime", "purpose_group", 1)[0];

  return {
    stage2,
    directSearch,
    selectedDirect,
    portfolio: { full12, resolved12, resolvedLife, active12, activeLife },
    segment: { activeGrade, resolvedGrade, activeTerm, resolvedTerm, activePurpose, resolvedPurpose },
    constants: {
      chargedRows: 268537,
      chargedMeanLgd: 0.9086,
      chargedMeanEad: 11170.0973,
      chargedMedianEad: 9500,
      chargedP90Ead: 22500,
      lgdMae: 0.0692,
      lgdRmse: 0.0892,
      lgdTrainingRows: 220529,
      resolvedRows: 225611,
      activeRows: 912609,
      observableRows: 443579,
    },
  };
}

function setBackground(slide) {
  slide.background = { color: COLORS.white };
}

function addTitle(slide, title, subtitle, slideNo) {
  setBackground(slide);
  slide.addText(title, {
    x: 0.55,
    y: 0.32,
    w: 10.8,
    h: 0.42,
    fontFace: FONT_HEAD,
    fontSize: 21,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.55,
      y: 0.78,
      w: 10.9,
      h: 0.28,
      fontFace: FONT_BODY,
      fontSize: 10,
      color: COLORS.muted,
      margin: 0,
      fit: "shrink",
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 1.12,
    w: 12.2,
    h: 0,
    line: { color: COLORS.line, pt: 0.6 },
  });
  slide.addText(String(slideNo).padStart(2, "0"), {
    x: 12.2,
    y: 0.34,
    w: 0.55,
    h: 0.22,
    fontFace: FONT_BODY,
    fontSize: 8.5,
    color: COLORS.muted,
    align: "right",
    margin: 0,
  });
}

function addFooter(slide, text = "Source: processed project outputs and course credit-loss framework.") {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 7.12,
    w: 12.2,
    h: 0,
    line: { color: "D9D9D9", pt: 0.5 },
  });
  slide.addText(text, {
    x: 0.55,
    y: 7.2,
    w: 11.8,
    h: 0.16,
    fontFace: FONT_BODY,
    fontSize: 7.2,
    color: "666666",
    margin: 0,
    fit: "shrink",
  });
}

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: opts.fontFace || FONT_BODY,
    fontSize: opts.fontSize || 12,
    color: opts.color || COLORS.ink,
    bold: opts.bold || false,
    italic: opts.italic || false,
    align: opts.align || "left",
    valign: opts.valign || "top",
    margin: opts.margin ?? 0.02,
    breakLine: false,
    fit: opts.fit || "shrink",
  });
}

function addBulletList(slide, items, x, y, w, h, fontSize = 10.2) {
  const rich = [];
  items.forEach((item) => {
    rich.push({
      text: item,
      options: {
        bullet: { type: "ul" },
        breakLine: true,
        hanging: 2,
      },
    });
  });
  slide.addText(rich, {
    x,
    y,
    w,
    h,
    fontFace: FONT_BODY,
    fontSize,
    color: COLORS.ink,
    fit: "shrink",
    margin: 0.02,
    paraSpaceAfterPt: 4,
    breakLine: false,
  });
}

function addMetric(slide, label, value, x, y, w, h = 0.64, color = COLORS.blue) {
  slide.addText(value, {
    x,
    y,
    w,
    h: h * 0.58,
    fontFace: FONT_HEAD,
    fontSize: 19,
    bold: true,
    color,
    margin: 0,
    fit: "shrink",
  });
  slide.addText(label, {
    x,
    y: y + h * 0.55,
    w,
    h: h * 0.36,
    fontFace: FONT_BODY,
    fontSize: 8.1,
    color: COLORS.muted,
    margin: 0,
    fit: "shrink",
  });
}

function addRuleLabel(slide, label, x, y, w, color = COLORS.blue) {
  slide.addShape(pptx.ShapeType.line, {
    x,
    y: y + 0.08,
    w: 0.42,
    h: 0,
    line: { color, pt: 1.4 },
  });
  addText(slide, label, x + 0.5, y, w - 0.5, 0.18, {
    fontSize: 8.4,
    bold: true,
    color,
  });
}

function addImageContain(slide, imagePath, x, y, w, h) {
  const dim = imageSize(imagePath);
  const ratio = dim.width / dim.height;
  const boxRatio = w / h;
  let iw;
  let ih;
  if (ratio > boxRatio) {
    iw = w;
    ih = w / ratio;
  } else {
    ih = h;
    iw = h * ratio;
  }
  const ix = x + (w - iw) / 2;
  const iy = y + (h - ih) / 2;
  slide.addImage({ path: imagePath, x: ix, y: iy, w: iw, h: ih });
}

function addFlowNode(slide, label, detail, x, y, w, h, fill = COLORS.softGray) {
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w,
    h,
    line: { color: COLORS.line, pt: 0.55 },
    fill: { color: fill },
  });
  addText(slide, label, x + 0.13, y + 0.12, w - 0.26, 0.22, {
    fontSize: 10,
    bold: true,
  });
  addText(slide, detail, x + 0.13, y + 0.39, w - 0.26, h - 0.48, {
    fontSize: 7.8,
    color: COLORS.muted,
  });
}

function addArrow(slide, x1, y1, x2, y2) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: { color: COLORS.muted, pt: 0.8, endArrowType: "triangle" },
  });
}

function addSmallTable(slide, rows, x, y, w, h, colW, fontSize = 8.6) {
  const tableRows = rows.map((row, ridx) =>
    row.map((cell) => ({
      text: String(cell),
      options: {
        fontFace: FONT_BODY,
        fontSize,
        bold: ridx === 0,
        color: COLORS.ink,
        fill: { color: ridx === 0 ? COLORS.light : COLORS.white },
        margin: 0.05,
        valign: "mid",
      },
    })),
  );
  slide.addTable(tableRows, {
    x,
    y,
    w,
    h,
    colW,
    border: { type: "solid", color: COLORS.line, pt: 0.35 },
    color: COLORS.ink,
    fontFace: FONT_BODY,
    fontSize,
    margin: 0.04,
  });
}

function addNotes(slide, text) {
  slide.addNotes(text.replace(/\s+/g, " ").trim());
}

function addFormula(slide, formula, x, y, w, h, accent = COLORS.blue) {
  slide.addShape(pptx.ShapeType.line, {
    x,
    y,
    w,
    h: 0,
    line: { color: accent, pt: 0.45 },
  });
  slide.addShape(pptx.ShapeType.line, {
    x,
    y: y + h,
    w,
    h: 0,
    line: { color: COLORS.line, pt: 0.35 },
  });
  addText(slide, formula, x + 0.15, y + 0.12, w - 0.3, h - 0.18, {
    fontFace: "Cambria Math",
    fontSize: 15,
    color: COLORS.ink,
    align: "center",
    valign: "mid",
  });
}

function buildDeck(metrics) {
  pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Minlei Hao";
  pptx.company = "IDS583 Final Project";
  pptx.subject = "Credit risk chapters 4-6 academic presentation";
  pptx.title = "Credit Loss Measurement, Expected Loss, and Risk Segmentation";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: FONT_HEAD,
    bodyFontFace: FONT_BODY,
    lang: "en-US",
  };

  const p = metrics.portfolio;
  const c = metrics.constants;
  const activeGrade = metrics.segment.activeGrade;
  const resolvedGrade = metrics.segment.resolvedGrade;
  const activeTerm = metrics.segment.activeTerm;
  const resolvedTerm = metrics.segment.resolvedTerm;
  const stage2 = metrics.stage2;
  const direct12 = stage2.direct_12m_model || {};
  const directCalibration = direct12.calibration || {};
  const selectedDirect = metrics.selectedDirect || {};
  const hazardDiagnostic = stage2.hazard_12m_comparison_model || stage2.hazard_12m_model || {};
  const staticLife = stage2.lifetime_model || {};

  // 1
  {
    const slide = pptx.addSlide();
    setBackground(slide);
    slide.addText("Credit Loss Measurement", {
      x: 0.65,
      y: 0.62,
      w: 9.1,
      h: 0.72,
      fontFace: FONT_HEAD,
      fontSize: 30,
      bold: true,
      color: COLORS.ink,
      margin: 0,
      fit: "shrink",
    });
    slide.addText("Dual-PD, LGD/EAD, Expected Loss, and Risk Segmentation", {
      x: 0.68,
      y: 1.36,
      w: 9.1,
      h: 0.3,
      fontFace: FONT_BODY,
      fontSize: 14,
      color: COLORS.muted,
      margin: 0,
      fit: "shrink",
    });
    slide.addShape(pptx.ShapeType.line, {
      x: 0.68,
      y: 1.88,
      w: 3.9,
      h: 0,
      line: { color: COLORS.line, pt: 0.8 },
    });
    addFormula(slide, "Expected Loss = PD x LGD x EAD", 0.75, 2.35, 5.4, 0.72, COLORS.blue);
    addText(
      slide,
      "Chapters 4-6 convert borrower-level risk estimates into dollar loss, reserve-style portfolio totals, and managerial segment views.",
      0.78,
      3.35,
      5.3,
      0.78,
      { fontSize: 13.2, color: COLORS.ink },
    );
    const blocks = [
      ["Probability of Default", "Direct 12-month PD and static HGB lifetime PD"],
      ["Loss Given Default", "Recovery-net-of-cost severity proxy"],
      ["Exposure at Default", "Remaining-principal exposure for defaults and active loans"],
      ["Segmentation", "Grade, term, FICO, purpose, income, and vintage views"],
    ];
    blocks.forEach((b, i) => {
      const x = 7.05;
      const y = 1.05 + i * 1.22;
      slide.addShape(pptx.ShapeType.line, {
        x: x - 0.16,
        y: y + 0.07,
        w: 0,
        h: 0.86,
        line: { color: i < 2 ? COLORS.blue : COLORS.orange, pt: 1.2 },
      });
      addText(slide, b[0], x, y, 4.8, 0.22, { fontSize: 12, bold: true });
      addText(slide, b[1], x, y + 0.3, 4.8, 0.35, { fontSize: 9.2, color: COLORS.muted });
    });
    addFooter(slide, "Source: project processed outputs; course framing: EL = PD x LGD x EAD.");
    addNotes(
      slide,
      "Open by stating that the presentation is not a generic default-classification exercise. The contribution is the chain from PD to LGD and EAD, then to loan-level and portfolio-level expected loss, and finally to segment-level risk management views.",
    );
  }

  // 2
  {
    const slide = pptx.addSlide();
    addTitle(slide, "Data Architecture And Temporal Samples", "The workflow separates severity estimation, resolved-loss evaluation, and active reserve measurement.", 2);
    const y = 1.65;
    const w = 2.18;
    const gap = 0.24;
    addFlowNode(slide, "Raw performance data", "Origination variables, loan status, payment and recovery fields", 0.65, y, w, 1.05, COLORS.softGray);
    addFlowNode(slide, "Stage2 PD predictions", "Dual-PD: static lifetime HGB plus direct 12M XGBoost", 0.65 + (w + gap), y, w, 1.05, COLORS.softBlue);
    addFlowNode(slide, "LGD/EAD proxies", "Charged-off severity and current exposure construction", 0.65 + 2 * (w + gap), y, w, 1.05, COLORS.softOrange);
    addFlowNode(slide, "Expected loss", "Loan-level and portfolio-level EL by horizon", 0.65 + 3 * (w + gap), y, w, 1.05, COLORS.softBlue);
    addFlowNode(slide, "Segmentation", "Grade, term, FICO, purpose, income, vintage", 0.65 + 4 * (w + gap), y, w, 1.05, COLORS.softGray);
    for (let i = 0; i < 4; i += 1) {
      addArrow(slide, 0.65 + (i + 1) * w + i * gap + 0.03, y + 0.52, 0.65 + (i + 1) * (w + gap) - 0.07, y + 0.52);
    }
    addText(slide, "Temporal discipline", 0.75, 3.18, 2.8, 0.22, { fontSize: 12.8, bold: true, color: COLORS.blue });
    addBulletList(
      slide,
      [
        "Training information is restricted to earlier vintages or earlier calendar snapshots.",
        "Validation is used for model selection and conservative calibration.",
        "Test and active outputs are treated as downstream evaluation and reserve views.",
      ],
      0.82,
      3.55,
      5.2,
      1.15,
      10.1,
    );
    addMetric(slide, "charged-off LGD/EAD proxy rows", fmtInt(c.chargedRows), 6.5, 3.22, 2.25, 0.78);
    addMetric(slide, "resolved holdout rows", fmtInt(c.resolvedRows), 9.15, 3.22, 2.15, 0.78, COLORS.orange);
    addMetric(slide, "active reserve snapshot rows", fmtInt(c.activeRows), 6.5, 4.28, 2.55, 0.78);
    addMetric(slide, "observable 12M test rows", fmtInt(c.observableRows), 9.15, 4.28, 2.55, 0.78, COLORS.orange);
    addSmallTable(
      slide,
      [
        ["Scope", "Use in slides"],
        ["Charged-off defaults", "LGD/EAD proxy estimation"],
        ["Resolved holdout", "Ex post loss comparison"],
        ["Active snapshot", "Reserve-style risk measurement"],
        ["Observable 12M cohort", "Matched short-horizon backtest"],
      ],
      0.82,
      5.25,
      11.65,
      1.28,
      [2.3, 9.35],
      8.6,
    );
    addFooter(slide);
    addNotes(
      slide,
      "This slide sets up why the project uses several samples. Charged-off loans identify severity, resolved loans allow ex post realized-loss comparison, and the active snapshot is the reserve portfolio. The observable 12-month cohort is especially important because it gives a cleaner short-horizon benchmark than resolved-only loans.",
    );
  }

  // 3
  {
    const slide = pptx.addSlide();
    addTitle(
      slide,
      "PD Input: Dual-Horizon Design",
      "Rule: 12-month PD is direct active-snapshot XGBoost; lifetime PD is static HGB-based.",
      3,
    );
    addText(slide, "12-month monitoring horizon", 0.78, 1.55, 4.2, 0.24, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addFlowNode(slide, "Active snapshot", "Each row is a loan still at risk at the monthly snapshot.", 0.78, 2.0, 2.75, 0.85, COLORS.softBlue);
    addFlowNode(slide, "Direct XGBoost", "The model predicts default within the next 12 months.", 4.0, 2.0, 2.75, 0.85, COLORS.softBlue);
    addFlowNode(slide, "Conservative calibration", "Loss-weighted calibration and grade-term floors buffer short-horizon PD.", 7.22, 2.0, 2.75, 0.85, COLORS.softBlue);
    addArrow(slide, 3.57, 2.42, 3.94, 2.42);
    addArrow(slide, 6.79, 2.42, 7.16, 2.42);
    addFormula(slide, "PD_12m = calibrated Pr(default within next 12 months | active snapshot)", 1.35, 3.15, 8.1, 0.55, COLORS.blue);
    addText(slide, "Lifetime reserve horizon", 0.78, 4.15, 4.2, 0.24, { fontSize: 13.5, bold: true, color: COLORS.orange });
    addFlowNode(slide, "Static HGB lifetime model", "Rerun original loan-level HistGradientBoosting model supplies lifetime PD.", 0.78, 4.62, 3.6, 0.88, COLORS.softOrange);
    addFlowNode(slide, "Lifetime EL", "Expected LGD and current EAD translate lifetime PD into reserve-style dollars.", 4.92, 4.62, 3.6, 0.88, COLORS.softOrange);
    addArrow(slide, 4.45, 5.06, 4.86, 5.06);
    addText(
      slide,
      "The one-month hazard model is retained as a research diagnostic, but it is not the main 12-month EL input in the final tables.",
      0.82,
      6.05,
      7.55,
      0.42,
      { fontSize: 10.2, color: COLORS.muted },
    );
    addSmallTable(
      slide,
      [
        ["PD measure", "Model", "Role"],
        ["12M PD", direct12.model_name || "XGBoost", "Short-horizon EL"],
        ["Lifetime PD", staticLife.model_name || "HistGradientBoosting", "Reserve-style EL"],
        ["Hazard diagnostic", hazardDiagnostic.model_name || "Calendar hazard", "Research benchmark"],
      ],
      8.8,
      3.55,
      3.75,
      1.72,
      [1.15, 1.25, 1.35],
      7.7,
    );
    addFooter(slide, "Source: stage2_champion_config.json and loss reserve report.");
    addNotes(
      slide,
      "The most important update is the dual-PD interpretation. Twelve-month PD is now a direct active-snapshot probability aligned to the next-12-month decision horizon. Lifetime PD is supplied by the rerun static HistGradientBoosting model. The one-month hazard model remains useful as a research diagnostic rather than the production 12-month EL input.",
    );
  }

  // 4
  {
    const slide = pptx.addSlide();
    addTitle(slide, "Direct Active-Snapshot Model For 12-Month PD", "The champion directly predicts next-12-month default for loans still active at a monthly snapshot.", 4);
    addImageContain(slide, path.join(FIGURES, "active_pd_compare.png"), 0.65, 1.45, 6.35, 3.6);
    addFormula(slide, "target_12m = 1 if charge-off/default occurs within (snapshot, snapshot + 12 months]", 0.72, 5.35, 6.95, 0.55, COLORS.blue);
    addText(slide, "Model facts", 7.45, 1.45, 2.6, 0.22, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addSmallTable(
      slide,
      [
        ["Item", "Current value"],
        ["12M engine", direct12.model_name || "XGBoost"],
        ["Feature profile", direct12.feature_profile || "full_calendar_hazard"],
        ["Feature count", fmtInt(direct12.feature_count || 167)],
        ["Validation AUC", num(selectedDirect.validation_auc).toFixed(3)],
        ["Full 12M coverage", fmtPct(num(p.full12.total_el) / num(p.full12.actual_loss_amount), 2)],
      ],
      7.45,
      1.9,
      4.75,
      1.95,
      [2.1, 2.65],
      8.2,
    );
    addText(slide, "Conservative calibration", 7.45, 4.35, 3.2, 0.22, { fontSize: 12.2, bold: true, color: COLORS.orange });
    addBulletList(
      slide,
      [
        `Method: ${directCalibration.method || "loss-weighted logit intercept plus grade-term floor"}.`,
        `Reference required PD: ${fmtPct(directCalibration.reference_required_pd || 0.0619659618, 2)}; target with buffer: ${fmtPct(directCalibration.reference_target_pd_with_buffer || 0.0681625579, 2)}.`,
        `Main output is capped at static lifetime PD, preserving PD_12m <= PD_life.`,
      ],
      7.48,
      4.75,
      4.85,
      1.15,
      9.2,
    );
    addFooter(slide, "Source: stage2_champion_config.json, direct_active_12m_search_results.csv, active_pd_compare.png.");
    addNotes(
      slide,
      "Explain that this slide replaces the earlier hazard-as-main approach. The model now predicts the intended twelve-month event directly from active-at-snapshot records. The hazard curve remains in the project as a diagnostic, but the expected-loss tables use the calibrated direct XGBoost output.",
    );
  }

  // 5
  {
    const slide = pptx.addSlide();
    addTitle(slide, "EAD Estimation For Installment Loans", "Exposure is proxied with remaining principal because exact balance-at-default is not observed in public data.", 5);
    addImageContain(slide, path.join(FIGURES, "ead_by_grade.png"), 0.58, 1.45, 6.35, 4.2);
    addText(slide, "Proxy design", 7.35, 1.48, 2.6, 0.24, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addFormula(slide, "EAD_proxy = max(funded amount - principal recovered, 0)", 7.35, 1.88, 4.9, 0.56, COLORS.blue);
    addBulletList(
      slide,
      [
        "For charged-off loans, principal already recovered is no longer economically exposed.",
        "For active loans, current EAD uses observed outstanding principal first.",
        "Scheduled-balance fallback uses funded amount, installment, interest rate, term, and months on book.",
      ],
      7.42,
      2.75,
      4.75,
      1.25,
      9.4,
    );
    addMetric(slide, "mean charged-off EAD proxy", fmtMoneyShort(c.chargedMeanEad), 7.42, 4.42, 1.9, 0.72);
    addMetric(slide, "median charged-off EAD proxy", fmtMoneyShort(c.chargedMedianEad), 9.75, 4.42, 1.9, 0.72, COLORS.orange);
    addMetric(slide, "90th percentile charged-off EAD", fmtMoneyShort(c.chargedP90Ead), 7.42, 5.38, 2.25, 0.72);
    addText(
      slide,
      "Empirical pattern: weaker grades tend to charge off with larger remaining balances, so exposure amplifies PD differences.",
      7.42,
      6.28,
      4.8,
      0.34,
      { fontSize: 9.6, color: COLORS.muted },
    );
    addFooter(slide, "Source: charged_off_loss_proxy.csv and ead_by_grade.png.");
    addNotes(
      slide,
      "The key point is observability. We do not observe a bank-grade balance-at-default ledger, so the project uses a transparent remaining-principal proxy. The boxplot shows that EAD is not flat across grades; weaker grades carry larger exposure at charge-off, which matters for dollar expected loss.",
    );
  }

  // 6
  {
    const slide = pptx.addSlide();
    addTitle(slide, "LGD Estimation And Credibility-Weighted Lookup", "Severity is measured from recoveries net of collection costs and stabilized through segment-level shrinkage.", 6);
    addImageContain(slide, path.join(FIGURES, "lgd_actual_vs_expected.png"), 0.72, 1.42, 4.75, 4.65);
    addText(slide, "Loss severity proxy", 6.0, 1.42, 3.2, 0.24, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addFormula(slide, "LGD_proxy = 1 - net recoveries / EAD_proxy", 6.0, 1.8, 5.3, 0.55, COLORS.blue);
    addBulletList(
      slide,
      [
        "Net recoveries subtract collection recovery fees from reported recoveries.",
        "Expected LGD uses grade x term x origination-year cells, shrunk toward the portfolio mean.",
        "Resolved future vintages use grade-term fallback to preserve temporal purity.",
      ],
      6.07,
      2.62,
      5.55,
      1.25,
      9.5,
    );
    addSmallTable(
      slide,
      [
        ["Diagnostic", "Value"],
        ["Mean realized LGD", fmtPct(c.chargedMeanLgd, 2)],
        ["Training charged-off rows", fmtInt(c.lgdTrainingRows)],
        ["Holdout LGD MAE", c.lgdMae.toFixed(4)],
        ["Holdout LGD RMSE", c.lgdRmse.toFixed(4)],
        ["Resolved lookup source", "grade-term fallback"],
      ],
      6.08,
      4.35,
      5.2,
      1.65,
      [2.7, 2.5],
      8.4,
    );
    addText(
      slide,
      "Interpretation: public-data severity is concentrated near high LGD, so error metrics are more informative than rank correlation.",
      6.08,
      6.27,
      5.35,
      0.34,
      { fontSize: 9.4, color: COLORS.muted },
    );
    addFooter(slide, "Source: charged_off_loss_proxy.csv, test_with_loss_metrics.csv, lgd_actual_vs_expected.png.");
    addNotes(
      slide,
      "LGD is high in this unsecured personal-loan portfolio. The credibility lookup is intentionally managerial rather than black-box: it uses segment averages but shrinks sparse cells toward the portfolio mean. The temporal fallback matters because exact future-vintage cells cannot be estimated without leakage.",
    );
  }

  // 7
  {
    const slide = pptx.addSlide();
    addTitle(slide, "Expected Loss At Loan And Portfolio Level", "The final EL tables combine horizon-specific PD, expected LGD, and exposure in dollars.", 7);
    addFormula(slide, "EL_12m = PD_12m,direct-calibrated x Expected LGD x EAD", 0.75, 1.45, 5.5, 0.52, COLORS.blue);
    addFormula(slide, "EL_life = PD_life,static-HGB x Expected LGD x EAD_current", 0.75, 2.1, 5.5, 0.52, COLORS.orange);
    const labels = ["Full 12M", "Resolved 12M", "Resolved Life", "Active 12M", "Active Life"];
    const values = [
      num(p.full12.total_el) / 1e6,
      num(p.resolved12.total_el) / 1e6,
      num(p.resolvedLife.total_el) / 1e6,
      num(p.active12.total_el) / 1e6,
      num(p.activeLife.total_el) / 1e6,
    ];
    slide.addChart(pptx.ChartType.bar, [{ name: "Expected EL ($M)", labels, values }], {
      x: 6.75,
      y: 1.38,
      w: 5.85,
      h: 4.05,
      showLegend: false,
      showTitle: false,
      chartColors: [COLORS.blue],
      valAxisLabelFontFace: FONT_BODY,
      valAxisLabelFontSize: 8,
      catAxisLabelFontFace: FONT_BODY,
      catAxisLabelFontSize: 8,
      valAxisMajorUnit: 500,
      valAxisMinVal: 0,
      valAxisNumFmt: "$#,##0",
      showValue: true,
      dataLabelFormatCode: "$#,##0",
      dataLabelPosition: "outEnd",
      valGridLine: { color: "E5E7EB", pt: 0.4 },
    });
    addSmallTable(
      slide,
      [
        ["Scope", "Avg PD", "Avg LGD", "Total EL"],
        ["Full 12M observable", fmtPct(p.full12.avg_pd, 2), fmtPct(p.full12.avg_lgd, 2), fmtMoneyShort(p.full12.total_el)],
        ["Resolved 12M", fmtPct(p.resolved12.avg_pd, 2), fmtPct(p.resolved12.avg_lgd, 2), fmtMoneyShort(p.resolved12.total_el)],
        ["Resolved lifetime", fmtPct(p.resolvedLife.avg_pd, 2), fmtPct(p.resolvedLife.avg_lgd, 2), fmtMoneyShort(p.resolvedLife.total_el)],
        ["Active 12M", fmtPct(p.active12.avg_pd, 2), fmtPct(p.active12.avg_lgd, 2), fmtMoneyShort(p.active12.total_el)],
        ["Active lifetime", fmtPct(p.activeLife.avg_pd, 2), fmtPct(p.activeLife.avg_lgd, 2), fmtMoneyShort(p.activeLife.total_el)],
      ],
      0.75,
      3.15,
      5.7,
      2.25,
      [2.25, 1.0, 1.0, 1.45],
      7.6,
    );
    addText(
      slide,
      "The lifetime reserve view is larger because static-HGB lifetime PD is materially above the direct 12M PD.",
      0.82,
      5.78,
      5.5,
      0.38,
      { fontSize: 9.3, color: COLORS.muted },
    );
    addFooter(slide, "Source: portfolio_expected_loss_summary.csv.");
    addNotes(
      slide,
      "This slide is the main accounting bridge. The same identity is used in both horizons, but the PD input changes with the horizon. The active lifetime reserve is much larger than active 12-month EL because the lifetime PD is about 26 percent versus about 8 percent for the direct 12-month horizon.",
    );
  }

  // 8
  {
    const slide = pptx.addSlide();
    addTitle(slide, "Backtesting: Predicted Versus Realized Loss", "Matched-horizon comparisons evaluate the full PD-LGD-EAD translation into dollars.", 8);
    addImageContain(slide, path.join(FIGURES, "predicted_vs_actual_loss.png"), 0.7, 1.35, 7.05, 4.2);
    addText(slide, "Key comparisons", 8.15, 1.42, 2.8, 0.22, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addSmallTable(
      slide,
      [
        ["Horizon", "Expected EL", "Actual net loss"],
        ["Full 12M observable", fmtMoneyShort(p.full12.total_el), fmtMoneyShort(p.full12.actual_loss_amount)],
        ["Resolved 12M diagnostic", fmtMoneyShort(p.resolved12.total_el), fmtMoneyShort(p.resolved12.actual_loss_amount)],
        ["Resolved lifetime", fmtMoneyShort(p.resolvedLife.total_el), fmtMoneyShort(p.resolvedLife.actual_loss_amount)],
      ],
      8.15,
      1.85,
      4.2,
      1.35,
      [1.75, 1.2, 1.25],
      7.8,
    );
    const coverage12 = num(p.full12.total_el) / num(p.full12.actual_loss_amount);
    const coverageLife = num(p.resolvedLife.total_el) / num(p.resolvedLife.actual_loss_amount);
    addMetric(slide, "full observable 12M coverage ratio", fmtPct(coverage12, 1), 8.18, 3.55, 2.15, 0.76);
    addMetric(slide, "resolved lifetime coverage ratio", fmtPct(coverageLife, 1), 10.42, 3.55, 2.05, 0.76, COLORS.orange);
    addBulletList(
      slide,
      [
        "Full observable 12M cohort is the cleaner short-horizon benchmark.",
        "Resolved-only 12M is a diagnostic because it excludes surviving active accounts.",
        "Lifetime resolved comparison is reserve-style and uses static HGB lifetime PD.",
      ],
      8.2,
      4.62,
      4.15,
      1.22,
      9.1,
    );
    addFooter(slide, "Source: predicted_vs_actual_loss.png and portfolio_expected_loss_summary.csv.");
    addNotes(
      slide,
      "The full observable 12-month cohort is the preferred short-horizon backtest because it includes loans whose twelve-month outcomes can be observed. The model is conservative on that view. The resolved-only diagnostic is less comparable because conditioning on resolution changes the risk composition.",
    );
  }

  // 9
  {
    const slide = pptx.addSlide();
    addTitle(slide, "Risk Segmentation: Expected Loss Concentration", "Segments identify where portfolio expected loss is concentrated after PD, LGD, and EAD are combined.", 9);
    addImageContain(slide, path.join(FIGURES, "segment_el_by_grade.png"), 0.62, 1.35, 6.8, 3.85);
    addText(slide, "Lifetime EL share by grade", 7.75, 1.36, 3.25, 0.24, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addSmallTable(
      slide,
      [
        ["Rank", "Active snapshot", "Resolved holdout"],
        [
          "1",
          `${activeGrade[0].segment_value}: ${fmtPct(activeGrade[0].portfolio_el_share, 2)}`,
          `${resolvedGrade[0].segment_value}: ${fmtPct(resolvedGrade[0].portfolio_el_share, 2)}`,
        ],
        [
          "2",
          `${activeGrade[1].segment_value}: ${fmtPct(activeGrade[1].portfolio_el_share, 2)}`,
          `${resolvedGrade[1].segment_value}: ${fmtPct(resolvedGrade[1].portfolio_el_share, 2)}`,
        ],
        [
          "3",
          `${activeGrade[2].segment_value}: ${fmtPct(activeGrade[2].portfolio_el_share, 2)}`,
          `${resolvedGrade[2].segment_value}: ${fmtPct(resolvedGrade[2].portfolio_el_share, 2)}`,
        ],
      ],
      7.78,
      1.82,
      4.5,
      1.35,
      [0.55, 1.9, 2.05],
      8.0,
    );
    addBulletList(
      slide,
      [
        "Grade C is the dominant dollar-loss bucket in both active and resolved views.",
        "Grades B-D jointly drive most lifetime EL due to volume and risk intensity.",
        "Weak grades have high loan-level risk but smaller aggregate shares because counts are lower.",
      ],
      7.82,
      3.62,
      4.25,
      1.15,
      9.1,
    );
    addSmallTable(
      slide,
      [
        ["Axis", "Dominant active lifetime segment", "Share"],
        ["Purpose", titleCaseSegment(metrics.segment.activePurpose.segment_value), fmtPct(metrics.segment.activePurpose.portfolio_el_share, 2)],
        ["Term", `${activeTerm[0].segment_value} months`, fmtPct(activeTerm[0].portfolio_el_share, 2)],
        ["FICO", "Fair", "58.94%"],
        ["Income", "$50k-$100k", "50.40%"],
      ],
      0.85,
      5.55,
      11.2,
      1.05,
      [1.6, 7.1, 2.5],
      7.7,
    );
    addFooter(slide, "Source: segment_expected_loss_summary.csv and segment_el_by_grade.png.");
    addNotes(
      slide,
      "The segmentation slide translates model output into management language. The important point is not just that lower grades are riskier, but that grade C has the largest aggregate expected-loss contribution because it combines material risk with large portfolio volume.",
    );
  }

  // 10
  {
    const slide = pptx.addSlide();
    addTitle(slide, "Term-Grade Interaction And Governance Implications", "Risk concentration is strongest when underwriting quality and contractual maturity are viewed jointly.", 10);
    addImageContain(slide, path.join(FIGURES, "grade_term_heatmap.png"), 0.7, 1.35, 6.15, 4.55);
    addText(slide, "Managerial interpretation", 7.25, 1.36, 3.2, 0.24, { fontSize: 13.5, bold: true, color: COLORS.blue });
    addMetric(slide, "active lifetime EL share in 60-month loans", fmtPct(activeTerm[0].portfolio_el_share, 2), 7.28, 1.82, 2.6, 0.8, COLORS.orange);
    addMetric(slide, "resolved lifetime EL share in 60-month loans", fmtPct(resolvedTerm[0].portfolio_el_share, 2), 10.0, 1.82, 2.55, 0.8, COLORS.blue);
    addBulletList(
      slide,
      [
        "Within each grade, 60-month loans have materially higher mean lifetime EL than 36-month loans.",
        "Within each term, mean lifetime EL increases from stronger grades toward weaker grades.",
        "Monitoring grade and term separately would miss the interaction that drives reserve exposure.",
      ],
      7.32,
      2.9,
      4.8,
      1.35,
      9.4,
    );
    addText(slide, "Governance use", 7.28, 4.72, 2.4, 0.22, { fontSize: 12.4, bold: true, color: COLORS.orange });
    addBulletList(
      slide,
      [
        "Reserve committee: focus on high-EL grade-term cells.",
        "Monitoring: track PD, LGD, EAD, and EL by standard business cuts.",
        "Next phase: feed EL into risk-based pricing and economic capital modules.",
      ],
      7.32,
      5.1,
      4.8,
      1.1,
      9.2,
    );
    addFooter(slide, "Source: grade_term_heatmap.png and segment_expected_loss_summary.csv.");
    addNotes(
      slide,
      "Close by emphasizing governance. The heatmap shows why grade and term need to be monitored jointly. This does not yet implement risk-based pricing or economic capital, but it supplies the expected-loss baseline those modules will need.",
    );
  }

  return pptx;
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const metrics = loadMetrics();
  const pptx = buildDeck(metrics);
  await pptx.writeFile({ fileName: OUT_PATH });
  console.log(OUT_PATH);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
