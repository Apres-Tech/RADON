"""Compiles the formal publication-grade RADON Technical Writeup PDF."""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
METRICS_DIR = BASE_DIR / "metrics"
OUTPUT_PDF = DOCS_DIR / "RADON_Technical_Writeup.pdf"


def generate_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("RADON: Multi-Agent Reinforcement Learning for Autonomous Search, Rescue, and Hazard Navigation under Adversarial Dynamics", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("System Architecture, Spatial Generalization, and Iterated Self-Play Arms Race", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("RADON Research & Development Initiative &bull; November 2026", meta_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # Abstract
    story.append(Paragraph("<b>Abstract</b> &mdash; Urban structure fires represent high-stakes, partially observable multi-agent environments characterized by rapid hazard propagation, structural collapse risks, and temporal urgency. We present <b>RADON</b>, a multi-agent reinforcement learning (MARL) simulation and execution framework where a team of autonomous rescuer agents coordinates to evacuate trapped victims from procedurally generated, multi-floor burning buildings. Addressing critical reinforcement learning pitfalls in spatial navigation, we design a dynamic 3D Breadth-First Search (BFS) potential reward shaping field that routes agents around dynamic fire fronts across inter-floor stairwells. By enforcing a disjoint train/test seed protocol across 1.65 million continuous environment steps on modern GPUs, we demonstrate robust generalization on never-seen procedural buildings, raising full-clear rescue rates from 0.0% to 11.0% (peaking at 20.0%) and hostage evacuation by +245% over random baselines. Furthermore, we formulate dynamic hazard propagation as a zero-sum learned adversary via iterated self-play, demonstrating an arms race between tactical chokepoint fire attacks and adaptive defensive pathfinding. The full framework is paired with an interactive 3D WebGL dashboard and production FastAPI backend.", body_style))
    story.append(Spacer(1, 8))

    # 1. Environment & Architecture
    story.append(Paragraph("1. System Architecture & Multi-Floor Environment", h1_style))
    story.append(Paragraph("RADON implements the PettingZoo <code>ParallelEnv</code> API, simulating discrete spatial dynamics across multiple floors ($F \\times H \\times W$). Each agent receives a localized egocentric $7 \\times 7$ grid view across six semantic feature channels: Walls, Fires, Rescuers, Hostages, Stairs/Exits, and Unexplored Frontiers, concatenated with egocentric status vectors (carrying flag, active health, vertical floor index, and time ratio).", body_style))
    story.append(Paragraph("Procedural building layouts are synthesized through cellular room decomposition with guaranteed 3D multi-floor reachability: every hostage spawn, stairwell, and rescuer insertion point is verified via 3D BFS to possess unobstructed clearance to ground-floor exit portals.", body_style))

    # 2. Reward Engineering & 3D Spatial Potential Shaping
    story.append(Paragraph("2. Overcoming Exploration Bottlenecks via Dynamic 3D BFS Shaping", h1_style))
    story.append(Paragraph("Initial training iterations revealed two severe failure modes: the <i>interact exploration trap</i> (requiring simultaneous spatial alignment and a 1-in-6 discrete interaction action) and <i>corridor memorization</i>. We resolved these by collapsing pickup/delivery onto spatial contact and introducing obstacle-aware multi-floor BFS potential shaping:", body_style))
    story.append(Paragraph("&Phi;(s) = -&gamma; &middot; dist<sub>3D-BFS</sub>(pos, target)", bullet_style))
    story.append(Paragraph("Crucially, both static walls and dynamic fire tiles are categorized as impassable obstacles within the BFS graph. When fire spreads into a corridor, distance gradients immediately re-route agents around the hazard. Furthermore, vertical stairwell incentives (+0.5 upon decreasing goal distance) and inter-floor delivery milestones (+3.0) eliminated vertical oscillation loops.", body_style))

    # 3. Held-Out Generalization & 1.65M-Step Training
    story.append(Paragraph("3. Unseen Building Generalization & Long-Horizon Scaling", h1_style))
    story.append(Paragraph("To satisfy strict generalization bars (Milestone #10), training randomized map seeds continuously across a training band ($seeds \\in [1, 10000]$), while holding out an independent verification distribution ($seeds \\in [50000, 50099]$). A genuine long-horizon run was executed across 1,500,000 steps (1.65M cumulative) using 8 parallel vectorized environments (24 parallel agents) on an NVIDIA RTX 3050 Laptop GPU, achieving ~1,800 agent steps/second with zero memory leaks.", body_style))
    story.append(Paragraph("Periodic held-out evaluation demonstrated continuous learning: test success climbed from 6.7% at resume to a peak of <b>20.0%</b> at step 1,250,160, with hostages rescued reaching 0.80/2. The peak model was preserved at <code>agents/rescuer_best_heldout.zip</code>.", body_style))

    # Learning curve image
    curve_img_path = METRICS_DIR / "day5_learning_curve.png"
    if curve_img_path.exists():
        story.append(Spacer(1, 4))
        story.append(Image(str(curve_img_path), width=510, height=148))
        story.append(Paragraph("<b>Figure 1:</b> 1.65M-Step Multi-Floor Learning Curve across 3 panels: All-Evacuated Success Rate (Train vs Held-Out Test), Mean Hostages Rescued, and Rescuer Survival & Fire Avoidance.", meta_style))
        story.append(Spacer(1, 6))

    # 4. Self-Play Adversarial Fire
    story.append(Paragraph("4. Iterated Self-Play & Adversarial Arms Race", h1_style))
    story.append(Paragraph("In Day 4, hazard dynamics were augmented from passive stochastic spread into an active PPO-driven adversary (Milestones #7 & #9). The fire policy observes the global multi-floor state and selects from an 8-action tactical frontier ignition space targeting rescuer proximity, stairwells, and ground-floor exits.", body_style))
    story.append(Paragraph("In Round 1 (R1), the adversary successfully dropped rescuer success from 10.0% to 0.0% and quadrupled mortality from 0.43 to 1.85 by trapping rescuers in stair corridors. In Round 2 (R2), rescuers retrained against this aggressive adversary learned evasive pathfinding, recovering rescue performance and lifting survival rates back to 1.3/3.", body_style))

    # Arms race image
    arms_img_path = METRICS_DIR / "day4_selfplay_arms_race.png"
    if arms_img_path.exists():
        story.append(Spacer(1, 4))
        story.append(Image(str(arms_img_path), width=510, height=135))
        story.append(Paragraph("<b>Figure 2:</b> Self-Play Arms Race progression across iterated training rounds (R0 Baseline &rarr; R1 Trained Fire Chokepoint Attack &rarr; R2 Retrained Rescuer Evasion).", meta_style))
        story.append(Spacer(1, 6))

    # 5. Benchmark Table
    story.append(Paragraph("5. 100-Episode Held-Out Benchmark Results", h1_style))
    story.append(Paragraph("To validate true generalization, the final model was benchmarked across 100 independent, unseen procedural buildings (seeds 50000..50099) against a random policy baseline under active fire:", body_style))

    table_data = [
        [
            Paragraph("<b>Evaluation Metric</b>", table_header_style),
            Paragraph("<b>Random Baseline</b>", table_header_style),
            Paragraph("<b>Trained PPO Rescuer</b>", table_header_style),
            Paragraph("<b>Relative Advantage</b>", table_header_style),
        ],
        [
            Paragraph("All-Evacuated Success Rate", table_cell_style),
            Paragraph("0.0%", table_cell_style),
            Paragraph("<b>11.0%</b>", table_cell_style),
            Paragraph("+11.0% (11/100 cleared)", table_cell_style),
        ],
        [
            Paragraph("Mean Hostages Rescued (/2)", table_cell_style),
            Paragraph("0.20 / 2", table_cell_style),
            Paragraph("<b>0.69 / 2</b>", table_cell_style),
            Paragraph("+245% (~3.5x higher)", table_cell_style),
        ],
        [
            Paragraph("Rescuer Team Survival Rate", table_cell_style),
            Paragraph("1.66 / 3 (55.3%)", table_cell_style),
            Paragraph("<b>2.27 / 3 (75.7%)</b>", table_cell_style),
            Paragraph("+36.8% higher team survival", table_cell_style),
        ],
        [
            Paragraph("Mean Fire Fatalities", table_cell_style),
            Paragraph("1.34", table_cell_style),
            Paragraph("<b>0.73</b>", table_cell_style),
            Paragraph("Mortality reduced by 46%", table_cell_style),
        ],
        [
            Paragraph("Mean Episode Return", table_cell_style),
            Paragraph("+1.23 &plusmn; 10.1", table_cell_style),
            Paragraph("<b>+28.15 &plusmn; 40.8</b>", table_cell_style),
            Paragraph("+26.92 reward gain", table_cell_style),
        ],
    ]

    t = Table(table_data, colWidths=[150, 105, 125, 130])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # 6. Dashboard & Production Deployment
    story.append(Paragraph("6. Interactive 3D WebGL Dashboard & FastAPI Backend", h1_style))
    story.append(Paragraph("To deliver accessible real-time verification for non-technical operators, RADON integrates a decoupled WebGL/FastAPI presentation layer. The FastAPI backend loads trained PyTorch weights once at startup and serves asynchronous rollout endpoints (<code>POST /generate</code>) alongside structured replay catalogs (<code>GET /replays</code>).", body_style))
    story.append(Paragraph("The Three.js frontend renders multi-floor buildings with extruded walls, transparent floor slabs, dynamic fire lighting, and animated rescuer entities, supporting arbitrary unseen procedural generation with zero browser restarts. Complete operating instructions are documented in <code>RUNBOOK.md</code>.", body_style))

    # Build PDF
    doc.build(story)
    print(f"[PDF Generated] Technical writeup compiled to: {OUTPUT_PDF}")


if __name__ == "__main__":
    generate_pdf()

