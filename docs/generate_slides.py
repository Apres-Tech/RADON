"""Generates a clean presentation slide deck PDF for RADON."""

from pathlib import Path
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_PDF = DOCS_DIR / "RADON_Presentation.pdf"


def build_slides():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(letter),
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    slide_title_style = ParagraphStyle(
        "SlideTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )

    slide_subtitle_style = ParagraphStyle(
        "SlideSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=15,
    )

    slide_body_style = ParagraphStyle(
        "SlideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )

    slide_bullet_style = ParagraphStyle(
        "SlideBullet",
        parent=slide_body_style,
        leftIndent=20,
        firstLineIndent=-12,
        spaceAfter=6,
    )

    footer_style = ParagraphStyle(
        "SlideFooter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#94a3b8"),
        alignment=TA_CENTER,
    )

    story = []

    def add_slide_header(title, subtitle):
        story.append(Paragraph(title, slide_title_style))
        story.append(Paragraph(subtitle, slide_subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=16))

    def add_slide_footer(slide_num):
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=6))
        story.append(Paragraph(f"RADON Project Presentation &bull; Slide {slide_num} of 6 &bull; November 2026", footer_style))
        story.append(PageBreak())

    # --- Slide 1: Title Slide ---
    story.append(Spacer(1, 40))
    story.append(Paragraph("RADON", ParagraphStyle("BigTitle", fontName="Helvetica-Bold", fontSize=44, leading=48, alignment=TA_CENTER, textColor=colors.HexColor("#1e3a8a"))))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Autonomous Multi-Agent Rescue Simulation Under Adversarial Dynamics", ParagraphStyle("Sub", fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#2563eb"))))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Multi-Floor Generalization &bull; Dynamic 3D BFS Shaping &bull; Self-Play Arms Race", ParagraphStyle("Sub2", fontName="Helvetica", fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#64748b"))))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Repository: https://github.com/Apres-Tech/RADON", ParagraphStyle("Link", fontName="Helvetica-Bold", fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#0f172a"))))
    add_slide_footer(1)

    # --- Slide 2: The Core Challenge & Architecture ---
    add_slide_header("1. Problem Formulation & Multi-Floor Simulation", "PettingZoo ParallelEnv with Procedural 3D Reachability")
    story.append(Paragraph("&bull; <b>Mission:</b> A squad of autonomous rescuers must locate, pick up, and evacuate trapped hostages to ground-floor exits within a burning multi-story structure.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Multi-Floor Discrete Dynamics:</b> Discrete grid ($F \\times H \\times W$) modeling rooms, corridors, stairwells, and expanding fire fronts.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Guaranteed Procedural Reachability:</b> Custom 3D BFS ensures every hostage, stairwell, and rescuer spawn maintains unobstructed exit clearance.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Egocentric Observation Space:</b> $7 \\times 7 \\times 6$ feature tensor (Walls, Fires, Rescuers, Hostages, Stairs/Exits, Frontiers) + status scalars.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Decoupled Architecture:</b> PettingZoo ParallelEnv &rarr; SuperSuit parameter-sharing &rarr; Stable-Baselines3 PPO &rarr; WebGL 3D Dashboard.", slide_bullet_style))
    add_slide_footer(2)

    # --- Slide 3: Reward Engineering & BFS Potential Shaping ---
    add_slide_header("2. Overcoming Exploration Bottlenecks", "Dynamic 3D BFS Potential Fields Treating Fire as Dynamic Obstacles")
    story.append(Paragraph("&bull; <b>The Interact Trap:</b> Requiring simultaneous spatial positioning and a 1-in-6 discrete action created an exponential exploration bottleneck. Solved via automatic pickup and delivery upon spatial contact.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Dynamic 3D BFS Potential Shaping:</b> $\\Phi(s) = -\\gamma \\cdot dist_{3D-BFS}(pos, target)$. Replaced Manhattan heuristics with true obstacle-aware shortest paths.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Fire Avoidance Rerouting:</b> Both static walls and active fire tiles are treated as impassable within BFS graphs; when fire engulfs a corridor, agents immediately re-route.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Inter-Floor Transition Incentives:</b> Goal-directed stair traversal bonus (+0.5) and floor-transit milestones (+3.0) eliminated vertical oscillation loops.", slide_bullet_style))
    add_slide_footer(3)

    # --- Slide 4: Generalization & 1.65M Long Horizon ---
    add_slide_header("3. Unseen Building Generalization", "1.65 Million Continuous Steps Across 8 Parallel GPU Environments")
    story.append(Paragraph("&bull; <b>Diagnosis of Overfitting:</b> Initial runs memorized fixed seeds. Enforced strict separation: training band ($seeds \\in [1, 10000]$) vs fixed held-out testing distribution ($seeds \\in [50000, 50099]$).", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Long-Horizon Scaling:</b> Trained for 1.5M steps (1.65M cumulative) using 8 parallel vectorized environments (24 parallel agents) on RTX 3050 GPU at ~1,800 steps/sec.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Generalization Milestones:</b> Held-out test success climbed from 0% (Day 3 initial) to 13.3% and peaked at <b>20.0%</b> at step 1.25M with 0.80/2 hostages evacuated.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Generalization Gap Eliminated:</b> Negative generalization gap observed at peak, proving true spatial heuristic acquisition rather than seed memorization.", slide_bullet_style))
    add_slide_footer(4)

    # --- Slide 5: Adversarial Self-Play Arms Race ---
    add_slide_header("4. Iterated Self-Play & Adversary Fire", "Active PPO Adversary Targeting Corridors and Stairwells")
    story.append(Paragraph("&bull; <b>Adversarial Hazard Formulation:</b> Fire upgraded from stochastic spread into a learned PPO agent with 8-action tactical frontier ignition.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Zero-Sum Objective:</b> Fire agent maximizes rescuer entrapment (+0.05/step) and fatalities (+6.0), penalizing successful deliveries (-12.0).", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Round 1 Attack (R1):</b> Trained fire actively blocked stairwell corridors, dropping rescuer success from 10% to 0% and quadrupling mortality (0.43 &rarr; 1.85 deaths).", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Round 2 Retrained Defense (R2):</b> Rescuers retrained against the aggressive adversary learned defensive routing, recovering evacuation progress and lifting survival back to 1.3/3.", slide_bullet_style))
    add_slide_footer(5)

    # --- Slide 6: 100-Episode Benchmark & Production Dashboard ---
    add_slide_header("5. Independent 100-Episode Benchmark & 3D Dashboard", "Empirical Proof on Never-Seen Buildings & Full WebGL Telemetry")
    story.append(Paragraph("&bull; <b>All-Evacuated Success Rate:</b> <b>11.0%</b> (Trained) vs <b>0.0%</b> (Random Baseline) &mdash; 11/100 maps completely cleared.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Hostages Rescued:</b> <b>0.69 / 2</b> (Trained) vs <b>0.20 / 2</b> (Random) &mdash; <b>+245% (~3.5x improvement)</b>.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Rescuer Survival:</b> <b>75.7% (2.27 / 3)</b> vs <b>55.3% (1.66 / 3)</b> &mdash; fire mortality reduced by 46%.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Production 3D Dashboard:</b> Three.js multi-floor viewer with live held-out map generator (<code>POST /generate</code>) and floor isolation.", slide_bullet_style))
    story.append(Paragraph("&bull; <b>Runbook & Reproducibility:</b> Complete copy-paste operator guide documented in <code>RUNBOOK.md</code>.", slide_bullet_style))
    add_slide_footer(6)

    doc.build(story)
    print(f"[Slides Generated] Presentation compiled to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_slides()

