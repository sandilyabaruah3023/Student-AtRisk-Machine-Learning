# visualize_predictions.py
from pathlib import Path
import sys
import pandas as pd  # CSV loading into DataFrame [web:82]
import matplotlib.pyplot as plt
import seaborn as sns  # Count plots for categorical data [web:81]

def main():
    # 1) Take CSV path as CLI arg or use your default path [web:70]
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\7th Sem\Student-AtRisk-Machine-Learning\OULAD\student_risk_predictions.csv")  # [web:70]
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # 2) Load predictions CSV (expects id_student, risk_label) [web:82]
    df = pd.read_csv(csv_path)  # [web:82]
    required = {"id_student", "risk_label"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns {required}, found: {df.columns.tolist()}")
        # Proceeding but visuals will expect 'risk_label' [web:82]

    # 3) Prepare counts and percentages [web:82]
    counts = df["risk_label"].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)

    # 4) Theme [web:81]
    sns.set_theme(style="whitegrid", font_scale=1.1)  # [web:81]

    # 5) Bar chart of risk distribution with annotations [web:81]
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["Not at Risk", "At Risk"]
    counts = counts.reindex(order).fillna(0).astype(int)
    pct = pct.reindex(order).fillna(0.0)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#4CAF50", "#F44336"])  # [web:81]
    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, c + max(counts.values) * 0.02, f"{c} ({p:.1f}%)", ha="center", va="bottom")  # [web:81]
    ax.set_xlabel("Risk label")
    ax.set_ylabel("Number of students")
    ax.set_title("Student Risk Prediction Distribution")
    fig.tight_layout()
    out_dir = csv_path.parent
    out_bar = out_dir / "risk_distribution_bar.png"
    fig.savefig(out_bar, dpi=150)
    print(f"Saved: {out_bar}")

    # 6) Pie chart of risk share [web:81]
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90,
        counterclock=False,
    )
    ax2.set_title("Student Risk Prediction Share")
    fig2.tight_layout()
    out_pie = out_dir / "risk_distribution_pie.png"
    fig2.savefig(out_pie, dpi=150)
    print(f"Saved: {out_pie}")

    # 7) Optional: probability histogram if present (risk_probability) [web:81]
    if "risk_probability" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.histplot(df["risk_probability"], bins=20, kde=True, ax=ax3, color="#2196F3")  # [web:81]
        ax3.set_xlabel("Predicted probability of being At Risk")
        ax3.set_ylabel("Count")
        ax3.set_title("Risk Probability Distribution")
        fig3.tight_layout()
        out_hist = out_dir / "risk_probability_hist.png"
        fig3.savefig(out_hist, dpi=150)
        print(f"Saved: {out_hist}")

if __name__ == "__main__":
    main()
