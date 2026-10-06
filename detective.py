"""Dopamine Detective: reward prediction error (TD learning) in the style of Schultz, Dayan & Montague.

A cue appears, a reward follows 10 steps later. Watch the "surprise" signal slide
from the reward back to the cue, then dip when an expected reward is skipped.

Run:  python detective.py   ->  assets/dopamine_detective.png
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, PANEL, INK, MUTE = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
PINK, LAV, MINT, PEACH, SKY = "#ff7eb6", "#b794f6", "#7ee8c7", "#ffb86b", "#79c0ff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE,
    "ytick.color": MUTE, "axes.edgecolor": "#30363d", "font.size": 10,
})

T, CUE, REWARD = 20, 4, 14
ALPHA, GAMMA = 0.3, 1.0
N_TRIALS = 250


def value(w, tau):
    """Predicted future reward at time tau (needs a cue to have happened)."""
    return w[tau - CUE] if CUE <= tau < T else 0.0


def run_trial(w, rewarded=True):
    delta = np.zeros(T)
    for tau in range(1, T):
        r = 1.0 if (rewarded and tau == REWARD) else 0.0
        delta[tau] = r + GAMMA * value(w, tau) - value(w, tau - 1)
        if tau - 1 >= CUE:
            w[tau - 1 - CUE] += ALPHA * delta[tau]
    return delta


def main():
    os.makedirs("assets", exist_ok=True)
    w = np.zeros(T)
    deltas = np.array([run_trial(w) for _ in range(N_TRIALS)])
    omission = run_trial(w, rewarded=False)

    print(f"cue response after learning:    {deltas[-1, CUE]:+.2f}")
    print(f"reward response after learning: {deltas[-1, REWARD]:+.2f}")
    print(f"omission dip:                   {omission[REWARD]:+.2f}")

    cmap = LinearSegmentedColormap.from_list("rpe", [SKY, PANEL, PINK])
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.4, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(deltas, aspect="auto", cmap=cmap, vmin=-1, vmax=1, origin="upper", interpolation="nearest")
    ax.axvline(CUE, color=LAV, lw=1.2, ls="--")
    ax.axvline(REWARD, color=MINT, lw=1.2, ls="--")
    ax.text(CUE + 0.2, -6, "cue", color=LAV, fontweight="bold")
    ax.text(REWARD + 0.2, -6, "reward", color=MINT, fontweight="bold")
    ax.set_xlabel("time within trial")
    ax.set_ylabel("trial #")
    ax.set_title("Prediction error across learning: the pink blob walks from reward to cue", loc="left")
    fig.colorbar(im, ax=ax, label="δ (surprise)", pad=0.01)

    panels = [
        (deltas[0], "Trial 1: reward is a surprise", PINK),
        (deltas[-1], f"Trial {N_TRIALS}: the cue is the surprise", LAV),
        (omission, "Reward skipped: negative dip", SKY),
    ]
    for col, (d, title, color) in enumerate(panels):
        a = fig.add_subplot(gs[1, col])
        a.bar(np.arange(T), d, color=color, width=0.7)
        a.axhline(0, color=MUTE, lw=0.6)
        a.axvline(CUE, color=LAV, lw=0.8, ls="--", alpha=0.6)
        a.axvline(REWARD, color=MINT, lw=0.8, ls="--", alpha=0.6)
        a.set_ylim(-1.2, 1.2)
        a.set_title(title, loc="left", color=color, fontsize=10, fontweight="bold")
        a.set_xlabel("time")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.savefig("assets/dopamine_detective.png", dpi=130, bbox_inches="tight")
    print("saved assets/dopamine_detective.png")


if __name__ == "__main__":
    main()
