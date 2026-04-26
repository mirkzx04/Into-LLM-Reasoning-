import os 
import sys
import matplotlib.pyplot as plt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from operator import itemgetter

from MATH_logic.eval_pipeline.generation import generate_answer
from MATH_logic.eval_pipeline.metric import evaluate_generated_answers
from MATH_logic.rlvr_pipeline.rewards_utils import acc_reward, format_reward

def plot_eval_metrics(metrics, title="Evaluation metrics", save_path=None):
    """
    Plots evaluation metrics as a bar chart.

    metrics example:
    {
        "acc@1": 0.42,
        "pass@6": 0.58,
        "sample_acc": 0.31,
        "format_acc": 0.90,
    }
    """

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)

    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()

def validate_model_answer(n_samples, sample_iter):

    results = generate_answer(n_samples = 2, sample_iter = 2, do_sample = True, batch_size=2)

    metrics = evaluate_generated_answers(results, sample_iter)

    plot_eval_metrics(metrics, title = f"Model evalutation - pass@{sample_iter}")