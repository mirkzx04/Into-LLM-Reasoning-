from MATH_logic.rlvr_pipeline.rewards_utils import acc_reward, format_reward

def evaluate_generated_answers(results, pass_k = None):
    total_problems = 0

    acc1_correct = 0
    passk_correct = 0

    total_generations = 0
    sample_correct = 0

    total_format = 0
    format_correct = 0

    for pid, item in results.items():
        # Take k model answer
        answers = item["answers"]
        solution = item["solution"]

        k = len(answers) if pass_k is None else min(pass_k, len(answers))
        answers_k = answers[:k]
        solutions_k = [solution] * len(answers_k)

        # Compute accuracy reward and format reward
        acc_rewards = acc_reward(completions=answers_k, solution=solutions_k)
        fmt_rewards = format_reward(completions=answers_k)

        # Check which answers got accuracy
        correct_flags = [r > 0.5 for r in acc_reward]
        format_flags = [r == 1.0 for r in fmt_rewards]

        acc1 = correct_flags[0]
        passk = any(correct_flags)

        total_problems += 1

        acc1_correct += int(acc1)
        passk_correct += int(passk)

        total_generations += len(correct_flags)
        sample_correct += sum(correct_flags)

        total_format += len(format_flags)
        format_correct += sum(format_flags)

    return {
        "acc@1": acc1_correct / total_problems,
        f"pass@{k}": passk_correct / total_problems,
        "sample_acc": sample_correct / total_generations,
        "format_acc": format_correct / total_format,
    }