
from model import ModelRunner
from engine import static_batch

runner = ModelRunner()

# manual loop batch size 1
print(runner.generate_single("Count to 5."))

# static batching
prompts = [
    "Does Sam Altman have Aura?.",
    "Who is the 67 Kid?",
    "Is Northeastern a Fire School?.",
    "Who is Nick Vocatura",
]
for o in static_batch(prompts, runner.model, runner.tok):
    print(o[:80], "...")