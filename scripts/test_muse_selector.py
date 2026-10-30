import json

from dotenv import load_dotenv

from realitycheck.muse_selector import choose_next_experiment

load_dotenv()

history = [
    {
        "transform": "brightness",
        "value": 0.6,
        "baseline_detections": 20,
        "perturbed_detections": 18,
        "matched": 18,
        "lost": 2,
        "detection_persistence": 0.90,
        "mean_confidence_delta": -0.018,
    }
]

proposal = choose_next_experiment(history)

print("\n=== MUSE SPARK EXPERIMENT PROPOSAL ===")
print(json.dumps(proposal, indent=2))
