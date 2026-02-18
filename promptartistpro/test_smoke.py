"""Quick smoke test for intent classification."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import preprocess
from prompt_dispatcher import PromptDispatcher
import joblib
import numpy as np

artifact = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'intent_classifier.joblib'))
pipeline = artifact['pipeline']
le = artifact['label_encoder']
dispatcher = PromptDispatcher()

tests = [
    ("Write a Cypress test for login", "TESTING"),
    ("Calculate delta for a call option", "GREEKS"),
    ("Help me reconcile trade data", "DATA_RECONCILIATION"),
    ("Generate a PnL report", "FINANCE"),
    ("Write a python function", "CODING"),
    ("Create marketing email", "MARKETING"),
    ("Hello how are you", "GENERAL"),
]

print("=" * 60)
print("  SMOKE TEST - Intent Classification")
print("=" * 60)

all_pass = True
for text, expected in tests:
    processed = preprocess(text)
    scores = pipeline.decision_function([processed])[0]
    idx = int(np.argmax(scores))
    predicted = le.classes_[idx]
    status = "PASS" if predicted == expected else "FAIL"
    if predicted != expected:
        all_pass = False
    tmpl_count = len(dispatcher.get_templates(predicted))
    print(f"  {status} | \"{text}\"")
    print(f"       -> Predicted: {predicted} (expected: {expected}), Templates: {tmpl_count}")

print("=" * 60)
print(f"  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
print("=" * 60)
