import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
import json
import difflib
from typing import Dict, List, Tuple, Optional
import os

class IntegratedCodeReview:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CodeBERT for bug detection
        print("Loading CodeBERT model...")
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("models/codebert_model")
        self.codebert_model = AutoModelForSequenceClassification.from_pretrained("models/codebert_model")
        self.codebert_model.to(self.device)
        self.codebert_model.eval()
        
        # Load CodeT5 for bug fixing
        print("Loading CodeT5 model...")
        self.codet5_tokenizer = AutoTokenizer.from_pretrained("models/codet5_model")
        self.codet5_model = T5ForConditionalGeneration.from_pretrained("models/codet5_model")
        self.codet5_model.to(self.device)
        self.codet5_model.eval()

    def detect_bugs(self, code: str) -> Tuple[bool, float]:
        """Detect if code contains bugs using CodeBERT."""
        inputs = self.codebert_tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            bug_probability = probabilities[0][1].item()
            has_bug = bug_probability > 0.5
            
        return has_bug, bug_probability

    def fix_bugs(self, code: str) -> str:
        """Generate fixed code using CodeT5."""
        inputs = self.codet5_tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.codet5_model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
        fixed_code = self.codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fixed_code

    def generate_diff(self, original_code: str, fixed_code: str) -> str:
        """Generate a diff between original and fixed code."""
        original_lines = original_code.splitlines(keepends=True)
        fixed_lines = fixed_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile='original',
            tofile='fixed',
            n=3
        )
        
        return ''.join(diff)

    def review_code(self, code: str) -> Dict:
        """Perform complete code review with bug detection and fixing."""
        # Step 1: Detect bugs
        has_bug, bug_probability = self.detect_bugs(code)
        
        result = {
            "has_bugs": has_bug,
            "bug_probability": bug_probability,
            "fixed_code": None,
            "diff": None,
            "review_comments": []
        }
        
        if has_bug:
            # Step 2: Fix bugs
            fixed_code = self.fix_bugs(code)
            result["fixed_code"] = fixed_code
            
            # Step 3: Generate diff
            result["diff"] = self.generate_diff(code, fixed_code)
            
            # Step 4: Add review comments
            result["review_comments"].append(
                f"Bug detected with {bug_probability:.2%} confidence."
            )
            if fixed_code != code:
                result["review_comments"].append(
                    "Suggested fixes have been generated. Please review the diff."
                )
        else:
            result["review_comments"].append(
                f"No bugs detected (confidence: {(1-bug_probability):.2%})"
            )
        
        return result

def main():
    # Example usage
    reviewer = IntegratedCodeReview()
    
    # Read Java file
    java_file = "test.java"  # You can change this to any Java file
    if not os.path.exists(java_file):
        print(f"Error: {java_file} not found!")
        return
    
    with open(java_file, 'r') as f:
        code = f.read()
    
    # Perform review
    print(f"\nReviewing {java_file}...")
    review_result = reviewer.review_code(code)
    
    # Print results
    print("\nReview Results:")
    print("=" * 50)
    print(f"Bug Detection: {'Bugs found!' if review_result['has_bugs'] else 'No bugs detected'}")
    print(f"Confidence: {review_result['bug_probability']:.2%}")
    
    if review_result['has_bugs']:
        print("\nDiff of suggested fixes:")
        print(review_result['diff'])
    
    print("\nReview Comments:")
    for comment in review_result['review_comments']:
        print(f"- {comment}")

if __name__ == "__main__":
    main() 