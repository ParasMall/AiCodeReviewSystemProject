import torch
import argparse
import difflib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_models():
    """Load the fine-tuned CodeBERT and CodeT5 models"""
    print("Loading CodeBERT model for bug detection...")
    codebert_tokenizer = AutoTokenizer.from_pretrained("models/codebert_model")
    codebert_model = AutoModelForSequenceClassification.from_pretrained("models/codebert_model").to(device)
    
    print("Loading CodeT5 model for bug fixing...")
    codet5_tokenizer = AutoTokenizer.from_pretrained("models/codet5_model")
    codet5_model = T5ForConditionalGeneration.from_pretrained("models/codet5_model").to(device)
    
    return codebert_tokenizer, codebert_model, codet5_tokenizer, codet5_model

def detect_bugs(code, tokenizer, model):
    """Use CodeBERT to detect if code is buggy"""
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        is_buggy = predictions[0, 1].item() > 0.5  # Class 1 is "buggy"
        confidence = predictions[0, 1].item() if is_buggy else predictions[0, 0].item()
    
    return is_buggy, confidence

def fix_bugs(code, tokenizer, model):
    """Use CodeT5 to generate fixed code"""
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
    fixed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fixed_code

def generate_diff(original_code, fixed_code):
    """Generate a unified diff between original and fixed code"""
    original_lines = original_code.splitlines(keepends=True)
    fixed_lines = fixed_code.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile='original',
        tofile='fixed',
        n=3
    )
    
    return ''.join(list(diff))

def generate_line_by_line_review(original_code, fixed_code):
    """Generate a line-by-line review with specific comments"""
    original_lines = original_code.splitlines()
    fixed_lines = fixed_code.splitlines()
    
    matcher = difflib.SequenceMatcher(None, original_lines, fixed_lines)
    review = []
    
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'replace':
            for i in range(i1, i2):
                line_num = i + 1
                if i < len(original_lines):
                    review.append(f"Line {line_num}: [Issue] '{original_lines[i]}' should be replaced.")
                    if j1 + (i - i1) < len(fixed_lines):
                        review.append(f"Suggestion: '{fixed_lines[j1 + (i - i1)]}'")
                    review.append("---")
        elif op == 'delete':
            for i in range(i1, i2):
                line_num = i + 1
                if i < len(original_lines):
                    review.append(f"Line {line_num}: [Issue] '{original_lines[i]}' should be removed.")
                    review.append("---")
        elif op == 'insert':
            line_num = i1
            review.append(f"After line {line_num}: [Issue] Missing code.")
            for j in range(j1, j2):
                if j < len(fixed_lines):
                    review.append(f"Suggestion: Add '{fixed_lines[j]}'")
            review.append("---")
    
    if not review:
        return ["No specific issues found in the code."]
    
    return review

def review_java_code(code):
    """Review Java code using the trained models"""
    codebert_tokenizer, codebert_model, codet5_tokenizer, codet5_model = load_models()
    
    # Bug detection
    print("Detecting bugs...")
    is_buggy, confidence = detect_bugs(code, codebert_tokenizer, codebert_model)
    
    if not is_buggy:
        return {
            "bug_status": "No bugs detected",
            "confidence": confidence * 100,
            "suggested_fix": None,
            "review_comments": ["Code appears to be without bugs."],
            "diff": None
        }
    
    # Bug fixing
    print("Fixing bugs...")
    fixed_code = fix_bugs(code, codet5_tokenizer, codet5_model)
    
    # Generate diff
    diff = generate_diff(code, fixed_code)
    
    # Generate line-by-line review
    review_comments = generate_line_by_line_review(code, fixed_code)
    
    return {
        "bug_status": "Bugs detected",
        "confidence": confidence * 100,
        "suggested_fix": fixed_code,
        "review_comments": review_comments,
        "diff": diff
    }

def main():
    parser = argparse.ArgumentParser(description="AI-powered Java code review")
    parser.add_argument("--file", type=str, help="Path to the Java file to review")
    parser.add_argument("--code", type=str, help="Java code string to review")
    args = parser.parse_args()
    
    if args.file:
        with open(args.file, 'r') as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("Please enter the Java code to review:")
        code = ""
        while True:
            line = input()
            if line.strip() == "EOF":
                break
            code += line + "\n"
    
    results = review_java_code(code)
    
    print("\n" + "="*50)
    print(f"Bug Status: {results['bug_status']} (Confidence: {results['confidence']:.2f}%)")
    print("="*50)
    
    if results['suggested_fix']:
        print("\nSuggested Fix:\n")
        print(results['suggested_fix'])
        
        print("\nDiff:\n")
        print(results['diff'])
    
    print("\nLine-by-Line Review:\n")
    for comment in results['review_comments']:
        print(comment)

if __name__ == "__main__":
    main() 