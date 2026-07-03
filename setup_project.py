from pathlib import Path

# ==========================================================
# AI Escalation Predictor - Project Setup Script
# ==========================================================

PROJECT_NAME = "AI-Escalation-Predictor"

directories = [
    "config",
    "detection",
    "tracking",
    "analysis",
    "alerts",
    "gui",
    "utils",
    "assets/icons",
    "assets/sounds",
    "data/videos",
    "data/samples",
    "logs",
    "tests",
    "docs",
]

files = {
    "app.py": '''def main():
    print("AI Escalation Predictor Started")


if __name__ == "__main__":
    main()
''',

    "README.md": '''# AI Escalation Predictor

A real-time AI-powered behavioral risk assessment system that estimates
the likelihood of physical conflict using computer vision and human pose analysis.

> 🚧 Project under active development.
''',

    "requirements.txt": "",

    ".gitignore": '''# Virtual Environment
venv/

# Python
__pycache__/
*.py[cod]
*.pyo

# Logs
logs/
*.log

# VS Code
.vscode/

# PyCharm
.idea/

# Environment Variables
.env

# Model Weights
*.pt
*.onnx

# Jupyter
.ipynb_checkpoints/

# Test Cache
.pytest_cache/
''',

}

package_dirs = [
    "config",
    "detection",
    "tracking",
    "analysis",
    "alerts",
    "gui",
    "utils",
    "tests"
]

print("\nCreating project structure...\n")

# Create directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"[DIR ] {directory}")

# Create __init__.py
for directory in package_dirs:
    init_file = Path(directory) / "__init__.py"
    init_file.touch(exist_ok=True)

# Create root files
for filename, content in files.items():
    file_path = Path(filename)

    if not file_path.exists():
        file_path.write_text(content, encoding="utf-8")
        print(f"[FILE] {filename}")
    else:
        print(f"[SKIP] {filename} already exists")

print("\nProject setup completed successfully!")