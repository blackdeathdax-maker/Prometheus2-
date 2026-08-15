# Repo Audit Python Tool Script
# This script bundles your code into a single markdown file for a ChatGPT audit.
# Save this file as bundle_repo.py and run it in your repo root.

import os

# Target file extensions to audit
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.cpp', '.h', '.cs', '.html', '.css'}
IGNORE_DIRS = {'node_modules', '.git', '__pycache__', 'dist', 'build', 'venv', '.next'}

def bundle_repo(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# Repository Code Bundle for ChatGPT Audit\n\n")
        outfile.write("Below is the structure and codebase of the project.\n\n")
        
        # Write directory tree structure
        outfile.write("## Project Structure\n```\n")
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            outfile.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                if os.path.splitext(f)[1] in ALLOWED_EXTENSIONS:
                    outfile.write(f"{sub_indent}{f}\n")
        outfile.write("```\n\n## Source Code Files\n\n")
        
        # Write file contents
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in ALLOWED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_dir)
                    outfile.write(f"### File: {rel_path}\n")
                    outfile.write(f"```{ext[1:]}\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"// Error reading file: {str(e)}")
                    outfile.write("\n```\n\n")

if __name__ == '__main__':
    bundle_repo('.', 'project_codebase.md')
    print("Bundle created successfully as project_codebase.md!")
