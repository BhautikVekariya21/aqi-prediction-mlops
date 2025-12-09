# Create fix_mutual_info.py
import re

file_path = "src/components/feature_selection.py"

with open(file_path, 'r') as f:
    content = f.read()

# Remove n_jobs from mutual_info_regression call
content = re.sub(
    r'(mutual_info_regression\([^)]*random_state=self\.random_state),\s*n_jobs=-1',
    r'\1',
    content
)

with open(file_path, 'w') as f:
    f.write(content)

print("✓ Fixed mutual_info_regression call")