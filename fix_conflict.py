import json

# Read the file as text
with open('preprocessing.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where the conflict markers are and extract the "ours" version
ours_lines = []
current = None

for line in lines:
    if '<<<<<<< Updated upstream' in line:
        current = 'ours'
    elif '=======' in line:
        current = 'theirs'
    elif '>>>>>>> Stashed changes' in line:
        current = None
    elif current == 'ours':
        ours_lines.append(line)
    elif current is None:
        ours_lines.append(line)

cleaned_content = ''.join(ours_lines)

# Validate it's valid JSON
try:
    json.loads(cleaned_content)
    print("✓ JSON is valid after cleanup")
    # Write back
    with open('preprocessing.ipynb', 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    print("✓ File updated successfully")
except json.JSONDecodeError as e:
    print(f"✗ JSON error: {e}")
