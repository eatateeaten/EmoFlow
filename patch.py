FILE = "env/lib/python3.10/site-packages/torchcfm/models/unet/nn.py"

import re

def apply_patch():
    # Read the file
    with open(FILE, 'r') as f:
        content = f.read()
    
    # Replace GroupNorm32(32, channels) with GroupNorm32(8, channels)
    modified_content = re.sub(r'GroupNorm32\(32, channels\)', 'GroupNorm32(8, channels)', content)
    
    # Write the changes back
    with open(FILE, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully patched {FILE}")

if __name__ == "__main__":
    apply_patch()
