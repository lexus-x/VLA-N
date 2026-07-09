import sys
sys.path.insert(0, r"C:\Users\islab01\vla-atlas")
from heads.backbone import SmolVLM2Backbone

bb = SmolVLM2Backbone(device="cuda")
m = bb.model
print(type(m).__name__)
for name, child in m.named_children():
    n_params = sum(p.numel() for p in child.parameters())
    print(f"  {name}: {type(child).__name__}  params={n_params/1e6:.1f}M")
    for name2, child2 in child.named_children():
        n2 = sum(p.numel() for p in child2.parameters())
        print(f"    {name}.{name2}: {type(child2).__name__}  params={n2/1e6:.1f}M")
