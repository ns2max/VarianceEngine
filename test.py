import sys, torch
sys.path.insert(0, '.')
from audiocraft.models import MusicGen
mg = MusicGen.get_pretrained('facebook/musicgen-medium', device='cpu')
seen = set()
for name, mod in mg.lm.transformer.named_modules():
      if isinstance(mod, torch.nn.Linear):
          if 'self_attn' not in name and 'cross_attention' not in name:
              suffix = name.split('.')[-1]
              if suffix not in seen:
                  seen.add(suffix)
                  print(name, mod.in_features, '->', mod.out_features)
