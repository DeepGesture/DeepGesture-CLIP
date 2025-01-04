# DeepGesture-CLIP

Gesture

## Usage

### Installation

```bash
git clone https://github.com/DeepGesture/DeepGesture-CLIP
```

### Getting embeddings

```python
import torch
import clip

CHECKPOINT_PATH = "<path to checkpoint>"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CHECKPOINT_PATH, device=device)

with torch.no_grad():
    features_from_inputs = model.embed({
        'text': torch.randn(580, 768),
        'audio': torch.randn(580, 768)  # You can send mismatched size, it will interpolate and send output as the maxed value
    })
    features_from_target_motion = model.embed_motion({
        'motion': torch.randn(500, 60)
    })

    logits_per_image, logits_per_text = features_from_inputs.mean(1), features_from_target_motion.mean(1)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```
