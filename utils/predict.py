import torch
from PIL import Image


def predict(model, image: Image.Image, class_names: list[str], top_k: int = 5):
    """image is PIL Image file."""
    model.eval()

    # image: PIL image
    x = model.transform(image).unsqueeze(0).cuda() # (1, 3, 480, 480)
    x.to('cuda')

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(top_k, dim=1)
    
    results = []
    for p, idx in zip(top_probs[0], top_idxs[0]):
        results.append((class_names[idx], float(p.item())))

    return results

def predict_batch(model, pil_images: list[Image.Image], class_names, top_k: int=5):
    """
    pil_images: list of PIL.Image
    returns: list of list of (classname, prob)
    """
    model.eval()

    # Transform images → stack into a batch
    xs = [model.transform(im) for im in pil_images]  # list of tensors (3, 480, 480)
    x = torch.stack(xs).to('cuda')                   # (B, 3, 480, 480)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(top_k, dim=1)

    results = []
    for i in range(len(pil_images)):
        r = []
        for p, idx in zip(top_probs[i], top_idxs[i]):
            r.append((class_names[idx], float(p.item())))
        results.append(r)

    return results
