import torch
from torch import nn
from torchvision import transforms

class SpeciesnetFeatures(nn.Module):
    def __init__(self, graph_module, feature_node_name):
        super().__init__()
        self.graph = graph_module
        self.feature_node_name = feature_node_name
        self._features = None
        
        # register hook on the ONNX-traced layer
        layer = getattr(self.graph, feature_node_name)
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = out

    def forward(self, x):
        _ = self.graph(x)
        return self._features

class NewClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SpeciesnetPolish(nn.Module):
    def __init__(self, feature_extractor: SpeciesnetFeatures, num_classes: int):
        super().__init__()
        self.feature_extractor = feature_extractor

        # freeze original SpeciesNet
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # get feature dim
        dummy = torch.randn(1, 480, 480, 3).to("cuda")
        feat = self.feature_extractor(dummy)
        feature_dim = feat.shape[1]

        self.classifier = NewClassifier(feature_dim, num_classes)

        self.transform = transforms.Compose([
            transforms.Resize((480, 480), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        f = self.feature_extractor(x)
        out = self.classifier(f)
        return out

def predict(model, image, class_names: list[str], top_k: int = 5):
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

def predict_batch(model, pil_images, class_names, top_k=5):
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

def get_feature_extractor(
        model_path: str = "speciesnet/models/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.pt",
        feature_node: str = "SpeciesNet/efficientnetv2-m/avg_pool/Mean_Squeeze__3825"):
    model = torch.load(model_path, map_location="cuda", weights_only=False)

    for p in model.parameters():
        p.requires_grad = False
    
    return SpeciesnetFeatures(model, feature_node).to("cuda")

def get_model(model_path: str = 'speciesnet/models/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.pt',
              checkpoint_path: str = 'fine_tuning/speciesnet_head/speciesnet_polish_checkpoint.pt') -> tuple[SpeciesnetPolish, list]:
    ckpt = torch.load(checkpoint_path)

    feat_extractor = get_feature_extractor(model_path)
    model = SpeciesnetPolish(feat_extractor, ckpt['num_classes'])
    model.load_state_dict(ckpt['state_dict'])

    class_names = ckpt['class_names']

    return model, class_names
