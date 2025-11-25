import torch
from torchvision.transforms import transforms, InterpolationMode


CROP_SIZE = 480


class Speciesnet:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()
        self.model.half()
        print(f'Speciesnet loaded onto {self.device}')

        # transform image to form usable by network
        self.transforms = transforms.Compose([
            transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def predictOnBatch(self, batch_tensor, withsoftmax=True):
        batch_tensor = batch_tensor.to(self.device).half()
        with torch.no_grad():
            logits = self.model(batch_tensor)
            preds = logits.softmax(dim=1) if withsoftmax else logits
        return preds.cpu().numpy()

    def preprocessImage(self, croppedimage):
        return self.transforms(croppedimage).unsqueeze(dim=0)  # batch dimension


def get_model(model_path: str = 'speciesnet/models/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.pt') -> tuple[Speciesnet, list[str]]:
    """Return Speciesnet model and class names."""
    labels_file = 'speciesnet/models/speciesnet-pytorch-v4.0.1a-v1/always_crop_99710272_22x8_v12_epoch_00148.labels.txt'
    with open(labels_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    model = Speciesnet(model_path)
    return model, class_names
