import torch

# https://stackoverflow.com/questions/78114412/import-torch-how-to-fix-oserror-winerror-126-error-loading-fbgemm-dll-or-depen

class_names = ['birds', 'mammals', 'reptiles']

transforms = torch.load("transforms.pth", map_location=torch.device('cpu'))
model = torch.load("model.pth", map_location=torch.device('cpu'))
model.eval()

def pred_img(img, model=model, transforms=transforms):

    with torch.inference_mode():
        img_t = transforms(img).unsqueeze(0)
        pred = model(img_t).softmax(dim=1)
        pred_prob = pred.max(dim=1)[0].item()
        pred_label = class_names[pred.argmax(dim=1)]

        return f"Predicted class: {pred_label} | Probability: {pred_prob * 100:.2f} %", pred_label
