import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from unet_dataset import UnetDataset
import torch
from unet.unet_model import UNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from glob import glob

def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)), # 크기 고정 확인
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = UnetDataset(dataset_dir=os.path.join('sample_dataset', 'train'), transforms=transform)    # shape: [1, height, width]
    test_dataset = UnetDataset(dataset_dir=os.path.join('sample_dataset', 'test'), transforms=transform)

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # shape: [N, 1, height, width]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=1).to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam: learning rate를 자동 튜닝해줌

    best_test_loss = np.inf

    num_epochs = 1000

    train_loss_list = []
    test_loss_list = []

    if args.command=='train':
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch} / {num_epochs-1}")

            model.train()   # 학습 모드로 전환
            loss_sum = 0
            for images, masks in tqdm(train_loader, desc="Train"):
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()     # weight 업데이트 할 값 계산 (편미분, Gradient)
                optimizer.step()    # weight 업데이트 실행
                loss_sum += loss.item()*batch_size
            train_loss = loss_sum/len(train_dataset)
            train_loss_list.append(train_loss)

            model.eval()    # 검증 모드로 전환
            with torch.no_grad():
                loss_sum = 0
                accuracy_list = []
                for images, masks in tqdm(test_loader, desc="Test"):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss_sum += loss.item()*batch_size
            test_loss = loss_sum/len(test_dataset)
            test_loss_list.append(test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                os.makedirs('output', exist_ok=True)    # 'output' 폴더가 존재하면 만들지 마라는 뜻.
                torch.save(model.state_dict(), os.path.join('output', 'model_best.pth'))

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss:  {test_loss:.4f}")

            plt.plot(train_loss_list, marker='.', label='Train Loss')
            plt.plot(test_loss_list, marker='.', label='Test Loss')
            plt.legend()
            plt.grid()
            plt.title("Loss Curve")
            plt.savefig("loss_curve.png")
            plt.close()

    if args.command=='test':
        print("This is test mode.")

        # load model
        state_dict = torch.load(os.path.join('output', 'model_best.pth'), weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        image_paths = glob(os.path.join('sample_dataset', 'test', '*.bmp'))
        idx = 0

        image_path = image_paths[idx]
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sigmoid = nn.Sigmoid()
        with torch.no_grad():
            output = model(transform(image).unsqueeze(0).to(device))
            output = sigmoid(output)
        mask_pred = (output[0][0].to('cpu').numpy()*255).astype(np.uint8)
        h, w = image.shape
        mask_pred = cv2.resize(mask_pred, (w, h))
        image_mask = np.hstack((image, mask_pred))
        cv2.imshow('mask_pred', image_mask)
        key = cv2.waitKey()





"""parsing and configuration"""
def argparse_args():  
    desc = "Pytorch implementation of 'UNet Image Segmentation'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('command', help="'train' or 'test' or 'labeling'")
    parser.add_argument('--num_epochs', default=1000, type=int, help="The number of epochs to run")
    parser.add_argument('--batch_size', default=4, type=int, help="The number of mini-batchs for each epoch")
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = argparse_args()    
    if args is None:
        exit()
    print(args)
    
    main(args)