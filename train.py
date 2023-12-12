import pandas as pd
import torch
import numpy as np
from torch import nn
import PIL.Image as Image
import glob


def accuracy_fn(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.sqrt(np.sum((y_true - y_pred) ** 2))
    return acc


loss_fn = nn.MSELoss()


def train(model, epochs, path_to_dataset, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    image_list = []
    the_csv = pd.read_csv(path_to_dataset + "/Label.csv")
    the_csv = the_csv[["file name", "x-coordinate", "y-coordinate"]]
    y = []

    for filename in glob.glob(path_to_dataset + "/*.jpg"):
        the_filename = filename.split("/")[-1]
        [cx, cy] = the_csv[the_csv["file name"] == the_filename][
            ["x-coordinate", "y-coordinate"]
        ].values[0]
        if np.isnan(cx) or np.isnan(cy):
            cx = -1
            cy = -1
        y.append([the_filename, int(cx), int(cy)])
        im = Image.open(filename)
        im = torch.from_numpy(np.array(im)).float().permute(2, 0, 1).unsqueeze(0)
        image_list.append(im)
    y = np.array(y)
    y = pd.DataFrame(y, columns=["file name", "x-coordinate", "y-coordinate"])

    X_train = torch.cat(image_list[: int(len(image_list) * 0.8)], dim=0)
    X_test = torch.cat(image_list[int(len(image_list) * 0.8) :], dim=0)

    y = y[["x-coordinate", "y-coordinate"]].to_numpy(int)
    y_train = torch.from_numpy(y[: int(len(y) * 0.8)]).float()
    y_test = torch.from_numpy(y[int(len(y) * 0.8) :]).float()

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    print("Dataset loaded!")
    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass
        y_logits = model(X_train)  # model outputs raw logits
        y_pred = y_logits
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test)
            test_pred = test_logits
            # 2. Calculate test loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Print out what's happening
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%"
            )
