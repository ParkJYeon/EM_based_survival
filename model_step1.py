import torch
import torch.nn as nn
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pickle


def get_discrim_patch(wsi_name, latent_bag, x_train, x_label):
    latent_patch = latent_bag[wsi_name]
    train_x = [x_train[i] for i in range(len(latent_patch)) if latent_patch[i][1] == 1]
    train_label = [x_label[i] for i in range(len(latent_patch)) if latent_patch[i][1] == 1]
    del latent_patch
    return train_x, train_label


def get_predict_all_patch(model1, model2, num_bag, batch_size, data_loader):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        prediction = {}
        for instance in range(num_bag):
            x, y, wsi = next(data_loader)
            x = torch.Tensor(x)
            x.permute(0, 3, 1, 2)
            #x = torch.transpose(dis_x, (0, 3, 1, 2))
            num_batch = math.ceil(len(x) / batch_size)
            output = []
            prediction[wsi + str(y[0])] = []
            for batch in range(num_batch):
                batch_x, batch_y = x[batch * batch_size: (batch + 1) * batch_size], y[batch * batch_size: (
                                                                                                                  batch + 1) * batch_size]
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                batch_output1 = model1(batch_x)
                batch_output2 = model2(batch_x)
                if batch_y[0] == '0':
                    batch_output1, batch_output2 = batch_output1[:, 0], batch_output2[:, 0]
                else:
                    batch_output1, batch_output2 = batch_output1[:, 1], batch_output2[:, 1]
                output.append(((batch_output1 + batch_output2) / 2))
                del batch_x
                del batch_y
            for i in range(1, len(output)):
                output[0] = torch.cat((output[0], output[i]), dim=0)
            prediction[wsi + str(y[0])].append(output[0].cpu().numpy())
            del x
            del y
            del wsi
            del output

    for i in prediction:
        prediction[i] = gaussian_filter(prediction[i], sigma=10)

    return prediction


def thresholding_discriminative(prediction_slide, prediction_class, latent_bag, p1, p2):
    H_i = {}
    R_i = {}

    for i in prediction_slide:
        H_i[i[0:7]] = np.percentile(prediction_slide[i][0], p1)
    for i in prediction_class:
        R_i[i] = np.percentile(prediction_class[i][0], p2)

    for i in prediction_slide:
        R = R_i[int(i[7])]
        H = H_i[i[0:7]]
        T = min(R, H)
        for j in range(len(prediction_slide[i][0])):
            if prediction_slide[i][0][j] > T:
                latent_bag[i[0:7]][j][1] = 1
            else:
                latent_bag[i[0:7]][j][1] = 0
    return latent_bag


def cal_discriminative(model1, model2, latent_bag, num_bag, batch_size, data_loader, p1, p2):
    prediction_slide = get_predict_all_patch(model1, model2, num_bag, batch_size, data_loader)
    prediction_class = {0: [prediction_slide[i] for i in prediction_slide if i[-1] == str(0)],
                        1: [prediction_slide[i] for i in prediction_slide if i[-1] == str(1)]}

    latent_bag = thresholding_discriminative(prediction_slide, prediction_class, latent_bag, p1, p2)
    return latent_bag


def count_discrim(latent_bag):
    for i in latent_bag:
        count = 0
        for j in range(len(latent_bag[i])):
            if latent_bag[i][j][1] == 1:
                count += 1
        print(i, count)
        del count


def train_step1(model1, model2, model1_iter, model2_iter, trn_num_bag, val_num_bag, batch_size, learning_rate, latent_bag, trn_loader, val_loader,
                per1, per2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model2.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    # criterion1 = nn.BCELoss()
    # criterion2 = nn.BCELoss()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)


    trn1_loss = []
    trn2_loss = []
    trn1_acc = []
    trn2_acc = []
    val1_loss = []
    val2_loss = []
    val1_acc = []
    val2_acc = []

    epochs = max(model1_iter, model2_iter)

    for epoch in range(epochs//2):
        print("{} - {} epochs".format(learning_rate, (epoch * 2)))
        for i in range(2):
            model1.train()
            model2.train()
            loss1_bag, loss2_bag, accu1_bag, accu2_bag = 0, 0, 0, 0
            # print("num of discriminative patches during training : ", end=" ")
            # count_discrim(latent_bag)
            for instance in range(trn_num_bag):
                train_x, train_y, train_wsi = next(trn_loader)
                dis_x, dis_y = get_discrim_patch(train_wsi, latent_bag, train_x, train_y)
                dis_x = torch.Tensor(dis_x)
                dis_x.permute(0, 3, 1, 2)
                #dis_x = torch.transpose(dis_x, (0, 3, 1, 2))
                num_batch = math.ceil(len(dis_x) / batch_size)
                loss1, accu1, loss2, accu2 = 0, 0, 0, 0
                for batch in range(num_batch):
                    x, y = dis_x[batch * batch_size: (batch + 1) * batch_size], dis_y[batch * batch_size: (
                                                                                                                  batch + 1) * batch_size]
                    if torch.cuda.is_available():
                        x = torch.Tensor(np.asarray(x)).to(device)
                        y = torch.from_numpy(np.asarray(y)).to(torch.long).to(device)
                    if model1_iter > (epoch * 2):
                        model1_output = model1(x)
                        loss = criterion1(model1_output, y)
                        optimizer1.zero_grad()
                        loss.backward()
                        optimizer1.step()
                        loss1 += loss.item()
                        model1_output = model1_output.argmax(dim=1)
                        corr = y[y == model1_output].size(0)
                        total = len(x)
                        accu1 += ((corr / total) * 100)
                        del loss
                        del corr
                        del total
                        del model1_output
                        torch.cuda.empty_cache()
                    if model2_iter > (epoch * 2):
                        model2_output = model2(x)
                        loss = criterion2(model2_output, y)
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer2.step()
                        loss2 += loss.item()
                        model2_output = model2_output.argmax(dim=1)
                        corr = y[y == model2_output].size(0)
                        total = len(x)
                        accu2 += ((corr / total) * 100)
                        del loss
                        del corr
                        del total
                        del model2_output
                        torch.cuda.empty_cache()
                    del x
                    del y
                    torch.cuda.empty_cache()
                loss1_bag += (loss1 / num_batch)
                loss2_bag += (loss2 / num_batch)
                accu1_bag += (accu1 / num_batch)
                accu2_bag += (accu2 / num_batch)
                del loss1
                del loss2
                del accu1
                del accu2
                del train_x
                del train_y
                del train_wsi
                del dis_x
                del dis_y
                torch.cuda.empty_cache()

            model1.eval()
            model2.eval()
            vloss1_bag, vloss2_bag, vacc1_bag, vacc2_bag = 0, 0, 0, 0

            with torch.no_grad():
                for instance in range(val_num_bag):
                    valid_x, valid_y, valid_wsi = next(val_loader)
                    valid_x = torch.Tensor(valid_x)
                    #valid_x = valid_x.reshape(len(valid_x), 3, 256, 256)
                    #valid_x = torch.transpose(valid_x, (0, 3, 1, 2))
                    valid_x.permute(0, 3, 1, 2)
                    num_batch = math.ceil(len(valid_x) / batch_size)
                    val_loss1, val_acc1, val_loss2, val_acc2 = 0,0,0,0
                    for batch in range(num_batch):
                        x, y = valid_x[batch * batch_size : (batch+1) * batch_size], valid_y[batch * batch_size : (batch+1) * batch_size]
                        if torch.cuda.is_available():
                            x = torch.Tensor(np.asarray(x)).to(device)
                            y = torch.from_numpy(np.asarray(y)).to(torch.long).to(device)
                        if model1_iter > (epoch * 2):
                            model1_output = model1(x)
                            loss = criterion1(model1_output, y)
                            val_loss1 += loss.item()
                            model1_output = model1_output.argmax(dim=1)
                            corr = y[y==model1_output].size(0)
                            total = len(x)
                            val_acc1 += ((corr/total) * 100)
                            del loss
                            del corr
                            del total
                            del model1_output
                            torch.cuda.empty_cache()
                        if model2_iter > (epoch * 2):
                            model2_output = model2(x)
                            loss = criterion2(model2_output, y)
                            val_loss2 += loss.item()
                            model2_output = model2_output.argmax(dim=1)
                            corr = y[y==model2_output].size(0)
                            total = len(x)
                            val_acc2 += ((corr/total)*100)
                            del loss
                            del corr
                            del total
                            del model2_output
                            torch.cuda.empty_cache()
                        del x
                        del y
                        torch.cuda.empty_cache()
                    vloss1_bag += (val_loss1 / num_batch)
                    vloss2_bag += (val_loss2 / num_batch)
                    vacc1_bag += (val_acc1 / num_batch)
                    vacc2_bag += (val_acc2 / num_batch)
                    del val_loss1
                    del val_loss2
                    del val_acc1
                    del val_acc2
                    del valid_x
                    del valid_y
            trn1_acc.append(accu1_bag / trn_num_bag)
            trn1_loss.append(loss1_bag / trn_num_bag)
            trn2_acc.append(accu2_bag / trn_num_bag)
            trn2_loss.append(loss2_bag / trn_num_bag)
            val1_acc.append(vacc1_bag / val_num_bag)
            val1_loss.append(vloss1_bag / val_num_bag)
            val2_acc.append(vacc2_bag / val_num_bag)
            val2_loss.append(vloss2_bag / val_num_bag)
        latent_bag = cal_discriminative(model1, model2, latent_bag, trn_num_bag, batch_size, trn_loader, per1, per2)
        # count_discrim(latent_bag)
        if(((epoch*2)%10) == 0):
            print(trn1_loss)
            print(trn1_acc)
            print(trn2_loss)
            print(trn2_acc)
            print(val1_loss)
            print(val1_acc)
            print(val2_loss)
            print(val2_acc)

    model_name = "model" + str(learning_rate) + ".pth"
    torch.save({
        'model1': model1.state_dict(),
        'model2': model2.state_dict(),
    }, model_name)
    latent_value = "latent_" + str(learning_rate) + ".txt"
    with open(latent_value, "wb") as f:
        pickle.dump(latent_bag, f)
    return model1, model2
