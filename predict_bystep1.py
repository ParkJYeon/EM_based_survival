import torch
import math
import pickle
from model_step1 import cal_discriminative


def get_predict_all_model(model1, model2, num_bag, batch_size, data_loader, lr):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        prediction = {}
        for instance in range(num_bag):
            x, y, wsi = next(data_loader)
            x = torch.Tensor(x)
            x = x.reshape(len(x), 3, 256, 256)
            num_batch = math.ceil(len(x) / batch_size)
            prediction[wsi] = {0: 0, 1: 0, 'label': y[0]}
            for batch in range(num_batch):
                batch_x, batch_y = x[batch * batch_size: (batch + 1) * batch_size], y[batch * batch_size: (
                                                                                                                  batch + 1) * batch_size]
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                batch_output1 = model1(batch_x)
                batch_output1 = batch_output1.argmax(dim=1)
                batch_output2 = model2(batch_x)
                batch_output2 = batch_output2.argmax(dim=1)
                for i in range(len(batch_output1)):
                    if batch_output1[i] == 0:
                        prediction[wsi][0] += 1
                    else:
                        prediction[wsi][1] += 1
                    if batch_output2[i] == 0:
                        prediction[wsi][0] += 1
                    else:
                        prediction[wsi][1] += 1
                del batch_x
                del batch_y
                del batch_output1
                del batch_output2
            del x
            del y
            del wsi
    file_name = 'prediction_allwsi_' + str(lr) + '.txt'
    with open(file_name, 'wb') as f:
        pickle.dump(prediction, f)
    return prediction
