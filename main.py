from dataloader import DataGenerator
from Instance_bag import data_generation_MIL
from Model import ConvolutionNN
import torch
from torchvision import transforms
from model_step1 import train_step1, count_discrim
from predict_bystep1 import get_predict_all_model
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_step
# step1_model train

#
# file_name = 'train_Instance.txt'
# with open(file_name, 'wb') as f:
#     pickle.dump(train_Instance, f)
#
# file_name = 'train_Instance_label.txt'
# with open(file_name, 'wb') as f:
#     pickle.dump(train_Instance_label, f)
#
# file_name = 'train_Instance_latent.txt'
# with open(file_name, 'wb') as f:
#     pickle.dump(train_Instance_latent, f)


batch_size = 128
#
model1_epoch = 40
model2_epoch = 50
learning_rate = [0.000001]

#
for i in learning_rate:

    # print("initial discriminative patch : ", end =" ")
    model1, model2 = ConvolutionNN(), ConvolutionNN()
    train_Instance, train_Instance_label, train_Instance_latent = data_generation_MIL(
        "C:\\Users\\yeon\\datasets\\CancerClassify\\Train")
    valid_Instance, valid_Instance_label, valid_Instance_latent = data_generation_MIL(
        "C:\\Users\\yeon\\datasets\\CancerClassify\\Valid")
    train_generator = DataGenerator(1).generate(train_Instance, train_Instance_label)
    valid_generator = DataGenerator(1).generate(valid_Instance, valid_Instance_label)
    num_bag = len(train_Instance)
    val_bag = len(valid_Instance)
    model1, model2 = train_step1(model1, model2, model1_epoch, model2_epoch, trn_num_bag=num_bag, val_num_bag=val_bag,
                                 batch_size=batch_size,
                                 learning_rate=i, latent_bag=train_Instance_latent, trn_loader=train_generator,
                                 val_loader=valid_generator, per1=65, per2=65)
    del train_Instance
    del train_Instance_label
    del train_Instance_latent
    del valid_Instance
    del valid_Instance_label
    del valid_Instance_latent
    del train_generator
    del valid_generator
    test_Instance, test_Instance_label, test_Instance_latent = data_generation_MIL("C:\\Users\\yeon\\datasets\\CancerClassify\\Test0")
    generator = DataGenerator(1).generate(test_Instance, test_Instance_label)
    prediction = get_predict_all_model(model1, model2, num_bag, batch_size, generator, i)

#train_Instance, train_Instance_label, train_Instance_latent = data_generation_MIL("C:\\Users\\yeon\\datasets\\CancerClassify\\Train")
#train_generator = DataGenerator(1).generate(train_Instance, train_Instance_label)

#print(train_generator[0])

#
# get prediction from step1_model

# model1, model2 = ConvolutionNN(), ConvolutionNN()
# checkpoint = torch.load("C:\\Users\\yeon\\PycharmProjects\\EM-based_Survival\\modelpth\\model1e-06.pth")
# model1.load_state_dict(checkpoint['model1'])
# model2.load_state_dict(checkpoint['model2'])

# trn_prediction = get_predict_all_model(model1, model2, num_bag, batch_size, train_generator, 0.000001)

# step2_model train
