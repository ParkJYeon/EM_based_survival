from torchvision import transforms
from PIL import Image

class DataGenerator(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    @staticmethod
    def __Get_exploration_order(dataset):
        indexes = list(dataset.keys())
        return indexes

    @staticmethod
    def __Data_Generation(batch_data, batch_label):
        bag_batch = []
        bag_label = []
        bag_latent = []

        transfers = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        for ibatch, batch in enumerate(batch_data):
            aug_batch = []
            # img_data = imread(batch)
            img_data = Image.open(batch)
            input_batch = transfers(img_data)
            input_batch = input_batch.numpy()

            bag_batch.append((input_batch))
            bag_label.append(batch_label)

        return bag_batch, bag_label, batch[46:53]

    def generate(self, data, data_label):
        while 1:
            indexes = self.__Get_exploration_order(data)

            for i in indexes:
                Batch_data = data[i]
                Batch_label = data_label[i]
                X, y, wsi_name = self.__Data_Generation(Batch_data, Batch_label)
                yield X, y, wsi_name
