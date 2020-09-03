import os
from skimage import io
import numpy as np
import collections
from tqdm import tqdm
import argparse


class ImageRecovery:

    def __init__(self):
        self.open_files = None
        self.image_input = None
        self.image_size = None
        self.distance = list()
        self.pdf_image_in = None

    @staticmethod
    def pdf(histogram, image_size):
        return histogram / image_size

    def euclidean_distance(self, pdf_image_input, pdf_image_compare, name_image):
        result = 0

        for i in range(len(pdf_image_input)):
            result = result + (pdf_image_input[i] - pdf_image_compare[i]) ** 2

        self.distance.append({'image_name': name_image,
                              'euclidean_distance': result})

    @staticmethod
    def occurrence(channel):
        ocurrence_list = np.zeros([1, 256])
        for value in channel:
            ocurrence_list[0, value] = ocurrence_list[0, value] + 1
        return ocurrence_list

    def histogram(self, image):
        r_channel = image[:, :, 0].ravel()
        g_channel = image[:, :, 1].ravel()
        b_channel = image[:, :, 2].ravel()

        return self.occurrence(r_channel), self.occurrence(g_channel), self.occurrence(b_channel)

    def open_images(self, path):
        self.open_files = os.listdir(path)

    def recovery(self, path, files, samples=None):
        if samples is None:
            index = files.index(self.image_input)
            self.pdf_image_in = self.process_recovery(files, index, path)
            return

        for index in tqdm(range(len(files))):
            pdf_image_compare = self.process_recovery(files, index, path)
            self.euclidean_distance(self.pdf_image_in, pdf_image_compare, files[index])

        sorted_list = sorted(self.distance, key=lambda k: k['euclidean_distance'])

        return sorted_list[:int(samples)]

    def process_recovery(self, files, image, path):
        img = io.imread(path + files[image])
        self.image_size = img.shape[0] * img.shape[1]

        r_histogram, g_histogram, b_histogram = self.histogram(img)

        r_pdf = self.pdf(r_histogram, self.image_size)
        g_pdf = self.pdf(g_histogram, self.image_size)
        b_pdf = self.pdf(b_histogram, self.image_size)

        return np.array([[r_pdf.tolist()], [g_pdf.tolist()], [b_pdf.tolist()]]).flatten()

    @staticmethod
    def calculate_score(result_recovery):
        classes = [item['image_name'][:4] for item in result_recovery]
        return collections.Counter(classes)

    @staticmethod
    def print_results(result_recovery, result_score, samples):
        print(f"------------- Recovery -------------")
        for item in result_recovery:
            print("Imagem: %(image_name)-15s Distância Euclidiana: %(euclidean_distance)s" % item)

        print(f"------------- Score -------------")
        for class_, score in result_score.items():
            score_ = score * 100 / int(samples)
            print(f"Classe: {class_}       Ocorrência: {score}       %: {score_:0.02f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageIn', '-i', required=True, help='Imagem de entrada')
    parser.add_argument('--folderImages', '-fi', required=True, help='Pasta de imagens')
    parser.add_argument('--samples', '-s', required=True, help='Quantidade de amostras a ser retornada')

    args = parser.parse_args()
    pathGlobal = args.folderImages
    imageIn = args.imageIn
    samples = args.samples

    ir = ImageRecovery()

    print("Image: ", imageIn)

    ir.open_images(pathGlobal)
    ir.image_input = imageIn
    ir.recovery(pathGlobal, ir.open_files)
    result_recovery = ir.recovery(pathGlobal, ir.open_files, samples)
    result_score = ir.calculate_score(result_recovery)

    ir.print_results(result_recovery, result_score, samples)


if __name__ == '__main__':
    main()