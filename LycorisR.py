"""Copyright information.
Copyright (c) 2020, RootHarold
All rights reserved.
Use of this source code is governed by a LGPL-3.0 license that can be found
in the LICENSE file.
"""

from LycorisNet import Lycoris
from LycorisNet import loadModel
import math
import random
import numpy as np
import logging
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class Recommender:
    """A lightweight recommendation algorithm framework based on LycorisNet.

    Attributes:
        __config: Store configuration information, including 11 configuration fields.
        __lie: The neural network based on LycorisNet.
        __mapping: Store the mapping between one-hot encoding and embedding vectors.
        __count: The counter for controlling "enrich()" of LycorisNet.
    """

    def __init__(self, config):
        """Initialization

        :param config: The configuration information, including 11 configuration fields. More details can be found in
                       the relevant documentation.
        """

        if config is not None:
            self.__check_config(config)
            self.__config = config
            self.__lie = Lycoris(capacity=config["capacity"], inputDim=config["dimension"] * (config["sequence"] - 1),
                                 outputDim=config["dimension"], mode="classify")
            self.__lie.setMutateOdds(0)
            self.__lie.preheat(config["nodes"], config["connections"], config["depths"])
            self.__mapping = {}
            self.__count = 0

    def embed(self, data):
        """Generate the mapping between one-hot encoding and embedding vectors.

        :param data: Each dimension of data is spliced by several one-hot codes. For example, the encoding of "A B-> C:"
                     can be expressed as: [0, 0, 1, 0, 1, 0, 1, 0, 0].
        :return: None.
        """

        if np.array(data).ndim == 1:
            data = [data]

        batch = math.ceil(len(data) / float(self.__config["batch_size"]))
        remainder = len(data) % self.__config["batch_size"]
        input_dim = self.__config["dimension"] * (self.__config["sequence"] - 1)

        data_copy = list(data).copy()
        if remainder > 0:
            data_copy.extend(random.choices(data, k=(self.__config["batch_size"] - remainder)))

        for i in range(self.__config["epoch"]):
            random.shuffle(data_copy)
            temp1 = [None] * self.__config["batch_size"]
            temp2 = [None] * self.__config["batch_size"]
            pos = 0

            for _ in range(batch):
                for k in range(self.__config["batch_size"]):
                    temp1[k] = data_copy[pos][:input_dim]
                    temp2[k] = data_copy[pos][input_dim:]
                    pos = pos + 1

                if self.__count == self.__config["evolution"]:
                    self.__lie.enrich()

                if self.__count < self.__config["evolution"]:
                    self.__lie.fitAll(temp1, temp2)
                else:
                    self.__lie.fit(temp1, temp2)

                self.__count = self.__count + 1

            if self.__config["verbose"]:
                logging.info("Epoch " + str(i + 1) + " : " + str(self.__lie.getLoss()))

        for item in data:
            self.__lie.compute(item[:input_dim])
            self.__mapping[tuple(item[input_dim:])] = self.__lie.getHiddenLayer(self.__config["middle_layer"])

    def most_similarity(self, positive, negative=None, top_n=10):
        """Calculate and get the most similar top_n items.

        :param positive: List of items that contribute positively.
        :param negative: List of items that contribute negatively.
        :param top_n: Top N.
        :return: The most similar top_n items and their similarities.
        """

        if negative is None:
            negative = []
        else:
            if np.array(negative).ndim == 1:
                negative = [negative]

        if np.array(positive).ndim == 1:
            positive = [positive]

        keys = list(self.__mapping.keys())
        positive_middle = [self.__mapping[tuple(item)] for item in positive]
        negative_middle = [self.__mapping[tuple(item)] for item in negative]

        similarities = []
        for item in keys:
            temp = self.__mapping[item]
            score = 0.0

            for i in positive_middle:
                score = score + self.__cos_sim(temp, i)

            for i in negative_middle:
                score = score - self.__cos_sim(temp, i)

            similarities.append(score)

        indexes = np.argsort(-np.array(similarities))
        length = top_n if top_n < len(similarities) else len(similarities)

        ret = []
        for i in range(length):
            ret.append([keys[indexes[i]], similarities[indexes[i]]])

        return ret

    def infer(self, items):
        """Inferring based on the trained model.

        :param items: Each dimension of data is spliced by several one-hot codes.
        :return: One-hot codes of desired items.
        """

        if np.array(items).ndim == 1:
            items = [items]

        ret = []
        items_ = self.__lie.computeBatch(items)
        for item in items_:
            temp = [0] * self.__config["dimension"]
            temp[int(np.argmax(np.array(item)))] = 1
            ret.append(temp)

        return ret

    def get_vector(self, items):
        """Get the embedding vectors of items.

        :param items: one-hot codes.
        :return: Embedding vectors of items.
        """

        if np.array(items).ndim == 1:
            items = [items]

        ret = []
        for item in items:
            ret.append(self.__mapping[tuple(item)])

        return ret

    def save(self, path1, path2):
        """Save the model and related configurations.

        :param path1: The path to store the model.
        :param path2: The path to store the configurations.
        :return: None.
        """

        self.__lie.saveModel(path=path1)
        config_copy = self.__config.copy()
        config_copy["mapping"] = {}
        for key, value in self.__mapping.items():
            config_copy["mapping"][str(key)] = value
        json_info = json.dumps(config_copy, indent=4)
        f = open(path2, 'w')
        f.write(json_info)
        f.close()

        if self.__config["verbose"]:
            logging.info("Model saved successfully.")

    @staticmethod
    def load(path1, path2):
        """Import pre-trained models and related configurations.

        :param path1: The path to import the model.
        :param path2: The path to import the configurations.
        :return: None.
        """

        l_r = Recommender(None)
        l_r.__count = 0

        l_r.__lie = loadModel(path1, capacity=1)

        f = open(path2, 'r')
        json_info = f.read()
        f.close()

        config = json.loads(json_info)
        l_r.__mapping = {}
        for key, item in config["mapping"].items():
            l_r.__mapping[eval(key)] = item
        config.pop("mapping")
        config["capacity"] = 1
        config["evolution"] = 0
        l_r.__check_config(config)
        l_r.__config = config
        if l_r.__config["verbose"]:
            logging.info("Model imported successfully.")

        return l_r

    def set_config(self, config):
        """Set the configuration information of Recommender.

        :param config: The configuration information, including 11 configuration fields.
        :return: None.
        """

        self.__check_config(config)
        self.__config = config

    def set_lr(self, learning_rate):
        """Set the learning rate of the neural network.

        :param learning_rate: The learning rate of the neural network.
        :return: None.
        """

        self.__lie.setLR(learning_rate)

    def set_workers(self, workers):
        """Set the number of worker threads to train the model.

        :param workers: The number of worker threads.
        :return: None.
        """

        self.__lie.setCpuCores(num=workers)

    @staticmethod
    def version():
        """Returns the version information of Recommender."""

        lycoris_version = Lycoris.version()
        return "LycorisR 1.5.3 By RootHarold." + "\nPowered By " + lycoris_version[:-15] + "."

    @staticmethod
    def __check_config(config):
        """Check whether the configuration information is valid.

        :param config: The configuration information, including 11 configuration fields.
        :return: None.
        """

        keys = ["capacity", "dimension", "sequence", "nodes", "connections", "depths", "batch_size", "epoch",
                "middle_layer"]
        for item in keys:
            if item not in config:
                raise Exception("Invalid configuration.")

        if "evolution" not in config:
            config["evolution"] = 0

        if "verbose" not in config:
            config["verbose"] = False

    @staticmethod
    def __cos_sim(a, b):
        """Computes the cosine similarity between two vectors."""

        a = np.mat(a)
        b = np.mat(b)
        num = float(a * b.T)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim
