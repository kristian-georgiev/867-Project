import torch

class meta_learner():
    def __init__(self, name):
        if name == "maml":
            self.algorithm = self.init_maml()
        elif name == "reptile":
            self.algorithm = self.init_reptile()
        elif name == "anil":
            self.algorithm = self.init_anil()


    def init_maml(self):
        pass

    def init_reptile(self):
        pass

    def init_anil(self):
        pass

    def train(self, data, save_gradient_steps):
        pass

