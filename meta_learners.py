import torch
import meta_ops

class meta_learner():
    def __init__(self, hparams):
        self.name = hparams.meta_learner
        if self.name == "maml":
            self.algorithm = self.init_maml()
            # TODO: replace this
            self.train = meta_ops.train_maml
            self.test  = meta_ops.test_maml

        elif self.name == "reptile":
            self.algorithm = self.init_reptile()

        elif self.name == "anil":
            self.algorithm = self.init_anil()
            # TODO: replace this
            self.train = meta_ops.train_anil
            self.test  = meta_ops.test_maml


    def init_maml(self):
        pass

    def init_reptile(self):
        pass

    def init_anil(self):
        pass

    def get_train_and_test(self):
        return self.train, self.test

    def toString(self):
        return self.name
