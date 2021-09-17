import os
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as wbl
import whiteboxlayer.extensions.utility as wblu

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.ksize = kwargs['ksize']
        self.k_max = kwargs['k_max']

        filters = kwargs['filters']
        self.filters = [self.dim_c]
        try:
            tmp_filters = filters.split(',')
            for idx, _ in enumerate(tmp_filters):
                tmp_filters[idx] = int(tmp_filters[idx])
            self.filters.extend(tmp_filters)
        except: self.filters.extend([16, 32, 64, 64])

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        self.__model = Neuralnet(dim_h=self.dim_h, dim_w=self.dim_w, dim_c=self.dim_c, num_class=self.num_class, ksize=self.ksize, filters=self.filters)
        self.__model.forward(x=tf.zeros((1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32), verbose=True)
        print("\nNum Parameter: %d" %(self.__model.layer.num_params))

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        self.list_pkey = list(self.__model.layer.parameters.keys())
        for key in self.list_pkey:
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

        conc_func = self.__model.__call__.get_concrete_function(tf.TensorSpec(shape=(1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32))

        # initialize moments
        theta = wblu.get_allweight(self.__model)
        self.theta_1 = theta
        self.theta_2 = theta**2
        self.theta_d = [theta - self.theta_1]
        self.theta_bank = [[self.theta_1, self.theta_2 - theta**2, self.theta_d]]

    def __loss(self, y, y_hat):

        smce_b = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        smce = tf.math.reduce_mean(smce_b)

        return {'smce_b': smce_b, 'smce': smce}

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['y']

        with tf.GradientTape() as tape:
            outputs = self.__model.forward(x=x, verbose=False)
            y_hat = outputs['y_hat']
            losses = self.__loss(y=y, y_hat=y_hat)

        if(training):
            gradients = tape.gradient(losses['smce'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/smce' %(self.__model.who_am_i), losses['smce'], step=iteration)

        return {'y_hat':tf.nn.softmax(y_hat).numpy(), 'losses':losses}

    def get_thetadict(self):

        return {'theta_1':self.theta_1, 'theta_2':self.theta_2, 'theta_d':self.theta_d, 'theta_bank':self.theta_bank}

    def swag(self, num_model=-1, num_sample=-1, training=False):

        if(training):
            # update moments
            theta = wblu.get_allweight(self.__model)
            self.theta_1 = (num_model*self.theta_1 + theta) / (num_model + 1)
            self.theta_2 = (num_model*(self.theta_2**2) + theta**2) / (num_model + 1)
            self.theta_d.append(theta - self.theta_1)
            self.theta_d[-self.k_max:] # self.theta_d x self.theta_d.T: covariance matrix
            self.theta_bank.append([self.theta_1, self.theta_2 - self.theta_1**2, self.theta_d]) # [swa, diagonal, d]

        else:
            self.save_params(model='backup')
            for idx_sample in range(min(num_sample, len(self.theta_bank))):
                tmp_sample = utils.bayesian_sampling(theta_bank=self.theta_bank[idx_sample])

                self.__model = wblu.set_allweight(self.__model, new_weight=tmp_sample)

                self.save_params(model='model_history_%d' %(idx_sample))
            self.load_params(model='backup')

    def save_params(self, model='base'):

        vars_to_save = self.__model.layer.parameters.copy()
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
        ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = "CNN_Basic"
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.ksize = kwargs['ksize']
        self.filters = kwargs['filters']

        self.filters = kwargs['filters']

        self.layer = wbl.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        y_hat = self.__classifier(x=x, name='clf', verbose=verbose)
        y_hat = tf.add(y_hat, 0, name="y_hat")

        return {'y_hat':y_hat}

    def __classifier(self, x, name='enc', verbose=True):

        if(verbose): print("\n* Classifier")

        for idx, _ in enumerate(self.filters):
            if(idx == 0): continue
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters[idx-1], self.filters[idx]], \
                activation='elu', name='%s-%d_c0' %(name, idx), verbose=verbose)

            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters[idx], self.filters[idx]], \
                activation='elu', name='%s-%d_c1' %(name, idx), verbose=verbose)
            x = self.layer.maxpool(x=x, ksize=2, strides=2, \
                name='%s-%d_mp' %(name, idx), verbose=verbose)

        [n, h, w, c] = x.shape
        x = tf.compat.v1.reshape(x, shape=[n, h*w*c], name="flat")
        x = self.layer.fully_connected(x=x, c_out=512, \
                batch_norm=False, activation='elu', name="%s-z0" %(name), verbose=verbose)
        x = self.layer.fully_connected(x=x, c_out=self.num_class, \
                batch_norm=False, activation=None, name="%s-z1" %(name), verbose=verbose)

        return x
