import os
import numpy as np
import matplotlib.pyplot as plt
import source.utils as utils

def training(agent, dataset, batch_size, epochs, c_update, s_avg):

    print("\n** Training to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0
    loss_best = 1e+12
    for epoch in range(epochs):

        list_loss = []
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            list_loss.append(step_dict['losses']['smce'])
            iteration += 1
            if(minibatch['terminate']): break
        list_loss = np.asarray(list_loss)
        loss_tmp = np.average(list_loss)

        if(loss_best > loss_tmp):
            loss_best = loss_tmp
            agent.save_params(model='model_1_best_loss')
        agent.save_params(model='model_0_finepocch')

        if(epoch % c_update == 0):
            agent.swag(num_model=epoch/c_update, training=True)
            agent.swag(num_sample=s_avg)

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, loss_tmp))
