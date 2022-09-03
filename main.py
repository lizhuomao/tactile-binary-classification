import os.path

import resnet
import torch
import dataManipulation as dm

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.05, 10, 256
    net = resnet.net
    train_iter, test_iter = dm.load_data(0.2, need_split = False)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
        break
    resnet.train(net, train_iter, test_iter, num_epochs, lr, try_gpu())
    if os.path.exists('./weight/'):
        pass
    else :
        os.makedirs('./weight/')
    torch.save(net.state_dict(), './weight/resnet.params')




