########################################
# Class SmoothedCrossEntropy
########################################

import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt


class SmoothedCrossEntropy(torch.nn.Module):
    def __init__(self, logits=True, smooth_factor=0.0, smooth_prior='uniform', reduce_batch=True, 
                 is_target_oh=False, device=None, reduce_time=None, num_classes=106, action_embeddings_csv_path=None, actions_weights=None):
        super(SmoothedCrossEntropy, self).__init__()
        self.logits = logits
        self.smooth_factor = smooth_factor
        self.smooth_prior = smooth_prior
        self.reduce_batch = reduce_batch
        self.is_target_oh = is_target_oh
        self.device = device
        self.reduce_time = reduce_time
        self.num_classes = num_classes

        ############################
        # Need to modify this
        ############################
        self.action_embeddings_csv_path = action_embeddings_csv_path  # sarebbe il phi
        self.prior_matrix = self.get_prior()
        ############################
        ############################

        self.actions_weights = torch.tensor(actions_weights, dtype=torch.float32, device=self.device) if actions_weights is not None else None
        
    def get_prior(self):
        '''
        This function load the prior matrix
        '''
        def str2float_np(e):  #trasforma ogni riga in float e la concatena
            #print(e)
            #e = [val for val in e[1:].split(',')]
            e = e[1:]
            if self.smooth_prior !="verb-noun":
                e = [float(val) for val in e if val != 0]
            else:
                e = [float(val) for val in e]
            e = np.array(e, dtype=np.float32)
            return e

        
        # Compute prior
        prior = None
        if self.smooth_prior == 'uniform':
            prior = np.ones([self.num_classes, self.num_classes], dtype=np.float32) / self.num_classes
        elif self.smooth_prior == 'glove':
    
            embeddings = pandas.read_csv(self.action_embeddings_csv_path).values.tolist()
            #print(np.shape(embeddings))
            emb = []
            for index, row in enumerate(embeddings):
                riga = str2float_np(row)
                emb.append(riga)

            # Action embedings
            act_emb = []
            for e in emb:  # e è una riga della phi
                act_emb += [e]
            act_emb = np.array(act_emb)

            # Compute prior
            act_sim = np.absolute(act_emb.dot(act_emb.T))
            act_sim = act_sim / act_sim.sum(axis=-1, keepdims=True)
            prior = act_sim
            # uncomment to show the heatmap
            #plt.imshow(prior, cmap='hot', interpolation='nearest')
            #plt.show()
        elif self.smooth_prior == 'glove-soft':  # temp>0 softmax

            embeddings = pandas.read_csv(self.action_embeddings_csv_path).values.tolist()
            # print(np.shape(embeddings))
            emb = []
            for index, row in enumerate(embeddings):
                riga = str2float_np(row)
                emb.append(riga)

            # Action embedings
            act_emb = []
            for e in emb:  # e è una riga della phi
                act_emb += [e]
            act_emb = np.array(act_emb)

            # Compute prior
            temp = 1.0
            act_sim = act_emb.dot(act_emb.T)
            act_sim = np.exp(temp * act_sim) / np.exp(temp * act_sim).sum(axis=-1, keepdims=True)
            prior = act_sim

        elif self.smooth_prior == 'verb-noun':  # verb noun ls load prior
            embeddings = pandas.read_csv(self.action_embeddings_csv_path).values.tolist()
            prior = []

            for index, row in enumerate(embeddings):
                riga = str2float_np(row)
                prior.append(riga)

            prior = np.array(prior)


        else:
            raise Exception(f'Label smoothing {self.smooth_prior} not supported.')
        return torch.tensor(prior, dtype=torch.float32, device=self.device) 
        
    def one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        if self.device is not None:
            y_one_hot = y_one_hot.to(self.device)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot
        
    def forward(self, y_pred, y_true, mask=None, eps=1e-7, weight=None):
        '''
        input y_pred: predictions of shape [bs, time, num_classes]
        input y_true: predictions of shape [bs, time]
        input mask: mask fo shape [bs, time]
        input eps: parameter for clipping the loss
        input weight: weight for the loss
        output xent: scalar
        '''
            
        # Convert to torch tensors, if needed
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true, dtype=torch.float32)
            
        # Convert target into one-hot vector, if needed
        if not self.is_target_oh:
            y_true = self.one_hot(y_true, y_pred.shape[-1])
            
        # Cast to float32, if needed
        if y_pred.dtype != torch.float32:
            y_pred = y_pred.to(torch.float32)
        if y_true.dtype != torch.float32:
            y_true = y_true.to(torch.float32)
            
        # Logits to probabilities, if needed
        if self.logits:
            y_pred = torch.nn.functional.softmax(y_pred, -1)
            
        # Clip prediction for numerical stability
        y_pred = torch.clamp(y_pred, min=eps, max=1.0 - eps)

        prior = self.prior_matrix[torch.argmax(y_true.view(-1, self.num_classes), -1), :]  # cioè prendo la riga corrispondente all'etichetta dove c'è 1
        #print(prior.size())
        prior = prior.view(*y_pred.shape)
        if self.smooth_factor > 0.0:
            y_true = (1.0 - self.smooth_factor) * y_true + self.smooth_factor * prior
            
        # Weight
        if self.actions_weights is not None:
            weights = self.actions_weights[torch.argmax(y_true.view(-1, self.num_classes), -1)]
            weights = weights.view(*[*y_pred.shape[:-1], 1])
            weights = weights.repeat( *[*[1 for _ in range(len(weights.shape[:-1]))], self.num_classes] )#weights.repeat(*[*weights.shape[:-1], self.num_classes])
            y_true = y_true * weights
                
        # Compute cross-entropy
        #xent = - y_true * torch.log(y_pred) - (1.0 - y_true) * torch.log(1.0 - y_pred) # [batch_size(, time), num_classes]  TODO:???
        xent = - y_true * torch.log(y_pred)


        # Mask for sequential data of shape [batch_size, time, num_classes]
        if self.reduce_time is not None:
            if mask is not None:
                seq_len = torch.sum(mask, -1)
                xent = torch.where(mask > 0, xent, torch.zeros_like(xent))
                xent = torch.sum(xent, 1) / seq_len  # [batch_size, num_classes]
            else:
                if self.reduce_time == 'mean':
                    xent = torch.mean(xent, 1)  # [batch_size, num_classes]
                if self.reduce_time == 'sum':
                    xent = torch.sum(xent, 1)  # [batch_size, num_classes]
                if 'idx' in self.reduce_time:
                    idx_t = int(self.reduce_time.replace('idx=', ''))
                    xent = xent[:, idx_t, :]  # [batch_size, num_classes]
        xent = torch.sum(xent, -1)  # [batch_size]

        if self.reduce_batch:
            
            # Remove nan values
            xent = torch.where(torch.isnan(xent), torch.zeros_like(xent, device=xent.device), xent)
            xent = torch.mean(xent, 0)  # scalar
        return xent  # This is a scalar
