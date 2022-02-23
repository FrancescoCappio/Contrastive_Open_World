import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

class PrototypeSet(nn.Module):

    def __init__(self, number, dimensions, data=None):
        super().__init__()

        if data is None: 
            # we start from random points
            prototypes = torch.randn((number,dimensions))
        else:
            # we start from data 
            prototypes = data
        self.weights = nn.Parameter(data=prototypes, requires_grad=True)

    def forward(self):
        # first we should move points on the hypersphere
        weights = self.weights/self.weights.norm(dim=1, keepdim=True)

        similarity_matrix = torch.matmul(weights, weights.T) + 1
        eye = torch.eye(similarity_matrix.shape[0]).to(self.weights.device)
        similarity_matrix -= 2*eye
        return similarity_matrix.max(dim=1)[0].max()

    @torch.no_grad()
    def get_prototypes(self):
        weights = self.weights/self.weights.norm(dim=1, keepdim=True)
        return weights.detach().clone()

def generate_prototypes(hyp_dimensions=128, n_prototypes=100, iterations=20000, data=None):

    model = PrototypeSet(n_prototypes, hyp_dimensions, data=data)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    #print(f"Generating {n_prototypes} prototypes on an hypersphere with {hyp_dimensions} dimensions")
    # use a queue to track losses
    from queue import Queue
    last_x = 100
    loss_queue = Queue(maxsize=last_x)
    conv_threshold = 0.0001
    enable_convergence_break = False
    for idx in tqdm(range(iterations)):
        loss = model.forward()
        if enable_convergence_break:
            loss_queue.put(loss)

            if idx >= last_x -1:
                old_loss = loss_queue.get()
                if old_loss - loss < conv_threshold:
                    print(f"Reached prototypes convergence. Old loss: {old_loss}, loss: {loss}")
                    break
        #print("Iter {:10d}. Loss: {:.5f}".format(idx, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    prototypes = model.get_prototypes()

    # analyze generated prototypes
    def rescale_cosine_similarity(similarities):
        return (similarities+1)/2

    prototypes = prototypes.numpy()
    def compute_top_sims(prototypes):
        top_sims = np.zeros(len(prototypes))
        for idx, prt in enumerate(prototypes):
            similarities = (prt*prototypes).sum(axis=1)
            similarities = rescale_cosine_similarity(similarities)
            similarities[similarities.argmax()] = -1
            top_sims[idx] = similarities.max()
        return top_sims

    top_sims = compute_top_sims(prototypes)
    print(f"Prototypes generated. Mean similarity: {top_sims.mean()}, Std: {top_sims.std()}")

    return prototypes

