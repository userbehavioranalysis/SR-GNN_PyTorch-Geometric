# -*- coding: utf-8 -*-
"""
Created on 31/3/2019
@author: RuihongQiu
"""

import pickle
import torch
from torch_geometric.data import InMemoryDataset, Data


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        
        for sequences, y in zip(data[0], data[1]):
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            del senders[-1]    # the last item is a receiver
            del receivers[0]    # the first item is a sender
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
