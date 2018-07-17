import numpy as np


class Node:    
    def __init__(self, data, data_idx, target, build = 1):
        self.data = data
        if build:
            self.add_node(data, data_idx, target)
        else:
            self.predict(data)


    def predict(self, data):
        if 

        
    def add_node(self, node_data, node_idx, target):
        #find split
        #loop through features (columns)
        print(node_data)
        max_gini = -1
        for feat in range(len(node_data)):
            split_vals = np.linspace(min(node_data[feat]), max(node_data[feat]), 100)
            for split_val in split_vals:
                ldata = []
                rdata = []
                lidx = []
                ridx = []
                for cx, x in enumerate(node_data[feat]):
                    if x < split_val:
                        ldata.append(x)
                        lidx.append(node_idx[cx])
                    else:
                        rdata.append(x)
                        ridx.append(node_idx[cx])
                gini = self.calc_gini(target, node_data[feat], node_idx, ldata, lidx, rdata, ridx)
                print(split_val, 'g', gini)
                if gini > max_gini:
                    max_gini = gini
                    best_feat = feat
                    best_split = split_val
        self.feature = best_feat
        self.split_val = best_split
        print(max_gini, best_feat, best_split)
        ldata = []
        lidx = []
        rdata = []
        ridx = []
        for cx,x in enumerate(node_data[best_feat]):
            if x < best_split:
                ldata.append(x)
                lidx.append(node_idx[cx])
            else:
                rdata.append(x)
                ridx.append(node_idx[cx])
        #assign left and right nodes (recursively? recursively)
        print(ldata, rdata)
        if max_gini > 0:
            self.left  = Node([ldata], lidx, target)
        else:
            self.left = None
        if max_gini > 0:
            self.right = Node([rdata], ridx, target)
        else:
            self.right = None
        


    def calc_gini(self, targets, node_data, node_idx, ldata, lidx, rdata, ridx):
        #calc node gini
        n_classes = len(set(targets))
        gini = self.calc_impurity(node_data, node_idx, n_classes)
        gini -= len(ldata)/len(node_data)*self.calc_impurity(ldata, lidx, n_classes)
        gini -= len(rdata)/len(node_data)*self.calc_impurity(rdata, ridx, n_classes)
        return gini


    def calc_impurity(self, node_data, node_idx, n_classes):
        impurity = 1
        if len(node_data) == 0:
            return 0
        #class partition
        nclass = [0 for _ in range(n_classes)]
        for idx in node_idx:
            nclass[targets[idx]] += 1
        for c in nclass:
            impurity -= (c/len(node_idx))**2
        #print(impurity)
        return impurity

#fake data
targets = [1,1,1,1,0,0,0,0]
features = [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]
feature_idx = list(range(len(targets)))

#head node
head = Node(features, feature_idx, targets)
print(head.split_val, head.feature, head.data, head.left.data, head.right.data)
head.predict([[0]])
