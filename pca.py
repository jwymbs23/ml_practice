#implement pca
import numpy as np
import matplotlib.pyplot as plt


def mean_shift_cols(features):
    shift_features = []
    for col in features.T:
        col_mean = np.mean(col)
        col_std = np.std(col)
        shift_col = (col - col_mean)#/col_std
        shift_features.append(shift_col)
    return np.asarray(shift_features).T



def eigen(features):
    e_vals, e_vecs = np.linalg.eig(np.matmul(features.T, features))
    return e_vals, e_vecs


def transform_coords(features, e_vals, e_vecs):
    #sort eigenvectors
    re_order = np.argsort(e_vals)[::-1]
    print(e_vecs, re_order)
    sort_e_vecs = e_vecs[:,re_order]
    print(sort_e_vecs)
    return np.matmul(features, sort_e_vecs)
    
    

#generate fake data:
#mean = [5, 10]
#cov = [[10, 100], [1,5]]#,0], [0, 0, 10]]  # diagonal covariance


#x, y = np.random.multivariate_normal(mean, cov, 1000).T

x = [5*np.random.random() for _ in range(1000)]
y = [2*np.random.random() for i in x]
z = [3*np.random.random() for i in range(1000)]
plt.plot(x, y, 'x')
plt.axis('equal')
#plt.show()


features = np.asarray([x,y,z]).T
#print(features)

#shift features
shift_features = mean_shift_cols(features)
x_shift = shift_features[:,0]
y_shift = shift_features[:,1]

print(eigen(shift_features))

#find eigenvalues of feature.T feature
e_vals, e_vecs = eigen(shift_features)

pca_feat = transform_coords(shift_features, e_vals, e_vecs)
print(np.matmul(pca_feat.T, pca_feat))

plt.plot(x_shift,y_shift,'o')
plt.plot(pca_feat[:,0], pca_feat[:,1], 'x')
plt.show()
