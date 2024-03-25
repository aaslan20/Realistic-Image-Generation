import numpy as np
import scipy
from scipy.stats import gaussian_kde
import torch 



# get activations from chosen layer
def activations(train_data, classifier, layer_dim, sa_layer, treshhold, num_act, device):
            all_ats = torch.zeros(0, layer_dim)
            for i, (x, x_class) in enumerate(train_data):
                x = x.to(device)
                ats = classifier(x, sa_layer).detach()
                all_ats = torch.cat([all_ats, ats], dim = 0)
            all_ats = torch.transpose(all_ats, 0, 1)
            rem_cols = torch.std(all_ats, axis=1) < treshhold
            ref_all_ats = all_ats[~rem_cols]
            ref_all_ats = ref_all_ats[:num_act]
            ref_all_ats_np = ref_all_ats.cpu().numpy()
            kde = gaussian_kde(ref_all_ats_np)
            return kde, rem_cols


# calculate lsa with kde from activations
def calc_lsa(at, kde):
    return -kde.logpdf(at)


def calc_img_lsa(img, classifier, sa_layer, rem_cols, kde):
    pr_at = classifier(img, sa_layer).detach()
    pr_at = pr_at.cpu().numpy().transpose()
    pr_at = pr_at[~rem_cols][:100]
    return calc_lsa(pr_at, kde)

"""
# get activations for image and then calculate it's lsa (alternative)
def calc_img_lsa2(img, classifier, sa_layer, rem_cols, kde):
    pr_at = classifier(img, sa_layer).detach().cpu()
    pr_at = pr_at.permute(1,0)
    pr_at = pr_at[~rem_cols][:100]
    return calc_lsa(pr_at, kde) """



# Calculate the Distance based suprise Adequacy which is an alternativ to lsa
def calculate_dsa(x, classifier, sa_layer, cx, train_data):
    # activation of the input
    at_x = classifier(x, sa_layer).detach().numpy()
    xa = None

    # get activation of a sample with the same label, calculate the distance and assign it as xa
    min_distance_same_class = float('inf')
    for batch_data, batch_labels in train_data:
        for xi, x_class in zip(batch_data, batch_labels):
            if x_class == cx:
                at_xi = classifier(xi, sa_layer).detach().numpy()
                distance = np.linalg.norm(at_x - at_xi)
                if distance < min_distance_same_class:
                    min_distance_same_class = distance
                    xa = at_xi

    
    # get activation of a sample with a different label, calculate the distance and assign it as xb        
    xb = None

    min_distance_other_class = float('inf')
    for batch_data, batch_labels in train_data:
        for xi, x_class in zip(batch_data, batch_labels):
            if x_class != cx:
                at_xi = classifier(xi, sa_layer).detach().numpy()
                distance = np.linalg.norm(xa - at_xi)
                if distance < min_distance_other_class:
                    min_distance_other_class = distance
                    xb = at_xi
    
    # Calculations
    dista = np.linalg.norm(at_x - xa)
    distb = np.linalg.norm(xa - xb)
    dsa_value = dista / distb
    
    return dsa_value



""""
# alternativ function for activations not used !
def activations2(train_data_loader, model, layer_dim, threshold, num_act):
    all_ats = []
    for i, (x, x_class) in enumerate(train_data_loader):
        x = x.cuda()
        ats = model.at_by_layer(x, layer_dim).detach().cpu().numpy()
        all_ats.append(ats)  # oder doch lieber torch cat damit ein tensor statt x tensors
    all_ats = torch.tensor(all_ats).view(-1, layer_dim).numpy()
    rem_cols = torch.std(all_ats, axis=0) < threshold
    ref_all_ats = all_ats[:, ~rem_cols.numpy()]
    ref_all_ats = ref_all_ats[:num_act]
    kde = gaussian_kde(ref_all_ats)
    return kde, rem_cols
"""


