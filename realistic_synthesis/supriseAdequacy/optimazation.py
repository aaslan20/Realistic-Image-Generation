import torch
import math
import random 

def hill_climbing(opt_epoch_num,opt_z, z_dim, prev_sa, target_sa, prev_loss, loss, opt_img, last_diff_prop, suprise):
    for e_idx in range(opt_epoch_num):
        last_epoch_z = opt_z
        update_num = 0
        print(f'-----epoch {e_idx} start-----')
        for i in range(z_dim):
            z_copy = opt_z.clone()
            z_copy[:, i] = z_copy[:, i] + last_diff_prop*torch.randn(1)
            new_sa = suprise(z_copy)
            new_loss = loss(new_sa, obj=target_sa)
            if prev_loss > new_loss:
                opt_z = z_copy
                prev_loss = new_loss
                prev_sa = new_sa
                update_num += 1
                print('\r'*100, end='')
                print(f'new_sa: {new_sa[0]:.4f}', end='')
            if prev_loss < 0.01: break
        print(f'\n# of updates: {update_num}')
        print(f'change though opt: {torch.sum(torch.abs(last_epoch_z - opt_z))}')
        last_diff_prop = max(torch.sum(torch.abs(last_epoch_z - opt_z))/update_num, update_num/z_dim)
        if (update_num == 0 or prev_loss < 0.01): break


def simulated_annealing(opt_epoch_num, opt_z, z_dim, prev_sa, target_sa, prev_loss, loss, opt_img, last_diff_prop, surprise):
    for e_idx in range(opt_epoch_num):
        last_epoch_z = opt_z.clone()
        update_num = 0
        print(f'-----epoch {e_idx} start-----')
        
        temperature = 1.0  
        cooling_rate = 0.95  

        for i in range(z_dim):
            z_copy = opt_z.clone()
            z_copy[:, i] = z_copy[:, i] + last_diff_prop * torch.randn(1)
            new_sa = surprise(z_copy)
            new_loss = loss(new_sa, obj=target_sa)
            
            
            energy_delta = new_loss - prev_loss
            energy_delta = energy_delta.item() if isinstance(energy_delta, torch.Tensor) else energy_delta
            
            if energy_delta < 0 or random.random() < math.exp(-energy_delta / temperature):
                opt_z = z_copy
                prev_loss = new_loss
                prev_sa = new_sa
                update_num += 1
                print('\r'*100, end='')
                print(f'new_sa: {new_sa[0]:.4f}', end='')

            if prev_loss < 0.01: 
                break

        print(f'\n# of updates: {update_num}')
        print(f'change though opt: {torch.sum(torch.abs(last_epoch_z - opt_z))}')

        temperature *= cooling_rate
        last_diff_prop = max(torch.sum(torch.abs(last_epoch_z - opt_z)) / update_num, update_num / z_dim)
        
        if update_num == 0 or prev_loss < 0.01:
            break