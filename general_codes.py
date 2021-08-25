import torch
import math
from tqdm import tqdm


def train_NCF_only(cf_model, iterator, optimizer, cf_criterion, batch_size, device):
    epoch_rmse_loss = 0
    
    cf_model.train()
    
    for i, batch in enumerate(tqdm(iterator, desc="Training Iteration ")):  
        batch = tuple(t.to(device) for t in batch)
        user_id, item_id, ratings = batch
        
        predictions = cf_model(user_id, item_id)
        rmse_loss = cf_criterion(predictions.view(-1).float(), ratings.view(-1).float())  

        rmse_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_rmse_loss += rmse_loss.item()

        
    epoch_rmse_loss = (epoch_rmse_loss / len(iterator)) ** 0.5
    return epoch_rmse_loss

'''
####################################
'''

def evaluate_NCF_only(cf_model, iterator, batch_size, device, mode="Validation"):
    total_mse_loss = 0.0
    cf_model.eval()    

    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="{} Iteration ".format(mode))):    
            batch = tuple(t.to(device) for t in batch)
            user_id, item_id, ratings = batch

            predictions = cf_model(user_id, item_id)
       
            mse_loss = torch.sum((predictions.view(-1).float() - ratings.view(-1).float()) ** 2)
            total_mse_loss += mse_loss.item()

    data_len = len(iterator.dataset)
    rmse = math.sqrt((total_mse_loss * 1.0) / data_len)

    return rmse