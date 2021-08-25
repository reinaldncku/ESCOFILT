import sys
import torch
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from summarizer.model_processors import Summarizer


''' #########################################################
#
# The CF/recsys architecture
#
######################################################### '''

class ESCOFILT(torch.nn.Module):
     
    def __init__(self, users, items, num_layers, emb_size, len_users, len_items, reduce_dim):
        super().__init__()

        self.ALL_user_embeddings = users
        self.ALL_item_embeddings = items   
        self.reduce_dim = reduce_dim  
        
        self.trad_user_embeddings = torch.nn.Embedding(len_users, reduce_dim)
        self.trad_item_embeddings = torch.nn.Embedding(len_items, reduce_dim)

        self.compress_u = torch.nn.Linear(emb_size, reduce_dim)
        self.compress_i = torch.nn.Linear(emb_size, reduce_dim)
        
        self.mlp = torch.nn.Sequential()

        ctr = 0  
        curr_in = reduce_dim * 2
        for ctr in range(num_layers):
            self.mlp.add_module("mlp"+str(ctr),torch.nn.Linear(curr_in, int(curr_in/2)))
            self.mlp.add_module("relu"+str(ctr), torch.nn.ReLU()) 
            curr_in = int(curr_in/2)
                
        self.mlp.add_module("last_dense",  torch.nn.Linear(curr_in, 1))
        #self.mlp.add_module("last_relu", torch.nn.ReLU()) 
        self.dropper = torch.nn.Dropout(0.5) # Dont forget!!!! 0.5 default

        
    def forward(self, us, it):
        
        emp_u = []
        emp_i = []
        
        emp_u = self.ALL_user_embeddings(us)
        emp_i = self.ALL_item_embeddings(it)
        
        emp_u = self.compress_u(emp_u)
        emp_i = self.compress_i(emp_i)
        
        trad_u = self.trad_user_embeddings(us)
        trad_i = self.trad_item_embeddings(it)
        
        emp_u += trad_u
        emp_i += trad_i
        
        cat_features = torch.cat((emp_u, emp_i), 1)
        cat_features = self.dropper(cat_features)
        out = self.mlp(cat_features)
         
        return out

#########################


''' #########################################################
#
# Other utilities/tools
#
######################################################### '''
    
            
def count_parameters(model):

    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_xs_bert_embeddings(tvt_data, ui_ids, key, ratio):

    summer = Summarizer()   
    agg_embeddings = {}

    with tqdm(total=len(ui_ids)) as pbar:
        for uiid in ui_ids:
            pbar.update(1)

            ui_reviews = [d["reviews"] for d in tvt_data if str(d[key]) == str(uiid)]
            #space = ". "
            space = " "
            body  = space.join(ui_reviews)
            
            try:
                ui_embedding = summer.run_embeddings(body, ratio=ratio, aggregate='mean', min_length=10, max_length=800)

                if (ui_embedding is None):
                    print ("Init. NaN ", uiid, " <<<<< ")
                    ui_embedding = summer.run_embeddings(body, ratio=ratio, aggregate='mean', min_length=10, max_length=1900)

                    if (ui_embedding is None):
                        print ("Still, NaN-affected ID: ", uiid, " <<<<< ")
            except:
                print ("Offending ID (via Exception): ", uiid)
                print ("Bye-bye for now!")
                sys.exit()                

            agg_embeddings[str(uiid)] = ui_embedding
    
    
    return agg_embeddings



def get_nn_embeddings(user_embeddings):
  
    u_keys = user_embeddings.keys()
    u_keys = sorted(u_keys)
    u_len = len(user_embeddings)
    user_pretrained_wts = [torch.zeros(1024)] * u_len
    user_embedding = torch.zeros(1024)

    for u in u_keys:
        try:
            user_embedding = torch.from_numpy(user_embeddings[u])
        except:
            print (u)

        user_pretrained_wts[int(u)] = user_embedding

    user_pretrained_wts = torch.stack(user_pretrained_wts).cuda()
    user_nn_embeddings = torch.nn.Embedding.from_pretrained(user_pretrained_wts).cuda()

    return user_nn_embeddings



def acquire_dataloader(df, b_size, to_random=False):

    # Preparing the tensors and iterators
    tensor_users = torch.tensor(df["user_id"].tolist(), dtype=torch.long)
    tensor_items = torch.tensor(df["item_id"].tolist(), dtype=torch.long)
    tensor_ratings = torch.tensor(df["ratings"].tolist(), dtype=torch.float)

    data = TensorDataset(tensor_users, tensor_items, tensor_ratings)
    
    if to_random:
        sampler = RandomSampler(data)
    else:
        sampler =  SequentialSampler(data)
    
    dataloader = DataLoader(data, sampler=sampler, batch_size=b_size)

    return dataloader


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
