SEED_CONST = 92687
EMB_TYPE = "exsumm_embedding"

###########
# Adjust these values for [0] Prepare ExSumm Emb.ipynb
PREP_DOMAIN = "video"
RATIOS = [0.4, 0.3]


###########
# Adjust these values for [1] Run AceCF (Train-Pred).ipynb
DOMAIN = "auto"
ITEM_RATIO = 0.4
USER_RATIO = 0.4
REDUCE_DIM = 32
CF_LRATE = 0.006

N_EPOCHS = 30
B_SIZE = 128
MLP_SIZE = 4
EMB_SIZE = 1024
RECORD_TB = False


