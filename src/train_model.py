import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from .model_architecture import NN, INPUT_DIM, OUTPUT_DIM, get_params
from .create_dataloader import Music, collate_batch, WINDOW_SIZE, TO_PREDICT, BATCH_SIZE

def train_model(name):
	cur_dir = os.getcwd()
	
	model = None
	try:
		with open(f"{cur_dir}/models/{name}", 'rb') as f:
			model = pickle.load(f)
		lr = 0.0003
	except:
		model = NN(INPUT_DIM, OUTPUT_DIM)
		
	with open(f"{cur_dir}/dataloaders/{name}", 'rb') as f:
		dataloader = pickle.load(f)
		lr = 0.002
		
	loss_fn_cat = nn.CrossEntropyLoss()
	loss_fn_reg1 = nn.L1Loss()
	loss_fn_reg2 = nn.L1Loss()
	
	epoch = 0
	try:
		while True:
			model.train()
			epoch += 1
			pbar = tqdm(range(10), position=0, leave=True)
			
			model_params = get_params(model)
			optimizer1 = torch.optim.Adam(model_params[0], lr=lr)
			optimizer2 = torch.optim.Adam(model_params[1], lr=lr)
			optimizer3 = torch.optim.Adam(model_params[2], lr=lr)
		
			notes_correct = 0
			times_loss = 0
			vlcty_loss = 0
			total_guessed = 0
			
			for _ in pbar:
				for xn, xl, xv, yn, yl, yv in dataloader:
					out_n, out_l, out_v = model(xn)
					
					out_n = out_n.view(-1, out_n.shape[-1])
					out_l = out_l.flatten()
					out_v = out_v.flatten()
					yn = yn[-1*TO_PREDICT:, :].flatten()
					yl = yl[-1*TO_PREDICT:, :].flatten()
					yv = yv[-1*TO_PREDICT:, :].flatten()
				
					n_loss = loss_fn_cat(out_n, yn.long())
					l_loss = loss_fn_reg1(out_l, yl.float())
					v_loss = loss_fn_reg2(out_v, yv.float())
					
					model.zero_grad()
					loss = n_loss + l_loss + v_loss
					loss.backward()
					optimizer1.step()
					optimizer2.step()
					optimizer3.step()
					
					notes_correct += torch.sum(yn==out_n.argmax(-1)).item()
					times_loss += l_loss.item()
					vlcty_loss += v_loss.item()
					total_guessed += yn.nelement()
					total_loss = loss.item()
					pbar.set_postfix({
						"loss": total_loss / total_guessed,
						"notes acc": notes_correct * 100 / total_guessed,
						"times loss": times_loss / total_guessed,
						"vlcty loss": vlcty_loss / total_guessed
					})
				
			lr *= 0.9999
			
			# model autosave
			if epoch%10 == 0:
				with open(f"{cur_dir}/models/{name}", 'wb') as f:
					pickle.dump(model, f)
	except:
		# process exited. saving the model
		with open(f"{cur_dir}/models/{name}", "wb") as f:
			pickle.dump(model, f)
