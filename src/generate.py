import os, sys
import pickle
import torch
from tqdm import tqdm
from .model_architecture import NN
from .create_dataloader import WINDOW_SIZE, TO_PREDICT
from .midi_work import encode_midi

def generate(path):
	cur_dir = os.getcwd()
	model_path = f"{cur_dir}/models/{path}"
	with open(model_path, 'rb') as f:
		model = pickle.load(f)
	
	notes_window = [0] * WINDOW_SIZE
	
	canvas_notes = list()
	canvas_times = list()
	canvas_vlcty = list()
	to_predict = 4096
	
	print("\nGenerating track...")
	pbar = tqdm(range(to_predict//TO_PREDICT))
	model.eval()
	for i in pbar:
		out_n, out_t, out_v = model(
			torch.as_tensor(notes_window).view(-1, 1)
		)
		
		out_n = out_n.argmax(-1).flatten().tolist()
		out_t = out_t.flatten().tolist()
		out_v = out_v.flatten().tolist()
		
		notes_window = notes_window[TO_PREDICT:] + out_n
		
		canvas_notes += out_n
		canvas_times += out_t
		canvas_vlcty += out_v
	
	canvas_times = [0 if x<=0 else int(x*32) for x in canvas_times]
	canvas_vlcty = [0 if x<8/32 else 127 if x>=127/32 else int(x*32) for x in canvas_vlcty]
	encode_midi(canvas_notes, canvas_times, canvas_vlcty, f"{cur_dir}/output/{path}.midi")
