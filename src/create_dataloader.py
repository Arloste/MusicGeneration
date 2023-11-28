import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from .midi_work import decode_midi
from tqdm import tqdm
from random import randint, uniform, randrange

WINDOW_SIZE = 256
TO_PREDICT = 4
BATCH_SIZE = 4

class Music(Dataset):
	def __init__(self, paths):
		self.notes = list()
		self.times = list()
		self.vlcty = list()
		
		pbar = tqdm(paths)
		for path in pbar:
			pbar.set_postfix_str(f"Processing {path}...")
			notes, times, vlcty = decode_midi(path)
			self.notes.append(notes)
			self.times.append(times)
			self.vlcty.append(vlcty)
	
	def __getitem__(self, idx):
		return self.notes[idx], self.times[idx], self.vlcty[idx]
	
	def __len__(self):
		return len(self.notes)
        
        
def collate_batch(batch):
	x_notes_batch = list()
	x_lengths_batch = list()
	x_vlcty_batch = list()

	y_notes_batch = list()
	y_lengths_batch = list()
	y_vlcty_batch = list()

	for notes, lengths, velocities in batch:
	
		# sampling a random part of the track
		point = randrange(0, len(notes) - TO_PREDICT)
		
		x_notes = notes[max(0, point-WINDOW_SIZE):point]
		x_lengths = lengths[max(0, point-WINDOW_SIZE):point]
		x_vlcty = velocities[max(0, point-WINDOW_SIZE):point]
        
		y_notes = notes[max(0, point-WINDOW_SIZE+TO_PREDICT):point+TO_PREDICT]
		y_lengths = lengths[max(0, point-WINDOW_SIZE+TO_PREDICT):point+TO_PREDICT]
		y_vlcty = velocities[max(0, point-WINDOW_SIZE+TO_PREDICT):point+TO_PREDICT]
		
		
		# random augmentations
		transpose = randint(-7, 7)
		len_mlt, vel_mlt = uniform(0.9, 1.1), uniform(0.9, 1.1)
		
		x_notes = [0 if x+transpose<0 else 127 if x+transpose>127 else x+transpose for x in x_notes]
		y_notes = [0 if y+transpose<0 else 127 if y+transpose>127 else y+transpose for y in y_notes]
		x_lengths = [x*len_mlt for x in x_lengths]
		y_lengths = [y*len_mlt for y in y_lengths]
		x_vlcty = [x*vel_mlt for x in x_vlcty]
		y_vlcty = [y*vel_mlt for y in y_vlcty]
		
		
		# padding
		x_notes = [0] * (WINDOW_SIZE - len(x_notes)) + x_notes
		x_lengths = [0] * (WINDOW_SIZE - len(x_lengths)) + x_lengths
		x_vlcty = [0] * (WINDOW_SIZE - len(x_vlcty)) + x_vlcty

		y_notes = [0] * (WINDOW_SIZE - len(y_notes)) + y_notes
		y_lengths = [0] * (WINDOW_SIZE - len(y_lengths)) + y_lengths
		y_vlcty = [0] * (WINDOW_SIZE - len(y_vlcty)) + y_vlcty

		x_notes_batch.append(x_notes)
		x_lengths_batch.append(x_lengths)
		x_vlcty_batch.append(x_vlcty)

		y_notes_batch.append(y_notes)
		y_lengths_batch.append(y_lengths)
		y_vlcty_batch.append(y_vlcty)

	x_notes_batch = torch.transpose(torch.as_tensor(x_notes_batch), -2, 1)
	x_lengths_batch = torch.transpose(torch.as_tensor(x_lengths_batch), -2, 1)
	x_vlcty_batch = torch.transpose(torch.as_tensor(x_vlcty_batch), -2, 1)
	
	y_notes_batch = torch.transpose(torch.as_tensor(y_notes_batch), -2, 1)
	y_lengths_batch = torch.transpose(torch.as_tensor(y_lengths_batch), -2, 1)
	y_vlcty_batch = torch.transpose(torch.as_tensor(y_vlcty_batch), -2, 1)
	
	
	return x_notes_batch, x_lengths_batch, x_vlcty_batch, y_notes_batch, y_lengths_batch, y_vlcty_batch


def create_dataloader(name, grep):
	cur_dir = os.getcwd()
	music_files = os.listdir(f"{cur_dir}/dataset/")
	music_files = [f"{cur_dir}/dataset/{x}" for x in music_files if grep in x]
	
	dataset = Music(music_files)
	dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
	
	with open(f"{cur_dir}/dataloaders/{name}", 'wb') as f:
		pickle.dump(dl, f)
	

