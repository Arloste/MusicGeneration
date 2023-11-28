import torch
import torch.nn as nn
from .create_dataloader import TO_PREDICT

INPUT_DIM = 128
OUTPUT_DIM = 128

class NN(nn.Module):
	def __init__(self, inp_dim, out_dim):
		super().__init__()

		embed_dim = 8
		hid_dim = 512
		
		self.embedding = nn.Embedding(inp_dim, embed_dim)
		self.lstm = nn.LSTM(embed_dim, hid_dim//2, num_layers=2, dropout=0.3, bidirectional=True)

		self.hidden_notes1 = nn.Sequential(nn.Linear(hid_dim, hid_dim//2), nn.Dropout(0.3), nn.LeakyReLU())
		self.hidden_notes2 = nn.Sequential(nn.Linear(hid_dim//2, hid_dim//4), nn.Dropout(0.3), nn.LeakyReLU())
		self.out_notes = nn.Linear(hid_dim//4, out_dim)

		self.hidden_lengths1 = nn.Sequential(nn.Linear(hid_dim, hid_dim//2), nn.Dropout(0.3), nn.LeakyReLU())
		self.hidden_lengths2 = nn.Sequential(nn.Linear(hid_dim//2, hid_dim//4), nn.Dropout(0.3), nn.LeakyReLU())
		self.out_lengths = nn.Linear(hid_dim//4, 1)

		self.hidden_vlcty1 = nn.Sequential(nn.Linear(hid_dim, hid_dim//2), nn.Dropout(0.3), nn.LeakyReLU())
		self.hidden_vlcty2 = nn.Sequential(nn.Linear(hid_dim//2, hid_dim//4), nn.Dropout(0.3), nn.LeakyReLU())
		self.out_vlcty = nn.Linear(hid_dim//4, 1)
		
	def forward(self, x):
		x = self.embedding(x)
		x, (hidden, cell) = self.lstm(x)
		x = x[-1*TO_PREDICT:, :, :]
		
		x_notes = self.hidden_notes1(x)
		x_notes = self.hidden_notes2(x_notes)
		x_notes = self.out_notes(x_notes)

		x_times = self.hidden_lengths1(x)
		x_times = self.hidden_lengths2(x_times)
		x_times = self.out_lengths(x_times)

		x_vlcty = self.hidden_vlcty1(x)
		x_vlcty = self.hidden_vlcty2(x_vlcty)
		x_vlcty = self.out_vlcty(x_vlcty)

		return x_notes, x_times, x_vlcty

def get_params(model):
	NOTES_PARAMS = [
		{"params": model.embedding.parameters()},
		{"params": model.lstm.parameters()},
		{"params": model.hidden_notes1.parameters()},
		{"params": model.hidden_notes2.parameters()},
		{"params": model.out_notes.parameters()}
	]
	
	TIMES_PARAMS = [
		{"params": model.hidden_lengths1.parameters()},
		{"params": model.hidden_lengths2.parameters()},
		{"params": model.out_lengths.parameters()}
	]
	
	VLCTY_PARAMS = [
		{"params": model.hidden_vlcty1.parameters()},
		{"params": model.hidden_vlcty2.parameters()},
		{"params": model.out_vlcty.parameters()}
	]
	
	return NOTES_PARAMS, TIMES_PARAMS, VLCTY_PARAMS
