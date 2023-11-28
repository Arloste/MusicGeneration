from src.generate import generate
from src.model_architecture import NN
from src.create_dataloader import create_dataloader, Music, collate_batch
from src.train_model import train_model
import os

print("Hello!", end=" ")

while True:
	print("What would you like to do?")
	print("[1] - Train a model")
	print("[2] - Generate a music track")
	print("[3] - Quit the program")
	choice = input()
	
	if choice == "1":
		print("\nCreate a new model or continue training an existing one?")
		model_list = os.listdir("models")
		
		print("[0] - Create a new model")
		for i, f in enumerate(model_list):
			print(f"[{i+1}] - Continue training {f}")
		choice = int(input())
		
		if choice == 0:
			name = input("\nWrite the model name: ")
			grep = input("\nWrite a string that all music files to be trained with should contain: ")
			create_dataloader(name, grep)
		else:
			try: name = model_list[choice-1]
			except:
				print("Something went wrong")
				continue
		
		print(f"\nThe model '{name}' is training... You can terminate the process with ctrl+C")	
		try: train_model(name)
		except: print("Training successfully stopped\n")
		
			
		
	elif choice == "2":
		print("\nChoose a model from the list:")
		model_list = os.listdir("models")
		for i, f in enumerate(model_list):
			print(f"[{i+1}] - {f}")
		
		try: choice = model_list[int(input()) - 1]
		except:
			print("Something went wrong")
			continue
		
		generate(choice)
		print(f"You can find the output at output/{choice}.midi\n")
		
	
	elif choice == "3":
		print("\nGoodbye!")
		break
	
	else:
		print("Wrong input... type '1', '2', or '3' to proceed.")
