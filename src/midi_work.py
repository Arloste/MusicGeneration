import os, sys
import mido

def decode_midi(file_path):
    mid = mido.MidiFile(file_path)
    track = mid.tracks[1] # in maestro dataset notes are in track 1
    notes, times, vlcty = list(), list(), list()
    for msg in track:
    	if msg.type == "note_on":
    		notes.append(msg.note)
    		times.append(msg.time/32)
    		vlcty.append(0 if msg.velocity<16 else msg.velocity/32)
    	else:
    		try: times[-1] += msg.time/32
    		except: pass
    return notes, times, vlcty



def encode_midi(notes, lengths, velocities, path):
    track = mido.MidiTrack()
    for note, length, velocity in zip(notes, lengths, velocities):
        track.append(
            mido.Message('note_on',  channel=0, note=note, velocity=velocity, time=length)
        )
    midifile = mido.MidiFile(ticks_per_beat=384)
    midifile.tracks.append(track)
    midifile.save(path)
