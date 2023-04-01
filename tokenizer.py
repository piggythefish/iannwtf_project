# tokenize with BPE
from miditok import REMI
from pathlib import Path
from functions import midi_valid

# Creates the tokenizer and list the file paths
tokenizer = REMI(sos_eos=True)
midi_paths = list(Path('data/new_grand_midi').glob('**/*.mid*'))

# Converts MIDI files to tokens saved as JSON files
tokenizer.tokenize_midi_dataset(        
    midi_paths,
    Path('data/new_grand_midi_noBPE'),
    midi_valid
)

# Learns the vocabulary with BPE
tokenizer.learn_bpe(
    'data/new_grand_midi_noBPE',
    512,
    'data/new_grand_midi_BPE'
)

# Converts the tokenized musics into tokens with BPE
tokenizer.apply_bpe_to_dataset(Path('data/new_grand_midi_noBPE'), Path('data/new_grand_midi_BPE'))

