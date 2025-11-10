from speechbrain.pretrained import SepformerSeparation as separator

# Load pretrained model
model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir="pretrained_sepformer")

# Separate the speakers
est_sources = model.separate_file(path="/Users/keshikaa/Desktop/new-voice-clone-project/data/raw/verified_host_only_episodes/Agentic-AI-for-Drone-Robotic-Swarming.mp3")

# Save each speaker separately
model.audio_write("Dan.wav", est_sources[:, 0])
model.audio_write("Chris.wav", est_sources[:, 1])
