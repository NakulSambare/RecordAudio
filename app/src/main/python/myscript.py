from keras.models import load_model
import librosa
import numpy as np
def predict(audio):
    model = load_model('src/main/res/best_model.hdf5')
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return labels[index]

def main(filepath):
    labels = ['bed', 'five', 'happy', 'four', 'go', 'house', 'left', 'marvin', 'nine', 'off', 'no', 'on', 'right', 'one', 'sheila', 'seven', 'six', 'stop', 'three', 'two', 'tree', 'up', 'zero', 'yes', 'wow', 'bird', 'eight', 'dog', 'down', 'cat']
    labels.sort()

    #reading the voice commands
    samples, sample_rate = librosa.load(filepath, sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    samples = samples[:8000]
   
    return predict(samples)
