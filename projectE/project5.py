import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import shutil
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal


from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from torch.utils.data import DataLoader
from YourCustomDataset import MorseCodeDataset   # Replace 'YourCustomDataset' with the actual dataset class name
from CNNModel import MyCNNModel          



# convert audio files to spectrogram images

def create_spectrogram(audio_path, output_path, sr=22050, n_fft=2048, hop_length=512):
    # loading audio file
    p, sr = librosa.load(audio_path, sr=sr)
 
 #computing short time fourier transform (stft)
    A = librosa.stft(p, n_fft=n_fft, hop_length=hop_length)
    
    #converting magnitude spectrogram to db scale
    S_db = librosa.amplitude_to_db(abs(A))
    
    #a plot of spectrogram
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.savefig(output_filename)
    plt.close()


#setting path to download morse code dataset and the output folder

morse_code_dataset_path = 'c:\Users\ser\Morse_Code_Dataset'
output_folder_path = 'c:\Users\ser\Documents\Spectrogram_Images'

#process each audio file dataset
for root, dirs, files in os.walk(morse_code_dataset_path):
    for file in files:
        if file.endswith('.wav'):
            audio_file_path = os.path.join(root, file)
            create_spectrogram(audio_file_path, output_folder_path)

#path to downloaded morse code dataset
morse_code_dataset_path = 'c:\Users\ser\Morse_Code_Dataset'
#main folder for holding the subfolders
organized_dataset_path = 'c:\Users\ser\Documents\Morse_Code_Dataset_Organized'
os.makedirs(organized_dataset_path, exist_ok=True)

#extracting unique letters/symbols from filenames
def extract_letters(filename):
    return filename.split('_')[0]

data_path = 'c:\Users\ser\Morse_Code_Dataset'
def get_symbol_from_filename(filename):
    return filename.slpit('_')[0]
unique_letters = []

audio_files = [filename for filename in os.listdir(data_path) if filename.endswith('.wav')]

# Loop through the audio files and extract the symbols
for audio_file in audio_files:
    symbol = get_symbol_from_filename(audio_file)
    if symbol not in unique_letters:
        unique_letters.append(symbol)

print("Unique letters/symbols found in the dataset:", unique_letters)

#set to store unique letters/symbols in the dataset
unique_letters = set()

#scan dataset and extract unique letters/simbols
for root, dirs, files in os.walk(morse_code_dataset_path):
    for file in files:
        if file.endswith(".wav"):
            letter = extract_letters(file)
            unique_letters.add(letter)
#ffor each unique letter/symbols
for letter in unique_letters:            
    letter_folder_path = os.path.join(organized_dataset_path, letter)
    os.makedirs(letter_folder_path, exist_ok=True)
print("Subfolders for unique letter&symbols created successfull...")

 
 #processing audio file in the dataset
for root, dirs, files in os.walk(morse_code_dataset_path):
    for file in files:
        if file.endswith('.wav'):
            audio_file_path = os.path.join(root, file)

            #extracting letter/symbol from filename
            letter = file.split('_')[0]
            
            #subfolder for letter/symbol if doesnot exist
            letter_folder_path = os.path.join(organized_dataset_path, letter)
            os.makedirs(letter_folder_path, exist_ok=True)

            #moving audio file to corresponding subfolder
            new_file_path = os.path.join(letter_folder_path, file)
            shutil.move(audio_file_path, new_file_path)
print("Dataset organized into subfolders successfully.")
#setunique letters$symbols found in the dataset
unique_letters = ['A', 'B', 'C', ..., 'Z', '0', '1', '2', ..., '9', '.', ',', '?', ...]
  # label mapping dictionary
label_mapping = {letter: index for index, letter in enumerate(unique_letters)}
print(label_mapping)

#setting path to organized morse code dataset
organized_dataset_path = 'c:\Users\ser\Morse_Code_Dataset_Organized'

#folder for trainning, validation, and testing sets
train_folder = os.path.join(organized_dataset_path, 'train')
val_folder = os.path.join(organized_dataset_path, 'val')
test_folder = os.path.join(organized_dataset_path, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

#move file to respective folders
def move_files(source_folder, target_folder, file_list):
    for file in file_list:
        source_path = os.path.join(source_folder, file)
        target_path = os. path.join(target_folder, file)
        shutil.move(source_path, target_path)

#spliting database into train, validation, and test
for letter in os.listdir(organized_dataset_path):
    letter_folder = os.path.join(organized_dataset_path, letter)
    files = os.listdir(letter_folder)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    #mv file to resoective folder
    move_files(letter_folder, os.path.join(train_folder, letter), train_files)
    move_files(letter_folder, os.path.join(val_folder, letter), val_files)
    move_files(letter_folder, os.path.join(test_folder, letter), test_files)
print("Dataset split into Train, Validation, & Test set is succeful.")

#lets define cnn model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn. Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn. Linear(128 * 16 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, z):
        z = self.conv_layers(z)
        z = z.view(z.size(0), -1)
        z = self.fc_layers(z)
        return z
    
    #lets set the number of output classes to unique letter&symbols
num_classes = len(label_mapping)
model = CNNModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# no set device (gpu or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defining data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])
 # load training and validation datasets
train_dataset= YourCustomDataset(train_folder, transform=transform)
val_dataset = YourCustomDataset(val_folder, transform=transform)

#dataloder defining, batch size can be adjust depping on memory of system
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#hyperparameters and initializing cnn model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#lets train loop
num_epochs = 10 
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct_predictions / len(train_dataset)

    #validate
    model.eval()
    val_correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == labels). sum().item()
    val_accuracy = val_correct_predictions / len(val_dataset)

    #training and validation metrics for epoch print
    print(f'Epoch [{epoch+1} / {num_epochs}] -'
          f'Train Loss: {train_loss:.4f}, Train Acuuracy: {train_accuracy:.4f},'
          f'Val Accuracy: {val_accuracy:.4f}')
    if val_accuracy > best_val_accuracy:
       best_val_accuracy = val_accuracy
       torch.save(model.state_dict(), 'best_model_checkingpoint.pth')
print('Training completed.')

#to load testing dataset
test_dataset = YourCustomDataset(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# trained model
model = YourModelClass(num_classes)
model.load_state_dict(torch.load('best_model_checkpoint.pth'))
model.to(device)

#loop evaluation
model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

      # let convert predictions to labels to numpy arrays for evaluation
predictions = np.array(all_predictions)
labels = np.array(all_labels)

#calculate evaluation 

accuracy = accuracy_score(labels, predictions)
precision =precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')

#we can now print evaluations metrics
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1-Score: {f1:.4f}')


#lets define a threshold for letter/symbol detction
threshold = 0.5

#predictions from trained cnn model
model.eval()
with torch.no_grad():
    spectrograms = torch.tensor(spectrograms, dtype=torch.float32). unsqueeze(1).to(device)
    probabilities = torch.softmax(model(spectrograms), dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)

#applying threshol to detect letter boudaries
letter_start_times = []
letter_end_times = []
current_letter_start = None
for i in range(len(predicted_labels)):
    if probabilities[i, predicted_labels[i]] >= threshold:
        if current_letter_start is None:
            current_letter_start =i

    else:
        if current_letter_start is not None:
            letter_end_times.append(current_letter_start)
            letter_end_times.append(i - 1)
            current_letter_start = None

        #if last letter&symbols continues to end of the audio, add its entime
if current_letter_start is not None:
    letter_start_times.end(current_letter_start)
    letter_end_times.append(len(predicted_labels) - 1)

detected_symbols = ['.--', '.-', ' ', '-.-.--', '...', '---', '.-', '.-', ' ', '--.', '---', ' ', '....', '.-', '.-..', '.-..', ' ', '-.--', '.-', '--.', ' ', '...-.', '.', '.-', '--.', ' ', '-.-', '.-', '.-.', ' ', '-.--', '--', '--..', ' ', '.--', '---', '.-', '.-.', ' ', '...-.', '.', '.-', '--.', ' ']
start_times = [0.00, 1.12, 1.65, 2.01, 2.85, 3.34, 4.27, 4.83, 5.11, 5.68, 6.22, 6.45, 7.25, 7.82, 8.35, 9.13, 9.70, 10.26, 10.48, 11.30, 11.88, 12.44, 13.28, 13.80, 14.60, 15.15, 15.68, 16.45, 17.00, 17.56, 17.81, 18.40, 19.10, 19.77, 20.22, 20.86, 21.40, 22.00]
end_times = [1.05, 1.60, 2.00, 2.67, 3.30, 3.97, 4.78, 5.07, 5.63, 6.16, 6.42, 7.19, 7.80, 8.30, 9.10, 9.60, 10.22, 10.45, 11.25, 11.82, 12.40, 13.22, 13.75, 14.55, 15.11, 15.62, 16.42, 16.98, 17.53, 17.78, 18.30, 18.90, 19.70, 20.20, 20.82, 21.37, 22.00, 22.00]

#converting detected symbols to morse code
morse_code_text = ''.join(detected_symbols)

#translate morse code into readble text
translated_text = []
current_word = []
current_start_time = None
current_end_time =None
# Function to create a test file for each audio file
def create_test_file(audio_file, detected_info):
    test_file_path = os.path.join(test_result_folder, os.path.splitext(os.path.basename(audio_file))[0] + "_test.txt")
    with open(test_file_path, "w") as test_file:
        for item in detected_info:
            letter = item["letter"]
            start_time = item["start_time"]
            end_time = item["end_time"]
            test_file.write(f"Letter: {letter}, Start Time: {start_time:.2f}, End Time: {end_time:.2f}\n")
    print(f"Test file created: {test_file_path}")

# Loop through detected symbols and timings to detect letters and create test files
detected_information_list = []
current_letter_info = None
for i in range(len(detected_symbols)):
    symbol = detected_symbols[i]
    start_time = start_times[i]
    end_time = end_times[i]

    if symbol != ' ':
        if current_letter_info is None:
            current_letter_info = {"letter": symbol, "start_time": start_time, "end_time": end_time}
        else:
            current_letter_info["end_time"] = end_time
    else:
        if current_letter_info is not None:
            detected_information_list.append(current_letter_info)
            current_letter_info = None

# If the last letter extends to the end of the audio, add its end time
if current_letter_info is not None:
    detected_information_list.append(current_letter_info)

# Change letter start and end times to actual time units
time_unit = 0.01
for item in detected_information_list:
    item["start_time"] *= time_unit
    item["end_time"] *= time_unit

# Print detected letter boundaries
for item in detected_information_list:
    print(f"Letter: {item['letter']}, Start Time: {item['start_time']:.2f}, End Time: {item['end_time']:.2f}")

# Let's create test files for each audio file
for audio_file in audio_files:
    create_test_file(audio_file, detected_information_list)

    
    #change letter start and end times to actual time units 
time_unit = 0.01
letter_start_times = [start_time * time_unit for start_time in letter_start_times]
letter_end_times = [end_time * time_unit for end_time in letter_end_times]

#we can print detected letter boundary
for start_time, end_time in zip(letter_start_times, letter_end_times):
    print(f"Letter Start Time: {start_time}, End Time: {end_time}")

# lets define the folder where the test files sre saved
test_result_folder = "test_results"

#result folder
if not os.path.exists(test_result_folder):
    os.makedirs(test_result_folder)

    #loop though audiofile and detected information
for audio_file, detected_info in zip(audio_files_list, detected_information_list):
        test_file_name = os.path.splitext(os.path.basename(audio_file))[0] + "_test.txt"
        with open(test_file_path, "w") as test_file:
            for item in detected_info:
                letter = item["letter"]
                start_time = item["start_time"]
                end_time = item["end_time"]
                test_file.write(f"Letter: {letter}, Start Time: {start_time:.2}, End Time: {end_time:.2f}\n")
        print(f"Test file created: {test_file_name}")


#morse code dictionary

# Define the Morse code dictionary
morse_code_dict = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
    '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&', '---...': ':',
    '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS',
    '': ' ',  # Morse code for space
}

# Assuming you have the detected letters/symbols and their timing information from test files
detected_symbols = ['.--', '.-', ' ', '-.-.--', '...', '---', '.-', '.-', ' ', '--.', '---', ' ', '....', '.-', '.-..', '.-..', ' ', '-.--', '.-', '--.', ' ', '...-.', '.', '.-', '--.', ' ', '-.-', '.-', '.-.', ' ', '-.--', '--', '--..', ' ', '.--', '---', '.-', '.-.', ' ', '...-.', '.', '.-', '--.', ' ']
start_times = [0.00, 1.12, 1.65, 2.01, 2.85, 3.34, 4.27, 4.83, 5.11, 5.68, 6.22, 6.45, 7.25, 7.82, 8.35, 9.13, 9.70, 10.26, 10.48, 11.30, 11.88, 12.44, 13.28, 13.80, 14.60, 15.15, 15.68, 16.45, 17.00, 17.56, 17.81, 18.40, 19.10, 19.77, 20.22, 20.86, 21.40, 22.00]
end_times = [1.05, 1.60, 2.00, 2.67, 3.30, 3.97, 4.78, 5.07, 5.63, 6.16, 6.42, 7.19, 7.80, 8.30, 9.10, 9.60, 10.22, 10.45, 11.25, 11.82, 12.40, 13.22, 13.75, 14.55, 15.11, 15.62, 16.42, 16.98, 17.53, 17.78, 18.30, 18.90, 19.70, 20.20, 20.82, 21.37, 22.00, 22.00]

# Convert detected symbols to Morse code
morse_code_text = ''.join(detected_symbols)

# Translate Morse code into readable text
translated_text = []
current_word = []
current_start_time = None
current_end_time = None

for symbol, start_time, end_time in zip(detected_symbols, start_times, end_times):
    if symbol == ' ':
        # End of word detected, convert Morse code to letter and add to translated_text
        morse_word





















