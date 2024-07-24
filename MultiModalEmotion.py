from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from math import ceil
import numpy as np

import torch
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    AutoTokenizer, AutoModelForSequenceClassification
)
import librosa
import os
import pandas as pd
import cv2
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr
from collections import Counter

class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        emotions = []
        for cropped_image, box2D in zip(self.crop(image, boxes2D), boxes2D):
            emotion = self.classify(cropped_image)['class_name']
            emotions.append(emotion)
            box2D.class_name = emotion
        image_with_boxes = self.draw(image, boxes2D)
        return image_with_boxes, emotions


video_path = "Video1/Crying_video.mp4"  # Change this to the path of your video file
audio_path = "Audio2/emotion.mp3"  # Output path for the extracted audio
video_chunks_dir = "Video2"  # Directory to save video chunks

# Load the video file and extract the audio
video_clip = VideoFileClip(video_path)
audio_clip = video_clip.audio
audio_clip.write_audiofile(audio_path)

#audio_clip.close()
#video_clip.close()

# Load the extracted audio
audio = AudioSegment.from_file(audio_path)

# Define the length of each chunk in milliseconds
chunk_length_ms = 3000

# Calculate the number of chunks needed
duration_in_ms = video_clip.duration * 1000  # Convert video duration from seconds to milliseconds
num_chunks = ceil(len(audio) / chunk_length_ms)

# Split the audio and save each chunk
for i in range(num_chunks):
    start_time = i * chunk_length_ms
    end_time = min((i + 1) * chunk_length_ms, len(audio))
    chunk = audio[start_time:end_time]
    chunk_name = f'Audio2\chunk_{i+1}.mp3'  # Naming each chunk
    chunk.export(chunk_name, format="mp3")
    #print(f'Exported {chunk_name}')

# After exporting all chunks, delete the original audio file

#print(f'Deleted original audio file at {audio_path}')

# Get fps from the original video clip
fps = video_clip.fps
print(fps)
# Split the video and save each chunk
for i in range(num_chunks):
    start_time = i * chunk_length_ms / 1000.0  # Convert milliseconds to seconds
    end_time = min((i + 1) * chunk_length_ms / 1000.0, duration_in_ms / 1000.0)
    # Create a subclip for each chunk
    chunk = video_clip.subclip(start_time, end_time)
    # Naming each chunk
    chunk_name = f'chunk_{i+1}.mp4'
    chunk_file_path = os.path.join(video_chunks_dir, chunk_name)
    print(chunk_file_path)
    # Save the chunk to file
    chunk.write_videofile(chunk_file_path, codec="libx264", audio_codec="aac", fps=fps)  # Here pass the fps
    print(f'Exported {chunk_name}')

os.remove(audio_path)
video_clip.close()
audio_clip.close()
print("Completed video chunking.")

# Directory where the chunks are saved
chunks_dir = 'Audio2'
audio_files = [f for f in os.listdir(chunks_dir) if f.endswith('.mp3')]
audio_files.sort()  # Optional, to process the files in a sorted order

df = pd.DataFrame(columns=['feature'])
bookmark = 0

for index, filename in enumerate(audio_files):
    # Adjust the condition according to your naming convention, if needed
    file_path = os.path.join(chunks_dir, filename)
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    df.loc[bookmark] = [feature]
    bookmark += 1

# Now, df contains the features extracted from each audio chunk
#print(df)
df = pd.DataFrame(df['feature'].values.tolist())
df[:]
df.fillna('0')

model=load_model('Voice/Voice-Emotion-Detector/saved_models/Emotion_Voice_Detection_Model.h5')   
print('Imported the model named %s ' % model)

labels = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']

lb = LabelEncoder()

# Fit the LabelEncoder instance to your known labels
lb.fit(labels)

x_traincnn =np.expand_dims(df, axis=2)

preds = model.predict(x_traincnn, batch_size=1, verbose=1)
preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})
preddf[:]

# Assuming preddf is your DataFrame
# Remove 'male_' and 'female_' prefixes from the 'predictedvalues' column
preddf['predictedvalues'] = preddf['predictedvalues'].str.replace('female_', '').str.replace('male_', '')

#print(preddf)
# Calculate the percentage of each emotion
emotion_counts = preddf['predictedvalues'].value_counts(normalize=True) * 100

# Convert the emotion_counts to a DataFrame for nicer formatting, if desired
emotion_percentage_df_audio_sound = emotion_counts.reset_index().rename(columns={'index': 'emotion', 'predictedvalues': 'percentage'})

# Display the emotion percentage DataFrame
#print(emotion_percentage_df_audio_sound)

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using device: {device}")

# Load Wav2Vec2 model and processor for transcription
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Load RoBERTa model and tokenizer for emotion classification
emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)

# Define labels
labels = emotion_model.config.id2label

# Function to transcribe audio
def transcribe(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = wav2vec2_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec2_processor.batch_decode(predicted_ids)
    return transcription[0]

# Function to classify text
def classify_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    top_probs, top_lbls = torch.topk(probabilities, 3, dim=-1)
    return [(labels[top_lbls[0][i].item()], top_probs[0][i].item()) for i in range(3)]


def combine_text_files(input_dir, output_dir, combine_every=10):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all text files in the input directory
    text_files = [f for f in os.listdir(input_dir) if f.endswith('.txt') and f.startswith('chunk_')]
    text_files.sort()  # Sort the files to maintain order

    combined_texts = []

    for i, filename in enumerate(text_files):
        # Read the content of each text file
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
            content = file.read().strip()
        combined_texts.append(content)

        # Every combine_every files, or at the end, combine and write to a new file
        if (i + 1) % combine_every == 0 or i == len(text_files) - 1:
            combined_filename = f"combined_{i // combine_every + 1}.txt"
            combined_filepath = os.path.join(output_dir, combined_filename)
            with open(combined_filepath, 'w', encoding='utf-8') as outfile:
                outfile.write(" ".join(combined_texts))
                print(f'Created {combined_filename} with {len(combined_texts)} texts.')
            combined_texts = []  # Reset for the next group


# Directory with audio chunks
chunks_dir = 'Audio2'  # Adjust as necessary
audio_chunks = [f for f in os.listdir(chunks_dir) if f.endswith('.mp3')]

# Initialize list for DataFrame
data = []


# Specify the directory where you want to save the text files
text_files_directory = 'Audio2'

# Ensure the directory exists
os.makedirs(text_files_directory, exist_ok=True)

# Process each chunk
for chunk in audio_chunks:
    audio_path = os.path.join(chunks_dir, chunk)
    transcription = transcribe(audio_path)
    
    # Define the path for the new text file
    text_filename = chunk.replace('.mp3', '.txt')
    text_file_path = os.path.join(text_files_directory, text_filename)
    
    # Save the transcription to the text file
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(transcription)
    
    print(f"Transcription saved to {text_file_path}")

    # Now proceed with emotion classification
    emotions = classify_emotion(transcription)
    
    input_dir = 'Audio2'
    output_dir = 'Audio2'
    combine_text_files(input_dir, output_dir)

    # Prepare data for DataFrame
    row = {}  # Using text_filename to refer to the saved file
    for i, emo in enumerate(emotions):
        row[f'Emotion {i+1}'] = emo[0]
        row[f'Prob Emotion {i+1}'] = f"{emo[1]:.4f}"
    

    data.append(row)

results = []

for filename in os.listdir('Audio2'):
    if filename.endswith('.txt') and filename.startswith('combined_'):
        filepath = os.path.join('Audio2', filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().strip()

        # Classify emotion of the combined text
        emotion_predictions = classify_emotion(text)
        results.append({
            'Filename': filename,
            'Emotion 1': emotion_predictions[0][0], 'Prob 1': emotion_predictions[0][1],
            'Emotion 2': emotion_predictions[1][0], 'Prob 2': emotion_predictions[1][1],
            'Emotion 3': emotion_predictions[2][0], 'Prob 3': emotion_predictions[2][1],
        })


# Create DataFrame
df = pd.DataFrame(data)
# Create a DataFrame from the results
df_combined = pd.DataFrame(results)


# Initialize the emotion detector
detect = EmotionDetector()

video_chunks_dir = 'Video2'
images_dir = os.path.join(video_chunks_dir, 'images')
emotion_counts_dir = os.path.join(video_chunks_dir, 'emotion_counts')
os.makedirs(emotion_counts_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Process each video chunk
for chunk_filename in os.listdir(video_chunks_dir):
    if chunk_filename.endswith('.mp4'):
        chunk_path = os.path.join(video_chunks_dir, chunk_filename)
        cap = cv2.VideoCapture(chunk_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        emotions_all_frames = []

        for frame_index in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, emotions = detect(frame_rgb)
            emotions_all_frames.extend(emotions)

            # Optionally, save the frame with detected emotions
            frame_save_path = os.path.join(images_dir, f"frame_{frame_index+1}_{chunk_filename}.jpg")
            cv2.imwrite(frame_save_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

        # Tally the emotions for the current chunk
        emotion_counts = Counter(emotions_all_frames)
        # Save the emotion counts to a CSV file
        counts_filename = f"{chunk_filename.split('.')[0]}_emotion_counts.csv"
        counts_filepath = os.path.join(emotion_counts_dir, counts_filename)
        pd.DataFrame.from_dict(emotion_counts, orient='index', columns=['Count']).to_csv(counts_filepath)

        cap.release()
        print(f"Finished processing {chunk_filename}. Emotion counts saved to {counts_filepath}.")

# Initialize a list to store the top emotion from each CSV
top_emotions_list = []

# Iterate over each CSV file in the emotion_counts directory
for counts_filename in os.listdir(emotion_counts_dir):
    if counts_filename.endswith('.csv'):
        counts_filepath = os.path.join(emotion_counts_dir, counts_filename)
        
        # Read the CSV file into a DataFrame
        df_counts = pd.read_csv(counts_filepath, index_col=0)
        print(counts_filepath)
        # Find the top emotion (the emotion with the highest count)
        # Check if the DataFrame is empty
        if df_counts.empty:
            top_emotion = 'neutral'
            top_count = 0
        else:
            # Find the top emotion (the emotion with the highest count)
            top_emotion = df_counts['Count'].idxmax()
            top_count = df_counts['Count'].max()
        
        # Extract the chunk name from the filename
        chunk_name = counts_filename.replace('_emotion_counts.csv', '')
        
        # Add the top emotion and its count to the list
        top_emotions_list.append({
            'Chunk Name': chunk_name,
            'Top Emotion': top_emotion,
            'Count': top_count
        })


# Convert the list of top emotions into a DataFrame
df_top_emotions = pd.DataFrame(top_emotions_list)



# Display DataFrame
print(df)
print(preddf)
print(df_combined)
print(df_top_emotions)
print(emotion_percentage_df_audio_sound)


# Optionally, save the DataFrame to a CSV
df_combined.to_csv('Final/combined_emotion_predictions.csv', index=False)
# Save to CSV
df.to_csv('Final/emotion_predictions_from_audio_text.csv', index=False)
preddf.to_csv('Final/emotion_predictions_from_sound.csv', index=False)
# Save the DataFrame to a CSV file
df_top_emotions.to_csv('Final/emotion_predictions_face.csv', index=False)
emotion_percentage_df_audio_sound.to_csv('Final/emotion_predictions_from_audio_sound.csv', index=False)