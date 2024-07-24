import os
import cv2
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr
from collections import Counter
import pandas as pd


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
        #print(f"Finished processing {chunk_filename}. Emotion counts saved to {counts_filepath}.")

# Initialize a list to store the top emotion from each CSV
top_emotions_list = []

# Iterate over each CSV file in the emotion_counts directory
for counts_filename in os.listdir(emotion_counts_dir):
    if counts_filename.endswith('.csv'):
        counts_filepath = os.path.join(emotion_counts_dir, counts_filename)
        
        # Read the CSV file into a DataFrame
        df_counts = pd.read_csv(counts_filepath, index_col=0)
        
        # Find the top emotion (the emotion with the highest count)
        top_emotion = df_counts['Count'].idxmax()
        top_count = df_counts['Count'].max()
        
        # Extract the chunk name from the filename
        chunk_name = counts_filename.replace('_emotion_counts.csv', '')
        
        # Add the top emotion and its count to the list
        top_emotions_list.append({
            'Top Emotion': top_emotion,
            'Count': top_count
        })

# Convert the list of top emotions into a DataFrame
df_top_emotions = pd.DataFrame(top_emotions_list)

# Specify the path for the final consolidated CSV file
final_csv_path = 'Final/emotion_predictions_face.csv'

# Save the DataFrame to a CSV file
df_top_emotions.to_csv(final_csv_path, index=False)

print(f"Consolidated emotion predictions saved to {final_csv_path}.")