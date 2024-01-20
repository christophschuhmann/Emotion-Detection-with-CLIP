from pytube import YouTube
import cv2
import moviepy.editor as mp
import numpy as np
from PIL import Image
import os
# Other imports and code (like for OpenCLIP) remain the same...

# Folder name
folder_name = "predictions"


save_images=False

# Create the folder
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

#comment this out, if the video is already downloaded and converted
'''
# Download YouTube Video
url = "https://youtu.be/uAQFxn2Ss84"  # Replace with your YouTube URL
yt = YouTube(url)
video_stream = yt.streams.filter(res="720p", file_extension='mp4').first()
audio_stream = yt.streams.filter(only_audio=True).first()

video_path = video_stream.download(filename='video.mp4')
audio_path = audio_stream.download(filename='audio.mp4')

# Combine video and audio
video_clip = mp.VideoFileClip(video_path)
audio_clip = mp.AudioFileClip(audio_path)
video_clip.audio = audio_clip
video_clip.write_videofile("combined.mp4", codec="libx264", audio_codec="aac")
'''

# Replace Webcam with Video
cap = cv2.VideoCapture("combined.mp4")


import cv2
from PIL import Image
import numpy as np
from time import sleep

import torch
from PIL import Image

import cv2
from PIL import Image

import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K")
simlist_person = []
simlist_valence = []
simlist_emotions = []


person = ["here you see a female, woman, girl", "here you see a male, man, boy" ]

valence=["positive mood, pleasant feelings, happy", "negative mood, unpleasant feelings, sad, fear, anger"]

emotiongroups = [
    "Sour/Tangy: 'tart', 'acidic', 'bitter', 'tangy', 'vinegary', 'sharp'",
    "Grateful/Thankful: 'thankful', 'appreciative', 'obliged', 'indebted', 'gratified', 'recognizant'",
    "Proud/Self-Important: 'dignified', 'haughty', 'arrogant', 'self-satisfied', 'vain', 'honored'",
    "Disgusted/Repulsed: 'repulsed', 'appalled', 'revolted', 'nauseated', 'repelled', 'sickened'",
    "Joyful/Elated: 'ebullient', 'merry', 'jovial', 'cheerful', 'lighthearted', 'joyful', 'beaming', 'grinning', 'elated', 'gleeful', 'happy'",
    "Hopeful/Optimistic: 'hopeful', 'gratitude', 'thankful', 'buoyant', 'upbeat', 'vibrant', 'radiant', 'exuberant', 'zestful', 'chirpy', 'peppy', 'jaunty', 'sprightly', 'brisk', 'lively', 'animated', 'energized', 'revitalized', 'invigorated', 'activated', 'energetic', 'dynamic', 'electrified', 'bouncy', 'effervescent', 'chipper', 'jubilant'",
    "Calm/Composed: 'mindful', 'unruffled', 'coolheaded', 'level-headed', 'poised', 'self-possessed', 'unflappable', 'collected', 'unperturbed', 'untroubled', 'unrattled', 'unshaken', 'unflustered', 'composed', 'relaxed', 'tranquil', 'serene', 'calm', 'centered', 'peaceful', 'imperturbable', 'reposeful', 'grounded', 'equanimous', 'harmonious'",
    "Attentive/Focused: 'engaging', 'focused', 'watchful', 'attentive', 'heedful', 'scrutinizing', 'investigating', 'alert', 'studious', 'analyzing', 'examining', 'cognizant', 'inquiring', 'questioning', 'probing', 'introspecting', 'introspective', 'observant'",
    "Intrigued/Awestruck: 'wondering', 'awe', 'intrigued', 'spellbinding', 'fascinated', 'mesmerized', 'captivated', 'bewitching', 'beguiling', 'agog', 'marveling', 'gazing', 'mystified', 'curious', 'riveted', 'enrapturing', 'entrancing', 'hypnotic', 'mesmerizing', 'alluring', 'enthralled'",
    "Thoughtful/Contemplative: 'pensive', 'ruminative', 'brooding', 'contemplating', 'meditative', 'reflective', 'pondering', 'cogitating', 'speculative'",
    "Fearful/Anxious: 'trembling', 'shuddery', 'afraid', 'spooked', 'apprehensive', 'fearful', 'terrorized', 'petrified', 'scared', 'horror-struck', 'quavering', 'shuddering', 'frightened', 'trepid', 'distraught', 'alarmed', 'fear-stricken', 'quaking', 'anxious', 'nervous', 'uneasy', 'worried', 'tense', 'jittery', 'jumpy', 'startled', 'edgy', 'antsy', 'rattled', 'distracted', 'disquieted', 'skittish', 'restless', 'restive', 'panic-stricken', 'panicked'",
    "Surprised/Amazed: 'dumbstruck', 'bewildered', 'dumbfounded', 'stunned', 'stupefied', 'thunderstruck', 'staggered', 'amazed', 'astonished', 'astounded', 'surprised', 'shocked', 'flabbergasted', 'befuddled', 'perplexed', 'confounded', 'baffled', 'discombobulated', 'flummoxed'",
    "Sad/Depressed: 'sad', 'dismal', 'forlorn', 'depressed', 'woebegone', 'plaintive', 'sorrowful', 'gloomy', 'lugubrious', 'melancholic', 'blue', 'desolate', 'miserable', 'downhearted', 'morose', 'somber', 'despairing', 'woeful', 'heartbroken', 'crestfallen', 'dispirited'",
    "Romantic/Passionate: 'romantic', 'amorous', 'passionate', 'sensual', 'infatuated', 'sensuous',  'in romantic love', 'steaminess', 'enticing', 'charming', 'flirty'",
    "Seductive/Provocative: 'seductive', 'titillating', 'sultry', 'erotic', 'salacious', 'libidinous',  'carnal', 'desirous', 'lecherous', 'lust-driven', 'prurient', 'lustful', 'ravishing',  'irresistible', 'provoked', 'craving'",
    "Angry/Aggravated: 'aggravated', 'perturbed', 'enraged', 'furious', 'irate', 'incensed', 'infuriated', 'wrathful', 'livid', 'cross', 'galled', 'resentful', 'bitter', 'indignant', 'outraged', 'exasperated', 'maddened', 'angry', 'annoyed', 'vexed', 'truculent', 'spiky', 'prickly', 'snarly', 'huffy', 'nettled', 'irritable', 'piqued', 'snappish', 'irascible', 'testy', 'nerved'",
    "Determined/Resolute: 'persistent', 'resilient', 'determined', 'unfailing', 'unyielding', 'tenacious', 'steadfast', 'adamant', 'resolute', 'undaunted', 'unwavering', 'unswerving', 'unflinching', 'unrelenting', 'enduring', 'indefatigable', 'motivated', 'driven'",
    "Disconcerted/Unsettled: 'discomposed', 'nonplussed', 'disconcerted', 'disturbed', 'ruffled', 'troubled', 'stressed', 'fractious', 'cringing', 'quailing', 'cowering', 'daunted', 'dread-filled', 'intimidated', 'unnerved', 'unsettled', 'fretful', 'ticked-off', 'flustered'",
    "Hostile/Combative: 'belligerent', 'pugnacious', 'contentious', 'quarrelsome', 'grumpy', 'grouchy', 'sulky', 'cranky', 'crabby', 'cantankerous', 'curmudgeonly', 'waspy', 'combative', 'argumentative', 'scrappy'",
    "Sleepy/Tired: 'fatigued', 'exhausted', 'weary', 'drowsy', 'sleepy', 'lethargic', 'drained'"

]

emotions= emotiongroups

tokenizer = open_clip.get_tokenizer('ViT-B-32')

import cv2

def get_largest_face_from_image(cv2_image):
    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade face detector model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # Or return cv2_image to return the original image if no faces are found

    # Find the largest face
    largest_face_area = 0
    largest_face = None

    for (x, y, w, h) in faces:
        face_area = w * h
        if face_area > largest_face_area:
            largest_face_area = face_area
            largest_face = (x, y, w, h)

    x, y, w, h = largest_face
    # Crop the largest face
    cropped_face = cv2_image[y:y+h, x:x+w]

    return cropped_face


def encode_text(text_list):
    text = tokenizer(text_list)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

person_features = encode_text(person)
valence_features = encode_text(valence)
emotion_features = encode_text(emotions)


# The CLIP B32 trained on datacomp has some bias towards this category,
# therefore I calculate the seductive_features to later substract their influnce from the image embeddings
# #before the zero shot classification

seductive_features = encode_text(["Seductive/Provocative: 'seductive', 'titillating', 'sultry', 'erotic', 'salacious', 'libidinous',  'carnal', 'desirous', 'lecherous', 'lust-driven', 'prurient', 'lustful', 'ravishing',  'irresistible', 'provoked', 'craving'"])
print(seductive_features.shape)

# The features are needed for detecting frames with CLIP that show big faces
moviestill_features = encode_text(["movie still, hollywood, nature, indoors"])
movietitles_features = encode_text(["movie titles, black screen, text, screenshot"])
face_features = encode_text(["face, closeup of a face/person"])
# moviestillface_features = moviestill_features *0.0+ face_features * 1

# Loading mean & standard deviation similarities for each emotion category of ~2000 images with people
# we will later substract the mean and divide by the stdev, because this will convert all sims into a similar scale
loaded_mean_sims_emo = np.array([[0.15599771 ,0.1511304 , 0.16875783, 0.19157822, 0.17763473 ,0.14812051,
  0.16381973, 0.15677759, 0.14437722, 0.17471613, 0.16821687, 0.17212224,
  0.1740183 , 0.15789438 ,0.16779166 ,0.17950825, 0.15191281, 0.16258015,
  0.14764176, 0.15238074]])

loaded_stdev_sims_emo = np.array([[0.0288077 , 0.02833298, 0.02468313, 0.03723702, 0.02814594, 0.0304083,
  0.03004773, 0.03482676, 0.03983417, 0.04059409, 0.03653273, 0.03492723,
  0.03197159, 0.03464679, 0.03582928, 0.03437715, 0.0293751,  0.02760981,
  0.03604855]] )


# Loading mean & standard deviation similarities for each gender (man / woman) of ~2000 images with people
# we will later substract the mean and divide by the stdev, because this will convert all sims into a similar scale

loaded_mean_sims_person = np.array([[0.18671854,0.1657077]]) #np.load(r'C:\Users\pc\PycharmProjects\realtimeSTT\mean_sims_valence.npy')

loaded_stdev_sims_person= np.array([[0.0340511,0.03612482]]) #np.load(r'C:\Users\pc\PycharmProjects\realtimeSTT\std_dev_sims_valence.npy')



# Loading mean & standard deviation similarities for positive and negative valence of ~2000 images with people
# we will later substract the mean and divide by the stdev, because this will convert all sims into a similar scale
loaded_mean_sims_valence = np.array([[0.155,0.19]]) #np.load(r'C:\Users\pc\PycharmProjects\realtimeSTT\mean_sims_valence.npy')
loaded_stdev_sims_valence = np.array([[0.05,0.05]]) #np.load(r'C:\Users\pc\PycharmProjects\realtimeSTT\std_dev_sims_valence.npy')


# change this if you want to use webcam images instead of the youtube frames
# Webcam initialisieren
#cap = cv2.VideoCapture(0)
# Replace Webcam with Video
cap = cv2.VideoCapture("combined.mp4")

original_fps = cap.get(cv2.CAP_PROP_FPS)

# desired framerate
desired_fps = 1

# font for displaying predictions
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_color = (255, 165, 0)  # Orange color in BGR
line_type = 2


# Frame-jump
frame_jump = round(original_fps / desired_fps)
frame_count=0

try:
    while True:
        # get frame
        ret, frame = cap.read()


        if not ret:
            print("Error with getting frame")
            break

        # if no frame was available, the video ended
        if not ret:
            break
        frame_count += 1
        # show only frame_jump-th frame
        if frame_count % frame_jump == 0:
            pass
        else:
            continue


        #cv2.imshow('Webcam-Bild', frame)

        face= get_largest_face_from_image(frame)
        frame = face
        if frame is None:
            continue
        height, width = frame.shape[:2]

        # Check if the image is at least 200x200
        if height < 256 and width < 256:
            continue

        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:
            continue
        image = preprocess(pil_image).unsqueeze(0)



        # Calculate the standard deviation and mean across the collected similarity scores
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            #print(image_features.shape)

            # substract seductive_features from image embedding
            image_features = image_features -seductive_features*0.18
            sims= image_features @ emotion_features.T
            simlist_emotions.append(sims)

            #get similarities for gender
            sims= image_features @ person_features.T
            simlist_person.append(sims)

            # get sims for valence
            sims= image_features @ valence_features.T
            simlist_valence.append(sims)

            # get features for face / closeup detection
            facesims= image_features @ face_features.T
            moviesims = image_features @ moviestill_features.T
            titlesims = image_features @ movietitles_features.T
            #print(np.asarray(moviesims)[0][0], np.asarray(facesims)[0][0])

        if np.asarray(titlesims)[0][0]> np.asarray(facesims)[0][0] :
            continue
        if np.asarray(moviesims)[0][0]> np.asarray(facesims)[0][0] :
            continue

        if len(simlist_emotions) > 1:  # Ensure there are enough samples for standard deviation calculation
            simlist_emotions_array = np.concatenate(simlist_emotions, axis=0)
            simlist_person_array = np.concatenate(simlist_person, axis=0)


            std_dev_emotion = np.std(simlist_emotions_array, axis=0) #loaded_stdev_sims_emo  #+
            mean_sims_emotion = np.mean(simlist_emotions_array, axis=0) #loaded_mean_sims_emo #+

            std_dev_person = loaded_stdev_sims_person #np.std(simlist_person_array, axis=0)
            mean_sims_person =  loaded_mean_sims_person  # np.mean(simlist_person_array, axis=0)


            std_dev_valence = loaded_stdev_sims_valence #np.std(simlist_person_array, axis=0)
            mean_sims_valence =  loaded_mean_sims_valence  # np.mean(simlist_person_array, axis=0)





            # Standardize the latest similarity scores
            # not dividing by std_dev_emotion seems to yield a bit better results, no idea why
            sims_standardized = (simlist_emotions[-1] - mean_sims_emotion) #/ std_dev_emotion

            print(sims_standardized.shape, mean_sims_emotion.shape)

            sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
            top_probs_emotion = (sims_tensor).softmax(dim=-1)
            #print(sims_standardized)
            # Get the top 5 predictions
            top_probs_emotion, top_labels = top_probs_emotion.topk(2, dim=-1)
            top_probs_emotion = top_probs_emotion[0].cpu().numpy()  # Move to CPU and convert to numpy
            top_labels_emotion = top_labels[0].cpu().numpy()  # Move to CPU and convert to numpy

            top_probs_emotion = np.atleast_1d(top_probs_emotion).flatten()
            top_labels_emotion = np.atleast_1d(top_labels_emotion).flatten()

            # Standardize the latest similarity scores
            # not dividing by std_dev_person seems to yield a bit better results, no idea why
            sims_standardized = (simlist_person[-1] - mean_sims_person) #/ std_dev_person
            sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
            text_probs_person = (sims_tensor).softmax(dim=-1)
            # Get the top 5 predictions
            text_probs_person, top_labels = text_probs_person.topk(1, dim=-1)
            top_probs_person= text_probs_person[0].cpu().numpy()  # Move to CPU and convert to numpy
            top_labels_person = top_labels[0].cpu().numpy()  # Move to CPU and convert to numpy

            top_probs_person = np.atleast_1d(text_probs_person).flatten()
            top_labels_person = np.atleast_1d(top_labels_person).flatten()

            # Standardize the latest similarity scores
            sims_standardized = (simlist_valence[-1] - mean_sims_valence)  / std_dev_person
            sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
            text_probs_valence = (sims_tensor).softmax(dim=-1)

            text_probs_valence, top_labels = text_probs_valence.topk(2, dim=-1)
            top_probs_valence = text_probs_valence[0].cpu().numpy()  # Move to CPU and convert to numpy
            top_labels_valence = top_labels[0].cpu().numpy()  # Move to CPU and convert to numpy

            top_probs_valence = np.atleast_1d(text_probs_valence).flatten()
            top_labels_valence = np.atleast_1d(top_labels_valence).flatten()



            print("Predicted Gender with Probabilities:")
            for prob, label_idx in zip(top_probs_person, top_labels_person):
                print(f"{person[label_idx]}: {prob * 100:.2f}%")

            print("Predicted Valence with Probabilities:")
            for prob, label_idx in zip(top_probs_valence, top_labels_valence):
                print(f"{valence[label_idx]}: {prob * 100:.2f}%")


            print("Top 3 Predicted Emotions with Probabilities:")

            emotion_text=""
            for prob, label_idx in zip(top_probs_emotion, top_labels_emotion):
                print(f"{emotions[label_idx]}: {prob * 100:.2f}%")

                # Prepare the text to be displayed
                emotion_text +=  emotions[label_idx].split(":")[0]+ "   " # ', '.join([f"{emotions[label_idx]}: {prob * 100:.2f}%" for prob, label_idx in zip(top_probs_emotion, top_labels_emotion)])

            # Overlay the text on the frame
            cv2.putText(frame,
                            emotion_text,
                            (10, frame.shape[0] - 20),  # Position at bottom of the frame
                            font,
                            font_scale,
                            font_color,
                            line_type)


            # Save with high quality
            if save_images==True:
                output_path = f'{folder_name}/B32_prediction_{frame_count}.jpg'  # Path where you want to save the image
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])



            cv2.imshow("Video", frame)
        # Warten auf 2 Sekunden
        #sleep(0.5)

        # Überprüfen, ob der Benutzer das Fenster schließen möchte
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Programm durch Benutzer unterbrochen")

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
