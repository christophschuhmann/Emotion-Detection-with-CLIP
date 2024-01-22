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



seductive_features = encode_text(["Seductive/Provocative: 'seductive', 'titillating', 'sultry', 'erotic', 'salacious', 'libidinous',  'carnal', 'desirous', 'lecherous', 'lust-driven', 'prurient', 'lustful', 'ravishing',  'irresistible', 'provoked', 'craving'"])
print(seductive_features.shape)






loaded_mean_sims_emo = np.array([[0.15599771 ,0.1511304 , 0.16875783, 0.19157822, 0.17763473 ,0.14812051,
  0.16381973, 0.15677759, 0.14437722, 0.17471613, 0.16821687, 0.17212224,
  0.1740183 , 0.15789438 ,0.16779166 ,0.17950825, 0.15191281, 0.16258015,
  0.14764176, 0.15238074]])

loaded_stdev_sims_emo = np.array([[0.0288077 , 0.02833298, 0.02468313, 0.03723702, 0.02814594, 0.0304083,
  0.03004773, 0.03482676, 0.03983417, 0.04059409, 0.03653273, 0.03492723,
  0.03197159, 0.03464679, 0.03582928, 0.03437715, 0.0293751,  0.02760981,
  0.03604855]] )


loaded_mean_sims_person = np.array([[0.18671854,0.1657077]])
loaded_stdev_sims_person= np.array([[0.0340511,0.03612482]]) #



loaded_mean_sims_valence = np.array([[0.155,0.19]])
loaded_stdev_sims_valence = np.array([[0.05,0.05]]) 

# Webcam initialisieren
cap = cv2.VideoCapture(0)

try:
    while True:
        # Bild von der Webcam erfassen
        ret, frame = cap.read()

        # Überprüfen, ob das Bild erfolgreich aufgenommen wurde
        if not ret:
            print("Fehler beim Erfassen des Bildes")
            break


        cv2.imshow('Webcam-Bild', frame)

        face= get_largest_face_from_image(frame)
        try:
            pil_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        except:
            continue
        image = preprocess(pil_image).unsqueeze(0)



        # Calculate the standard deviation and mean across the collected similarity scores
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            #print(image_features.shape)
            image_features = image_features -seductive_features*0.15
            sims= image_features @ emotion_features.T
            simlist_emotions.append(sims)

            sims= image_features @ person_features.T
            simlist_person.append(sims)

            sims= image_features @ valence_features.T
            simlist_valence.append(sims)

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
            sims_standardized = (simlist_emotions[-1] - mean_sims_emotion) #/ std_dev_emotion
            #sims_standardized = simlist_emotions[-1]

            print(sims_standardized.shape, mean_sims_emotion.shape)
            sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
            top_probs_emotion = (sims_tensor).softmax(dim=-1)
            print(sims_standardized)
            # Get the top 5 predictions
            top_probs_emotion, top_labels = top_probs_emotion.topk(2, dim=-1)
            top_probs_emotion = top_probs_emotion[0].cpu().numpy()  # Move to CPU and convert to numpy
            top_labels_emotion = top_labels[0].cpu().numpy()  # Move to CPU and convert to numpy

            top_probs_emotion = np.atleast_1d(top_probs_emotion).flatten()
            top_labels_emotion = np.atleast_1d(top_labels_emotion).flatten()


            sims_standardized = (simlist_person[-1] - mean_sims_person) #/ std_dev_person
            sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
            text_probs_person = (sims_tensor).softmax(dim=-1)
            # Get the top 5 predictions
            text_probs_person, top_labels = text_probs_person.topk(1, dim=-1)
            top_probs_person= text_probs_person[0].cpu().numpy()  # Move to CPU and convert to numpy
            top_labels_person = top_labels[0].cpu().numpy()  # Move to CPU and convert to numpy

            top_probs_person = np.atleast_1d(text_probs_person).flatten()
            top_labels_person = np.atleast_1d(top_labels_person).flatten()


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
            for prob, label_idx in zip(top_probs_emotion, top_labels_emotion):
                print(f"{emotions[label_idx]}: {prob * 100:.2f}%")


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
