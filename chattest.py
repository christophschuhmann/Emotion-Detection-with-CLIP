import threading
import cv2
from PIL import Image
import numpy as np
from time import sleep

import tkinter as tk
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

api_key = "xxxxxxxxxxxxxxxx"
mistral_model = "mistral-tiny"

client = MistralClient(api_key=api_key)



import tkinter as tk

import torch
from PIL import Image

import cv2
from PIL import Image
def release_all_cameras(max_cameras=10):
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        cap.release()

release_all_cameras()  # Call this at the beginning of your program

import open_clip
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
model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K")

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
    text_tokenized = tokenizer(text_list)
    print(type(text_tokenized))
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokenized)
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
loaded_mean_sims_valence = np.array([[0.14,0.19]]) #np.load(r'C:\Users\pc\PycharmProjects\realtimeSTT\mean_sims_valence.npy')
loaded_stdev_sims_valence = np.array([[0.05,0.05]]) #np.load(r'C:\Users\pc\PycharmProjects\realtimeSTT\std_dev_sims_valence.npy')

recognized_emotions = []
# font for displaying predictions
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_color = (255, 165, 0)  # Orange color in BGR
line_type = 2



def emotion_recognition_process():
    # Initialize emotion_text before the loop
    emotion_text = ""
    cap = cv2.VideoCapture(0)


    try:
        while True:
            # get image from webcam
            ret, frame = cap.read()

            # Überprüfen, ob das Bild erfolgreich aufgenommen wurde
            if not ret:
                #print("Fehler beim Erfassen des Bildes")
                break

            print("##########################")

            # detect face
            face = get_largest_face_from_image(frame)
            frame = face
            if frame is None:
                continue
            height, width = frame.shape[:2]

            # Check if the image is at least 200x200
            if height < 100 and width < 100:
                continue


            try:
                pil_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            except:
                continue
            image = preprocess(pil_image).unsqueeze(0)



            # Calculate the standard deviation and mean across the collected similarity scores
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # substract seductive_features from image embedding
                image_features = image_features - seductive_features * 0.18
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
                # not dividing by std_dev_emotion seems to yield a bit better results, no idea why
                sims_standardized = (simlist_emotions[-1] - mean_sims_emotion) #/ std_dev_emotion
                #sims_standardized = simlist_emotions[-1]

                #print(sims_standardized.shape, mean_sims_emotion.shape)
                sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
                top_probs_emotion = (sims_tensor).softmax(dim=-1)
                #print(sims_standardized)
                # Get the top 5 predictions
                top_probs_emotion, top_labels = top_probs_emotion.topk(1, dim=-1)
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


                sims_standardized = (simlist_valence[-1] - mean_sims_valence)  / std_dev_person
                sims_tensor = torch.tensor(sims_standardized).to(image_features.device)
                text_probs_valence = (sims_tensor).softmax(dim=-1)

                text_probs_valence, top_labels = text_probs_valence.topk(2, dim=-1)
                top_probs_valence = text_probs_valence[0].cpu().numpy()  # Move to CPU and convert to numpy
                top_labels_valence = top_labels[0].cpu().numpy()  # Move to CPU and convert to numpy

                top_probs_valence = np.atleast_1d(text_probs_valence).flatten()
                top_labels_valence = np.atleast_1d(top_labels_valence).flatten()



                #print("Predicted Gender with Probabilities:")
                #for prob, label_idx in zip(top_probs_person, top_labels_person):
                    #print(f"{person[label_idx]}: {prob * 100:.2f}%")
                    #recognized_emotions.append(f"{person[label_idx]}: {prob * 100:.2f}%")

                #print("Predicted Valence with Probabilities:")
                #for prob, label_idx in zip(top_probs_valence, top_labels_valence):
                    #print(f"{valence[label_idx]}: {prob * 100:.2f}%")
                #    recognized_emotions.append(f"{valence[label_idx]}: {prob * 100:.2f}%")

                emotion_text = ""
                for prob, label_idx in zip(top_probs_emotion, top_labels_emotion):
                    print(f"{emotions[label_idx]}: {prob * 100:.2f}%")
                    recognized_emotions.append(emotions[label_idx].split(":")[0] + "   ")

                    # Prepare the text to be displayed
                    emotion_text += emotions[label_idx].split(":")[0] + "   "


                # Overlay the text on the frame
                cv2.putText(frame,
                            emotion_text,
                            (10, frame.shape[0] - 20),  # Position at bottom of the frame
                            font,
                            font_scale,
                            font_color,
                            line_type)
                cv2.imshow('Webcam-Bild', frame)

                sleep(1)

            # Überprüfen, ob der Benutzer das Fenster schließen möchte
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Programm durch Benutzer unterbrochen")

        # Ressourcen freigeben
        cap.release()
        cv2.destroyAllWindows()





# Create and start the emotion recognition thread
emotion_thread = threading.Thread(target=emotion_recognition_process)
emotion_thread.start()
#chat_history=r"CHATBOT: Hello, how can I help you?\nUSER: "
#print(chat_history)
'''
# Main process logic
try:
    while True:
        reply = input("")
        # Access recognized emotions
        if recognized_emotions:
            latest_emotions = '\n'.join(recognized_emotions[-6:])
            print("Recognized Emotions:", latest_emotions)
        # ... rest of your main process logic ...
except KeyboardInterrupt:
    print("Main process interrupted")
finally:
    # Optionally join the thread if needed
    emotion_thread.join()
'''



def generate_text():
    # Retrieve and print the values from the text boxes
    temperature = float( temperature_entry.get() )
    #min_tokens = int( min_tokens_entry.get() )
    max_tokens = int( max_tokens_entry.get() )

    story_so_far = textbox1.get("1.0", tk.END)

    global instruction
    instruction = textbox2.get("1.0", tk.END)

    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")

    # use the last 5 recognized emotions
    latest_emotions = '\n'.join(recognized_emotions[-5:])
    #print("Recognized Emotions:", latest_emotions)
    textbox3.delete(1.0, tk.END)
    textbox3.insert(tk.END, "Detected emotion: "+latest_emotions)

    messages = [
        ChatMessage(role="system", content="You are a a very kind, empathetic, patient, humble, altruistic, emotionally intelligent chatbot. You love to help the user with whatever they ask from you, as long as it is ethically okay and legal. You also love to chat with users, socialize and understand their thoughts and emotions. You are very humble about your perceptions of their emotions and acknowledge that your perceptions of their emotional expressions could be wrong. Do not tell explicitly about your insecurities about the user's emotions, just keep a kind and humble style and tone in your replies. You are repectful and have high social intelligence."),
        ChatMessage(role="user", content=f"THE CHAT HISTORY SO FAR:\n{story_so_far}\nTHE EMOTIONS THE USER SEEMS TO FEEL AT THE MOMENT:{latest_emotions}\nINSTRUCTION:\n{instruction}")
    ]

    # No streaming
    chat_response = client.chat(
        model=mistral_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,

    )
    output = chat_response.choices[0].message.content
    # Replace text in the main text box as an example action
    textbox1.delete(1.0, tk.END)
    textbox1.insert(tk.END, story_so_far+"\n"+output+"\nUSER: ")
    textbox1.see(tk.END)

    file_path = "chat_so_far.txt"
    # Open the file in write mode ('w') which will overwrite the file if it already exists
    #with open(file_path, 'w') as file:
    #    file.write(story_so_far+"\n"+output)
    print(chat_response.choices[0].message.content)


def create_gui():
    # Create a new tkinter window
    window = tk.Tk()
    window.title("Chat")

    # Set window size
    window.geometry('900x600')  # width x height

    # Create the first text box with a scrollbar
    global textbox1
    textbox1 = tk.Text(window, height=15)
    scrollbar1 = tk.Scrollbar(window, command=textbox1.yview)
    textbox1.configure(yscrollcommand=scrollbar1.set)
    textbox1.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)

    # Create the second text box with a scrollbar
    global textbox2
    textbox2 = tk.Text(window, height=5)
    scrollbar2 = tk.Scrollbar(window, command=textbox2.yview)
    textbox2.configure(yscrollcommand=scrollbar2.set)
    textbox2.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)


    # Create the second text box with a scrollbar
    global textbox3
    textbox3 = tk.Text(window, height=2)
    scrollbar3 = tk.Scrollbar(window, command=textbox2.yview)
    textbox3.configure(yscrollcommand=scrollbar2.set)
    textbox3.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)

    # Create frame for inputs and button
    frame = tk.Frame(window)
    frame.pack(side=tk.BOTTOM, pady=20)

    # Create labels and entries for temperature, min tokens, and max tokens
    tk.Label(frame, text="Temperature:").pack(side=tk.LEFT)
    global temperature_entry
    temperature_entry = tk.Entry(frame, width=5)
    temperature_entry.pack(side=tk.LEFT, padx=(0, 20))
    temperature_entry.insert(tk.END, 0.6)
    '''
    tk.Label(frame, text="Min tokens:").pack(side=tk.LEFT)
    global min_tokens_entry
    min_tokens_entry = tk.Entry(frame, width=5)
    min_tokens_entry.pack(side=tk.LEFT, padx=(0, 20))
    min_tokens_entry.insert(tk.END, 50)
    '''
    tk.Label(frame, text="Max tokens:").pack(side=tk.LEFT)
    global max_tokens_entry
    max_tokens_entry = tk.Entry(frame, width=5)
    max_tokens_entry.pack(side=tk.LEFT)
    max_tokens_entry.insert(tk.END, 100)

    # Create the generate button
    generate_button = tk.Button(frame, text="Generate", command=generate_text)
    generate_button.pack(side=tk.RIGHT, padx=20)

    # Initialize the text boxes with default text
    story_so_far = "CHATBOT: Hello, how can I help you?\nUSER: "

    # The path to the file where you want to write the string
    file_path = "chat_so_far.txt"
    try:
        with open(file_path, 'r') as file:
            story_so_far = file.read()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    instruction = "Reply to the user given the previous chat history. If appropriate and helpful, try to involve your perceptions of the user's emotions in your reply, but only if this seems kind, helpful and appropriate to you. Be very modest and humble in your assumptions about the user's emotions and consider that you could perceive them wrong. If you think the user could be in an unpleasant emotion, subtly offer your emotional support, if appropriate. Keep your reply brief, 1-3 sentences. Fulfilling the user's requests will always be more important to you than his or her emotions. Be helpful and useful. Begin your reply with 'CHATBOT: '"

    textbox1.insert(tk.END, story_so_far)
    textbox2.insert(tk.END, instruction)

    # Start the GUI
    window.mainloop()


# Create and start the GUI
create_gui()

