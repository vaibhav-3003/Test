from tkinter import *
from PIL import Image,ImageTk
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import pyttsx3 as pp
import numpy as np
import speech_recognition as S

#Initialize 
engine=pp.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
rate=engine.getProperty('rate')
engine.setProperty('rate',145)

intents=json.loads(open('intents.json').read())
lemmatizer=WordNetLemmatizer()
model=load_model('chatbot_model.h5')
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

def speak(word):
    engine.say(word)
    engine.runAndWait()

# creating a function for enter
def enter_func(event):
    send_mic_btn.invoke()

def bow(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag=[0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
    return (np.array(bag))

def predict_class(sentence):
    sentence_bag=bow(sentence)
    res=model.predict(np.array([sentence_bag]))[0]
    Error_Threshold=0.25
    results=[[i,r] for i,r in enumerate(res) if r>Error_Threshold]
    #sort by probability
    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def getResponse(ints):
    tag=ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
   ints=predict_class(msg)
   res=getResponse(ints)
   return res

def take_query():
    TextEntryBox.delete("1.0","end")
    sr = S.Recognizer()
    sr.energy_threshold = 800
    with S.Microphone() as m:
        audio=sr.listen(m)
        query=sr.recognize_google(audio,language='eng-in')
        TextEntryBox.insert("end",query)
        mic_send()

def send():
    msg=TextEntryBox.get("1.0","end-1c").strip()
    TextEntryBox.delete('1.0','end')
    if msg !='':
        ChatHistory.config(state=NORMAL)
        ChatHistory.image_create("end",image=user_image)
        ChatHistory.insert("end","  "+msg+"\n\n")
        res=chatbot_response(msg)
        ChatHistory.image_create("end",image=jarvis_image)
        ChatHistory.insert("end","  "+res+"\n\n")
        ChatHistory.config(state=DISABLED)
        ChatHistory.yview("end")

def mic_send():
    msg=TextEntryBox.get("1.0","end-1c").strip()
    TextEntryBox.delete('1.0','end')
    if msg !='':
        ChatHistory.config(state=NORMAL)
        ChatHistory.image_create("end",image=user_image)
        ChatHistory.insert("end","  "+msg+"\n\n")
        res=chatbot_response(msg)
        ChatHistory.image_create("end",image=jarvis_image)
        ChatHistory.insert("end","  "+res+"\n\n")
        speak(res)
        ChatHistory.config(state=DISABLED)
        ChatHistory.yview("end")
        if res[-1] == "?":
            take_query()

def move_app(e):
    base.geometry(f'+{e.x_root}+{e.y_root}')       
def cross():
    base.destroy()
def click(*args):
    send_mic_btn.config(image=send_button_img,command=send)
    if TextEntryBox.get("1.0","1.13")=="Enter text...":
        TextEntryBox.delete("1.0", 'end')

def click2(*args):
    send_mic_btn.config(image=send_button_img2, command=send)
    if TextEntryBox.get("1.0", "1.13") == "Enter text...":
        TextEntryBox.delete("1.0", 'end')

def leave(*args):
    if len(TextEntryBox.get("1.0","end"))==1:
        TextEntryBox.insert("end","Enter text...")
        send_mic_btn.config(image=new_jarvis_voice_icon,command=take_query)
        base.focus()
def leave2(*args):
    if len(TextEntryBox.get("1.0","end"))==1:
        TextEntryBox.insert("end","Enter text...")
        send_mic_btn.config(image=new_jarvis_voice_icon2,command=take_query)
        base.focus()

# Normal Mode
def normal_mode():
    toggle_btn.config(image=toggle_btn_img,bg="#aaf0d1",command=dark_mode)
    frame_label.config(image=frame_photo)
    ChatHistory.config(bg="#aaf0d1",fg="#003A14")
    TextEntryBox.config(bg="#aaf0d1",fg="#003A14")
    cross_button.config(image=new_cross_photo,bg="#0B6545")
    send_mic_btn.config(bg="#aaf0d1",image=new_jarvis_voice_icon)
    send_mic_btn.place(x=290, y=410)
    TextEntryBox.bind("<Button-1>", click)
    TextEntryBox.bind("<Leave>", leave)

# Dark Mode
def dark_mode():
    toggle_btn.config(image=toggle_btn_img2,bg="#282727",command=normal_mode)
    frame_label.config(image=frame_photo2)
    ChatHistory.config(bg="#282727",fg="white")
    TextEntryBox.config(bg="#282727",fg="white")
    cross_button.config(image=new_cross_photo2,bg="#545050")
    send_mic_btn.config(bg="#282727",image=new_jarvis_voice_icon2)
    send_mic_btn.place(x=290,y=415)
    TextEntryBox.bind("<Button-1>", click2)
    TextEntryBox.bind("<Leave>", leave2)

base=Tk()
base.geometry('328x451')
base.overrideredirect(1)
base.wm_attributes("-transparentcolor","grey")

frame_photo=PhotoImage(file='mint-interface (1).png')
frame_photo2=PhotoImage(file='dark-mode.png')

frame_label=Label(base,border=0,bg='grey',image=frame_photo)
frame_label.pack(fill=BOTH,expand=True)

frame_label.bind("<B1-Motion>",move_app)

#chat history textview
ChatHistory=Text(base,bd=0,bg="#aaf0d1",font=("Varela Round",11,"bold"),fg="#003A14")
ChatHistory.config(state=DISABLED)

cross_button_image=Image.open('cross (2).png')
resized=cross_button_image.resize((20,20))
new_cross_photo=ImageTk.PhotoImage(resized)

cross_button_image2=Image.open('close (1).png')
resized=cross_button_image2.resize((14,14))
new_cross_photo2=ImageTk.PhotoImage(resized)

cross_button=Button(base,image=new_cross_photo,bd=0,bg="#0B6545",command=cross)
cross_button.place(x=290,y=22)

# Send Button
send_button_img=PhotoImage(file="send.png")

send_button_image2=Image.open('send (1).png')
resized=send_button_image2.resize((18,18))
send_button_img2=ImageTk.PhotoImage(resized)

# Microphone button
jarvis_voice_icon=Image.open('microphone (2).png')
resized=jarvis_voice_icon.resize((25,25))
new_jarvis_voice_icon=ImageTk.PhotoImage(resized)

jarvis_voice_icon2=Image.open('microphone (3).png')
resized=jarvis_voice_icon2.resize((18,18))
new_jarvis_voice_icon2=ImageTk.PhotoImage(resized)

send_mic_btn=Button(base,image=new_jarvis_voice_icon,bd=0,bg="#aaf0d1",command=take_query)
send_mic_btn.place(x=290,y=410)

# Text entrybox
TextEntryBox=Text(base,font=("Inter",11),bg="#aaf0d1",bd=0,fg="#003A14")
TextEntryBox.insert("end",'Enter text...')
TextEntryBox.bind("<Button-1>", click)
TextEntryBox.bind("<Leave>", leave)

ChatHistory.place(x=5,y=89,height=318,width=317)
TextEntryBox.place(x=17,y=415,height=30,width=240)

# Toggle Button
toggle_btn_img=PhotoImage(file="toggle-button.png")
toggle_btn_img2=PhotoImage(file="toggle-button (1).png")

toggle_btn=Button(base,image=toggle_btn_img,bd=0,bg="#aaf0d1",command=dark_mode)
toggle_btn.place(x=280,y=60)

# User Image
user_image=Image.open('user (1).png')
resized=user_image.resize((22,22))
user_image=ImageTk.PhotoImage(resized)
# Dark mode user image
user_image2=Image.open('user (2).png')
resized=user_image2.resize((22,22))
user_image2=ImageTk.PhotoImage(resized)

jarvis_image=Image.open('chatbot (2).png')
resized=jarvis_image.resize((22,22))
jarvis_image=ImageTk.PhotoImage(resized)

# bind the main window with enter key
base.bind('<Return>',enter_func)
base.mainloop()