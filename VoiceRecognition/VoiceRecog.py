# @Time: 1/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: VoiceRecog.py

import speech_recognition as sr

filename = "C:\\Users\\lenovo\\Downloads\\Recording.wav"

r = sr.Recognizer ()

src = sr.AudioFile ( filename )

with src as source :
    audio = r.record ( source )

text = r.recognize_google ( audio , language = "en-US" , show_all = True )
print ( text [ 'alternative' ] [ 0 ] [ 'transcript' ] )