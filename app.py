from tkinter import *
from predict import *

tfidf_vect, selector = get_ready()

root = Tk()
root.geometry("460x460")
root.title('Sentiment Analysis using IMDB Sentiments')

l1 = Label(root, text='Write your Review here:')
l1.pack(side=TOP)

text = Text(root)
text.pack()

def predict_btn_clicked():
    ttX = text.get('1.0', 'end');
    prediction = predict(ttX, tfidf_vect, selector)
    senti.set('sdfsd')

Button(root, text='Predict Sentiment', command=predict_btn_clicked).pack()

l1 = Label(root, text='The Sentiment is: ')
l1.pack(side=LEFT)

senti = StringVar()
Label(root, textvariable=senti, fg='green').pack(side=LEFT)

root.mainloop()
