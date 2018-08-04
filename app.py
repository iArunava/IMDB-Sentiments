from tkinter import *


root = Tk()
root.geometry("460x460")
root.title('Sentiment Analysis using IMDB Sentiments')

l1 = Label(root, text='Write your Review here:')
l1.pack(side=TOP)

text = Text(root)
text.pack()

def predict_btn_clicked():
    senti.set('sdfsd')

Button(root, text='Predict Sentiment', command=predict_btn_clicked).pack()

l1 = Label(root, text='The Sentiment is: ')
l1.pack(side=LEFT)

senti = StringVar()
Label(root, textvariable=senti, fg='green').pack(side=LEFT)

root.mainloop()
