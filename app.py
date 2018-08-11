from tkinter import *
from predict import *

tfidf_vect, selector, model = get_ready()
threshold = .6

root = Tk()
root.geometry("660x460")
root.title('Sentiment Analysis using IMDB Sentiments')

l1 = Label(root, text='Write your Review here:')
l1.pack(side=TOP)

text = Text(root)
text.pack()

def predict_btn_clicked():
    ttX = text.get('1.0', 'end');
    prediction = predict(ttX, tfidf_vect, selector, model)
    sentiment = 'Positive' if prediction[0][0] > threshold else 'Negetive'
    senti.set(str(sentiment) +
              ' || Probability of the review being a positive sentiment is: ' +
              str(prediction[0][0]))

Button(root, text='Predict Sentiment', command=predict_btn_clicked).pack()

l1 = Label(root, text='The Sentiment is: ')
l1.pack(side=LEFT)

senti = StringVar()
Label(root, textvariable=senti, fg='green').pack(side=LEFT)

root.mainloop()
