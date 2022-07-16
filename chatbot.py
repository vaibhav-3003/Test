# Importing Modules
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD

#Making empty lists
words=[]
classes=[]
documents=[]

ignore_words=['?','!','.']
data_file=open('intents.json').read()
intents=json.loads(data_file)
lemmatizer=WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenization
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        # add the tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Converting the list into set for removing the duplicate items
words=list(set(words))

#lemmatize the classes also
classes=list(set(classes))

#Now save the data into a pickle file
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#Training and Creating Model
training=[]
output_empty=[0]*len(classes) #[0,0,0,0,0,0,...... upto len(classes)]
for doc in documents:
    bag=[]
    pattern_words=doc[0]
    #Lemmatize the pattern_words
    pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    #Create Bag
    for w in words:
        if w in pattern_words:
            bag.append(1) 
        else:
            bag.append(0)

    #Now Create output
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1

    training.append([bag,output_row])

random.shuffle(training)
#changing the training data into numpy array
training=np.array(training)
train_x=list(training[:,0]) #contains all the bags
train_y=list(training[:,1]) #contains all the outputs

# Make Sequential model
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# compiling the Model & define an optimizer function
# optimizer function -> To reduce the losses by changing the weights and bias
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True) #Stochastic Gradient Descent
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# Now pass the training data to the model
mfit=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',mfit)
print('Created my first model') 