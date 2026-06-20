import numpy as np
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from collections import Counter
import tensorflow as tf
from keras import layers
from keras import Sequential
import hashlib as hl
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

main_path='./podaci/'
img_size=(64,64)
batch_size=64
#ucitavanje podataka i podela na trening i validacioni skup
Xtrain=image_dataset_from_directory(main_path,
                                    subset='training',
                                    validation_split=0.4,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

Xval=image_dataset_from_directory(main_path,
                                  subset='validation',
                                  validation_split=0.4,
                                  image_size=img_size,
                                  batch_size=batch_size,
                                  seed=123)
val_batches=tf.data.experimental.cardinality(Xval)
Xtest=Xval.take((val_batches) // 2)
Xval=Xval.skip((val_batches) // 2)

classes=Xtrain.class_names
print(classes)
#brojanje odabiraka za svaku klasu
class_count={}
for _, labels in Xtrain:
    for label in labels:
        class_name=classes[label.numpy()]
        if class_name in class_count:
            class_count[class_name]+=1
        else:
            class_count[class_name]=1
plt.figure()
plt.bar(class_count.keys(),class_count.values())
plt.xlabel('Klasa')
plt.ylabel('Broj odabiraka')
plt.title('Broj odabiraka klasa po klasi')
plt.show()


plt.figure()

different_class={}
counter=0
#prikazivanje po jednog primerka svake klase
for img, lab in Xtrain.take(1):
    for i in range(len(lab)):
        if classes[lab[i]] not in different_class:
            plt.subplot(1,5,counter+1)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(classes[lab[i]])
            plt.axis('off')
            different_class[classes[lab[i]]]=1
            counter+=1
        if len(different_class)==5:
            break

plt.show()

#podela trening skupa podataka na ulazne(slika) i izlazen(oznaka) podatke
X_array=[]
Y_array=[]
for images, labels in Xtrain:
    X_array.extend(images.numpy())
    Y_array.extend(labels.numpy())
X_array=np.array(X_array)
Y_array=np.array(Y_array)

#podela validacionog skupa podataka na ulazne(slika) i izlazne(oznaka) podatke
"""X_array_val=[]
Y_array_val=[]
for images, labels in Xval:
    X_array_val.extend(images.numpy())
    Y_array_val.extend(labels.numpy())
X_array_val=np.array(X_array_val)
Y_array_val=np.array(Y_array_val)"""

#nalazenje maximalne vrednost broja odabiri medju svim klasama radi balansiranja broja odabiraka klasa
counter = Counter(Y_array)
max_count=max(counter.values())
"""counterVal=Counter(Y_array_val)
max_count_val=max(counterVal.values())"""

#definisanje koje ce nasumicne promene slike dobiti prilikom kreiranja neuralne mreze
data_augmentation=Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_size[0],img_size[1],3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.15),
    ])

X_balanced=[]
Y_balanced=[]

"""X_balanced_val=[]
Y_balanced_val=[]"""
#ovde dolazi do balansiranja klasa, svaka klasa dobija tacno max_count odabiraka iz vec postojeceg skupa odabiraka za tu klasi
#ovo znaci da ce neke slike(nasumicne) da se ponavljaju, ovaj problem cemo u daljem delu koda da resimo
for label,count in counter.items():
    indexes=np.where(Y_array==label)[0]
    new_indexes=np.random.choice(indexes,size=max_count,replace=True)
    X_balanced.extend(X_array[new_indexes])
    Y_balanced.extend([label]*max_count)
X_balanced=np.array(X_balanced)
Y_balanced=np.array(Y_balanced)

"""for label,count in counterVal.items():
    indexes=np.where(Y_array_val==label)[0]
    new_indexes=np.random.choice(indexes,size=max_count,replace=True)
    X_balanced_val.extend(X_array_val[new_indexes])
    Y_balanced_val.extend([label]*max_count)
X_balanced_val=np.array(X_balanced_val)
Y_balanced_val=np.array(Y_balanced_val)"""

#ovde za svaku fotografiju koja se ponavlja u nasem nizu balansiranih odabiraka ce biti izvrsena augmentacija
#augmentovana(izmenjena) fotografija ce se naci na mestu ponovljene fotografije
#ovo radimo jer je lose da imamo iste primerke neke klase
previous_images={}
for ind, img in enumerate(X_balanced):
    if hl.md5(img).hexdigest() in previous_images:
        X_balanced[ind]=data_augmentation(np.expand_dims(img,axis=0))[0]
    else:
        previous_images[hl.md5(img).hexdigest()]=1

"""previous_images_val={}
for ind, img in enumerate(X_balanced_val):
    if hl.md5(img).hexdigest() in previous_images:
        X_balanced_val[ind]=data_augmentation(np.expand_dims(img,axis=0))[0]
    else:
        previous_images[hl.md5(img).hexdigest()]=1
"""

#ovde nas niz fotografija i oznaka(labela) pretvaramo u Dataset objekat
X_balanced_dataset=tf.data.Dataset.from_tensor_slices((X_balanced,Y_balanced))
X_balanced_dataset=X_balanced_dataset.shuffle(len(X_balanced_dataset)).batch(batch_size)

"""X_balanced_dataset_val=tf.data.Dataset.from_tensor_slices((X_balanced_val,Y_balanced_val))
X_balanced_dataset_val=X_balanced_dataset_val.shuffle(len(X_balanced_dataset_val)).batch(batch_size)"""
#ovde vrsimo proveru da vidimo da li nam je broj odabiraka za svaku klasu stvarno jednak
class_count={}
for _, labels in X_balanced_dataset:
    for label in labels:
        class_name=classes[label.numpy()]
        if class_name in class_count:
            class_count[class_name]+=1
        else:
            class_count[class_name]=1
plt.figure()
plt.bar(class_count.keys(),class_count.values())
plt.xlabel('Klasa')
plt.ylabel('Broj odabiraka')
plt.title('Broj odabiraka klasa po klasi')
plt.show()

#ovde vizuelno vrsimo proveru da li je balansiranje odabiraka klase Komarac uspelo kako smo zamislisi
#Zeljeni rezultat je da postoji max_count fotografija komaraca i da fotografije koje se ponavljaju izgledaju
#modifikovano (rotirane su, flipovane su, zumirane su, promenjen je kontrast, osvetljenje) - nasumicno se desavaju ove
#promene
plt.figure(figsize=(10, 16))
i=0
for image, label in X_balanced_dataset.unbatch().as_numpy_iterator():
    if classes[label] == 'Mosquito':
        plt.subplot(16, 54, i + 1)
        plt.imshow(image.astype('uint8'),extent=[0,255,0,255])
        plt.axis('off')
        i=i+1
plt.show()

num_classes=len(classes)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu',kernel_regularizer=l2(0.001) ),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3,padding='same', activation='relu',kernel_regularizer=l2(0.001) ),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu',kernel_regularizer=l2(0.001) ),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu',kernel_regularizer=l2(0.001)),
    layers.Dense(256,activation='relu',kernel_regularizer=l2(0.001)),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss',mode='min', patience=16,verbose=1,restore_best_weights=True)

history=model.fit(X_balanced_dataset,
                  epochs=50,
                  validation_data=Xval,
                  callbacks=[es],
                  verbose=0)

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']


plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()
print('Tacnost modela je: ' +str(100*model.evaluate(Xtest,verbose=0)[1]) +'%')

labels = np.array([])
pred = np.array([])
for img, lab in X_balanced_dataset:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

all_test_images = []
all_test_labels = []
all_pred = []
for img, lab in Xtest:
    all_test_images.extend(img.numpy())
    all_test_labels.extend(lab.numpy())
    all_pred.extend(np.argmax(model.predict(img, verbose=0), axis=1))

all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels)
all_pred = np.array(all_pred)

cm = confusion_matrix(all_test_labels, all_pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()

correctly_classified_indices = np.where(all_test_labels == all_pred)[0]
incorrectly_classified_indices = np.where(all_test_labels != all_pred)[0]

plt.figure(figsize=(15, 3))
for i in range(5):
    idx = correctly_classified_indices[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(all_test_images[idx].astype("uint8"))
    plt.title(f"Stvarna: {classes[int(all_test_labels[idx])]}\nPredviđena: {classes[int(all_pred[idx])]}", fontsize=8)
    plt.axis('off')
plt.suptitle("Primeri uspešne klasifikacije")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 3))
for i in range(5):
    idx = incorrectly_classified_indices[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(all_test_images[idx].astype("uint8"))
    plt.title(f"Stvarna: {classes[int(all_test_labels[idx])]}\nPredviđena: {classes[int(all_pred[idx])]}", fontsize=8)
    plt.axis('off')
plt.suptitle("Primeri neuspešne klasifikacije")
plt.tight_layout()
plt.show()