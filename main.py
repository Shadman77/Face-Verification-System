from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from numpy import expand_dims
import numpy as np
from keras.models import load_model

# load images from the directory(verify_images)
def getImages(directory):
    images = list()
    # for all images in the directory, we expect two images
    for imageName in listdir(directory):
        # load the image
        image = Image.open(directory + '/' + imageName)
        # add to the list of images
        images.append(image)

    return images


# extract faces for all images, two in our case
def extractFaces(images):
    faces = list()
    # create the detector from mtcnn
    detector = MTCNN()
    
    print('Shape of the faces found are:')
    for image in images:
        # convert the image to RGB, if not already RGB
        image = image.convert('RGB')
        # convert image to array
        image_pixels = asarray(image)
        # detect all the faces in the image
        result = detector.detect_faces(image_pixels)
        # if we find at least one face we save the first face (usually the largest face in the image in the first one)
        if len(result) > 0:
            # extract the bounding box from the first face
            x1, y1, width, height = result[0]['box']
            # prevent negative values
            x1, y1 = abs(x1), abs(y1)
            # extract the face
            face = image_pixels[y1:y1+height, x1:x1+width]
            # resize pixels to the correct size for facenet
            face_image = Image.fromarray(face)
            face_image = face_image.resize((160, 160))
            # convert face to array
            face_pixels = asarray(face_image)
            # save the face in the list of faces
            faces.append(face_pixels)
            # show the face dimention
            print(face_pixels.shape)

    return faces

# get embeddings using facenet
def getEmbeddings(model, faces):
    embeddings = list()
    for face in faces:
        # scale pixel values
        face = face.astype('float32')
        # get mean
        mean = face.mean()
        # get std
        std = face.std()
        # standardize pixel values across channels (global)
        face = (face - mean) / std
        # transform face into one sample that is expand the dimension
        samples = expand_dims(face, axis=0)
        # make prediction to get embedding
        embedding = model.predict(samples)
        # add embedding to the list of embeddings
        embeddings.append(embedding)

    return embeddings


# verify if two faces are of the same persion
def verify(faces):
    model = load_model('facenet_keras.h5')
    embeddings = getEmbeddings(model, faces)
    print("Number of embeddings found: %d" % len(embeddings))

    # get l2 distance
    distance = np.linalg.norm(embeddings[0] - embeddings[1])
    print('L2 distance is : %f' % distance)

    # get result, we choosed threshold is 11
    if distance < 11:
        return 'Same person'
    else:
        return 'Different person'

    


def main():
    # load all images
    images = getImages('verify_images')
    print("Number of images found : %d" % len(images))

    # extract faces from all images
    faces = extractFaces(images)
    print("Number of faces found : %d" % len(faces))
    # show both the faces, this won't work if there are not exactly two images in the verify_images folder
    fig=pyplot.figure()
    for i in range(len(faces)):
        ax=fig.add_subplot(1,2,i+1)
        ax.axis('off')
        ax.imshow(faces[i])
    fig.suptitle('Found faces')
    pyplot.show()

    # verify faces
    verdict = verify(faces)

    # show final result
    fig=pyplot.figure()
    for i in range(len(images)):
        ax=fig.add_subplot(1,2,i+1)
        ax.axis('off')
        ax.imshow(images[i])
    fig.suptitle(verdict)
    pyplot.show()


if __name__== "__main__":
    main()