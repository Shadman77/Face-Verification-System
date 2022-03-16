# Class imports
from keras.models import load_model
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import expand_dims
import numpy as np

class FacenetTensorflow:

    def __init__(self, facenet_model_path = 'facenet_keras.h5', face_comp_thres = 0.8):
        self.__mtcnn = MTCNN()
        self.__model = load_model(facenet_model_path)
        self.__face_comp_thres = face_comp_thres


    def __extractFaces(self, images):
        faces = list()
        # create the detector from mtcnn
        
        for image in images:
            # convert the image to RGB, if not already RGB
            image = image.convert('RGB')
            # convert image to array
            image_pixels = asarray(image)
            # detect all the faces in the image
            result = self.__mtcnn.detect_faces(image_pixels)
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

        return faces

    # get embeddings using facenet
    def __getEmbeddings(self, faces):
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
            embedding = self.__model.predict(samples)
            # add embedding to the list of embeddings
            embeddings.append(embedding)
        
        return embeddings

    def verify(self, images):
        faces = self.__extractFaces(images)
        embeddings = self.__getEmbeddings(faces)

        # get l2 distance
        distance = np.linalg.norm(embeddings[0] - embeddings[1])
        print('L2 distance is : %f' % (distance/10))

        # get result, we choosed threshold is 11
        if distance/10 < self.__face_comp_thres:
            return 'Same person'
        else:
            return 'Different person'

if __name__ == "__main__":
    from os import listdir
    def getImages(directory):
        images = list()
        # for all images in the directory, we expect two images
        for imageName in listdir(directory):
            # load the image
            image = Image.open(directory + '/' + imageName)
            # add to the list of images
            images.append(image)

        return images


    images = getImages('verify_images')
    facenetTensorflow = FacenetTensorflow()
    
    import time
    start_time = time.time()
    for i in range(1):
        print(facenetTensorflow.verify(images))
    end_time = time.time()
    print("time taken = {}".format(end_time - start_time))
    print("avarage time taken = {}".format((end_time - start_time)/10))


