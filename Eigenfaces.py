import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

from sklearn.svm import SVC

# image size
N = 64

# Define the classifier in clf - Try a Support Vector Machine with C = 0.025 and a linear kernel
# DON'T change this!
clf = SVC(kernel="linear", C=0.025)


def create_database_from_folder(path):
    '''
    DON'T CHANGE THIS METHOD.
    If you run the Online Detection, this function will load and reshape the
    images located in the folder. You pass the path of the images and the function returns the labels,
    training data and number of images in the database
    :param path: path of the training images
    :return: labels, training images, number of images
    '''
    labels = list()
    filenames = np.sort(path)
    num_images = len(filenames)
    train = np.zeros((N * N, num_images))
    for n in range(num_images):
        img = cv2.imread(filenames[n], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (N, N))
        assert img.shape == (N, N), 'Image {0} of wrong size'.format(filenames[n])
        train[:, n] = img.reshape((N * N))
        labels.append(filenames[n].split("eigenfaces\\")[1].split("_")[0])
    print('Database contains {0} images'.format(num_images))
    labels = np.asarray(labels)
    return labels, train, num_images


def process_and_train(labels, train, num_images, h, w):
    '''
    Calculate the essentials: the average face image and the eigenfaces.
    Train the classifier on the eigenfaces and the given training labels.
    :param labels: 1D-array
    :param train: training face images, 2D-array with images as row vectors (e.g. 64x64 image ->  4096 vector)
    :param num_images: number of images, int
    :param h: height of an image
    :param w: width of an image
    :return: the eigenfaces as row vectors (2D-array), number of eigenfaces, the average face
    '''

    # TODO:Compute the average face --> calculate_average_face()
    averageFace = calculate_average_face(train=train)
    # TODO:calculate the maximum number of eigenfaces
    numOfEigenfaces = num_images - 1
    # TODO:calculate the eigenfaces --> calculate_eigenfaces()
    eigenfaces = calculate_eigenfaces(train=train, avg=averageFace, num_eigenfaces=numOfEigenfaces, h=h, w=w)
    # TODO:calculate the coefficients/features for all images --> get_feature_representation()
    X_train = get_feature_representation(train, eigenfaces, averageFace, numOfEigenfaces)
    # TODO:train the classifier using the calculated features
    clf.fit(X_train, labels)
    return eigenfaces, numOfEigenfaces, averageFace


def calculate_average_face(train):
    '''
    Calculate the average face using all training face images
    :param train: training face images, 2D-array with images as row vectors
    :return: average face, 1D-array shape(#pixels)
    '''

    # bsp:
    #   176 images, each in 64 * 64 pixels ---> <train> in size (4096 x 176), each image exist as a row vector
    #   avg face is a matrix in size (1 x 4096)
    return np.mean(a=train, axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces, h, w):
    '''
    Calculate the eigenfaces from the given training set using SVD
    :param train: training face images, 2D-array with images as row vectors
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to return from the computed SVD
    :param h: height of an image in the training set
    :param w: width of an image in the training set
    :return: the eigenfaces as row vectors, 2D-array --> shape(num_eigenfaces, #pixel of an image)
    '''

    # TODO:subtract the average face from every training sample
    X = np.zeros(np.shape(train))
    num_of_images = len(train)
    for i in range(num_of_images):
        X[i] = train[i] - avg
    # TODO:compute the eigenfaces using svd.
    #  You might have to swap the axes so that the images are represented as column vectors
    #  represent your eigenfaces as row vectors in a 2D-matrix & crop it to the requested amount of eigenfaces
    X = np.swapaxes(X, axis1=0, axis2=1)
    u, s, v = np.linalg.svd(X)
    u = u.T
    # print("The size of U is: ")
    # print(np.shape(u)), 4096 * 4096,
    # print("The size of V is: ")
    # print(np.shape(v)), 176 * 176
    # print("The size of S is: ")
    # print(np.shape(s))
    # u: Left singular, 4096 * 4096,
    # s: singular values for every matrix,
    # v: right singular, 176 * 176
    eigenface = u[:num_eigenfaces]
    # eigenvalue = s[num_eigenfaces] ** 2
    imageShape = (h, w)
    # TODO:plot few eigenfaces to check whether you're using the right axis,
    #      comment out when submitting your exercise via studOn
    for i in range(0, 25):
        plt.subplot(5, 5, i+1)
        plt.imshow(np.reshape(u[:, i], imageShape), 'gray')
        plt.axis('off')
    plt.show()

    return eigenface


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    '''
    For all images, compute their eigenface-coefficients with respect to the given amount of eigenfaces
    :param images: 2D-matrix with a set of images as row vectors, shape (#images, #pixels)
    :param eigenfaces: 2D-array with eigenfaces as row vectors, shape(#pixels, #pixels)
                       -> only use the given number of eigenfaces
    :param avg: average face, 1D-array
    :param num_eigenfaces: number of eigenfaces to compute coefficients for
    :return: coefficients/features of all training images, D-matrix2 (#images, #used eigenfaces)
    '''

    # TODO:compute the coefficients for all images and save them in a 2D-matrix
    # 1. iterate through all images (one image per row)
    # 1.1 compute the zero mean image by subtracting the average face
    # 1.2 compute the image's coefficients for the expected number of eigenfaces
    zero_mean = np.zeros(np.shape(images))
    eigenface = eigenfaces[:num_eigenfaces]
    # print(np.shape(eigenface))
    # !WARNING: not [:,num], cuz we need a matrix to represent all the eigenfaces
    for i in range(len(images)):
        zero_mean[i] = images[i] - avg
    coefficients = np.dot(zero_mean, eigenface.T)
    # print(np.shape(coefficients))
    return coefficients


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Reconstruct the given image by weighting the eigenfaces according to their coefficients
    :param img: input image to be reconstructed, 1D array
    :param eigenfaces: 2D array with all available eigenfaces as row vectors
    :param avg: the average face image, 1D array
    :param num_eigenfaces: number of eigenfaces used to reconstruct the input image
    :param h: height of a original image
    :param w: width of a original image
    :return: the reconstructed image, 2D array (shape of a original image)
    '''
    # reshape the input image to fit in the feature helper method
    image = np.reshape(a=img, newshape=(1, h * w))
    # compute the coefficients to weight the eigenfaces --> get_feature_representation()
    coefficients = get_feature_representation(images=image, eigenfaces=eigenfaces,
                                              avg=avg, num_eigenfaces=num_eigenfaces)
    # use the average image as starting point to reconstruct the input image
    recon_images = np.zeros((num_eigenfaces + 1, h * w))
    recon_images[0, :] = avg  # starting point
    temp = avg # reconstructed image with corresponding eigenface and coefficients
    # TODO:reconstruct the input image using the coefficients
    for i in range(1, num_eigenfaces + 1):
        temp += coefficients[:, i-1] * eigenfaces[i-1, :]
        recon_images[i, :] = temp
    # reshape the reconstructed image back to its original shape
    reconstructed_image = np.reshape(recon_images[num_eigenfaces, :], (h, w))
    # for i in range(0, num_eigenfaces):
    #     plt.subplot(10, 10, i+1)
    #     plt.imshow(np.reshape(recon_images[i, :], (h, w)), 'gray')
    #     plt.axis('off')
    # plt.show()
    return reconstructed_image


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    '''
    Classify the given input image using the trained classifier
    :param img: input image to be classified, 1D-array
    :param eigenfaces: all given eigenfaces, 2D array with the eigenfaces as row vectors
    :param avg: the average image, 1D array
    :param num_eigenfaces: number of eigenfaces used to extract the features
    :param h: height of a original image
    :param w: width of a original image
    :return: the predicted labels using the classifier, 1D-array (as returned by the classifier)
    '''

    # reshape the input image as an matrix with the image as a row vector
    sample = np.reshape(img, (1, h*w))
    # extract the features/coefficients for the eigenfaces of this image
    coefficients = get_feature_representation(sample, eigenfaces, avg, num_eigenfaces)
    # predict the label of the given image by feeding its coefficients to the classifier
    predicted = clf.predict(coefficients)
    return predicted


if __name__ == '__main__':
    # coeff_1 = get_feature_representation(images=5 * np.ones((6, 4)),
    # eigenfaces=np.ones((8, 4)), avg=np.zeros(4), num_eigenfaces=7)
    # print(np.shape(coeff_1))
    # print(coeff_1)
    #
    # reco_1 = reconstruct_image(np.ones(12), np.ones((6, 12)), np.zeros(12), 6, 3, 4)
    # print(np.shape(coeff_1))
    labels, train, num_images = create_database_from_folder(glob.glob('eigenfaces/*.png'))
    # print(labels)
    avg = calculate_average_face(train.T)
    eigenfaces = calculate_eigenfaces(train.T, avg, 176, 64, 64)
    # print(np.shape(eigenfaces))
    coeff = get_feature_representation(images=train.T, eigenfaces=eigenfaces, avg=avg, num_eigenfaces=num_images)
    print(coeff.shape)
    # process_and_train(labels, train.T, 176, 64, 64)

    plt.subplot(1, 4, 1)
    plt.title('avg face')
    plt.imshow(avg.reshape((64, 64)), 'gray')

    recon_10 = reconstruct_image(train.T[15, :], eigenfaces, avg, 10, 64, 64)
    plt.subplot(1, 4, 2)
    plt.title('recon 10')
    plt.imshow(recon_10, 'gray')
    recon_50 = reconstruct_image(train.T[15, :], eigenfaces, avg, 50, 64, 64)
    plt.subplot(1, 4, 3)
    plt.title('recon 50')
    plt.imshow(recon_50, 'gray')
    recon_100 = reconstruct_image(train.T[15, :], eigenfaces, avg, 100, 64, 64)
    plt.subplot(1, 4, 4)
    plt.title('recon 100')
    plt.imshow(recon_100, 'gray')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(train.T[25, :], (64, 64)), 'gray')
    plt.title('Input Image')
    plt.axis('off')

    recon_all = reconstruct_image(train.T[25, :], eigenfaces, avg, 175, 64, 64)
    plt.subplot(1, 2, 2)
    plt.imshow(recon_all, 'gray')
    plt.title('Reconstructed Image')

    plt.axis('off')

    plt.show()
