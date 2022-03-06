"""
DSC 20 Mid-Quarter Project
PID:  A16296514 and A16290003
"""

# Part 1: RGB Image #
class RGBImage:
    """
    Creates instances of images and stores and returns information relevant to
    their color or their "RBG colorspace"
    """

    def __init__(self, pixels):
        """
        Initializes the instance of the image and takes in/stores
        data about its pixel matrix.
        """
        # YOUR CODE GOES HERE #
        self.pixels = pixels # initialze the pixels list here

    def size(self):
        """
        This function returns the dimensions of the image in terms of the
        number of rows and the number of columns.
        """
        # YOUR CODE GOES HERE #
        row = len(self.pixels[0])
        col=len(self.pixels[0][0])
        return (row, col)

    def get_pixels(self):
        """
        This function returns a deep copy of the pixels_matrix so that
        that the original pixels's matrix is not manipulated by other functions.
        """
        # YOUR CODE GOES HERE #
        return [[[i for i in elem] for elem in element] for element in self.pixels]

    def copy(self):
        """
        This function uses the previously mentioned get_pixels to create a new
        instance of the same image so that the original image
        is not manipulated by other functions.
        """
        # YOUR CODE GOES HERE #
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        A function that returns the (R,G,B) format color information for a
        pixel specified by the row and column index given by the user.
        """
        # YOUR CODE GOES HERE #
        try:
            if not (isinstance(row, int) and isinstance(col, int)):
                raise TypeError()
            if (row>=self.size()[0] and row<0 and col>=self.size()[1] and col<0):
                raise ValueError()
        except TypeError as e:
            raise
        except ValueError as e:
            raise
        return (self.pixels[0][row][col], self.pixels[1][row][col], self.pixels[2][row][col])

    def set_pixel(self, row, col, new_color):
        """
        Replaces the color of the pixel at the given row and column index with
        the new color specified by the user.
        """
        # YOUR CODE GOES HERE #
        try:
            if not (isinstance(row, int) and isinstance(col, int)):
                raise TypeError()
            if (row>=self.size()[0] and row<0 and col>=self.size()[1] and col<0):
                raise ValueError()
        except TypeError as e:
            raise
        except ValueError as e:
            raise
        for i in range(len(new_color)):
            if new_color[i] <0:
                continue
            else:
                self.pixels[i][row][col] = new_color[i]



# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    This class defines methods that are used to process and manipulate images/
    instances of the RGBImage class mentioned above.
    """

    @staticmethod
    def negate(image):
        """
        This function returns the negative version of the given image, meaning
        that it finds the inverted color value for each pixel and returns the new
        matrix as a part of a new RGBImage image/ instance.
        """
        # YOUR CODE GOES HERE #
        new_pix = [[[255-i for i in elem] for elem in element] for element in image.get_pixels()]
        #new_pix=list(map(lambda k: list(map(lambda x: list(map(lambda y: list(map(lambda z: 255-z, y), x)), k)), image)))
        return RGBImage(new_pix)

    @staticmethod
    def tint(image, color):
        """
        This function returns a tinted version of the given image, meaning that at each pixel
        the function replaces the orignal value with the average of the orignal value and the
        value at the corresponding channel of the new color. It then returns this new image
        as an RGBImage instance.
        """
        # YOUR CODE GOES HERE #
        new_red = [[(i + color[0])//2 for i in elem] for elem in image.get_pixels()[0]]
        new_green = [[(i + color[1])//2 for i in elem] for elem in image.get_pixels()[1]]
        new_blue = [[(i + color[2])//2 for i in elem] for elem in image.get_pixels()[2]]
        new_pixel=[new_red, new_green, new_blue]
        return RGBImage(new_pixel)

    @staticmethod
    def clear_channel(image, channel):
        """
        This function clears the given channel, meaning it replaces all values of the
        specified channel with 0, while leaving the other two channels untouched. It
        then returns the new image, with the cleared channel, as an RGBImage instance.
        """
        # YOUR CODE GOES HERE #
        new_channel = [[0 for i in elem] for elem in image.get_pixels()[channel]]
        if channel==0:
            return  RGBImage([new_channel, image.get_pixels()[1], image.get_pixels()[2]])
        if channel==1:
            return  RGBImage([image.get_pixels()[0],new_channel, image.get_pixels()[2]])
        if channel==2:
            return  RGBImage([image.get_pixels()[0], image.get_pixels()[1], new_channel])

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        Returns an image that is a cropped version of the original image, the
        desired dimensions are provided by the target size and desired cropping
        point of the image are provided by the tl_col and tl_row. The pixels at
        and past the cropping point are returned as a new image/RGB instance.
        """
        # YOUR CODE GOES HERE #
        lst=[]
        if  tl_row + target_size[0] <= image.size()[0] and tl_col + target_size[1] <= image.size()[1]:
            br_row= tl_row + target_size[0] - 1
            br_col= tl_col + target_size[1] - 1
            lst = [[[x[i] for i in range(tl_col, br_col)] for x in elem] for elem in image.get_pixels()]
            lst = [[elem[i] for i in range(tl_row, br_row)] for elem in lst]
        else:
            lst = [[[x[i] for i in range(tl_col, image.size()[1])] for x in elem] for elem in image.get_pixels()]
            lst = [[elem[i] for i in range(tl_row, len(elem))] for elem in lst]
        return RGBImage(lst)


    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        This function reviews all of the pixels and if the pixels are the same
        color as the color given by the user, the pixel in the orignal image is
        replaced by the corresponding pixel in the background image. The new
        merged image is returned as an RGBImage.
        """
        # YOUR CODE GOES HERE #
        try:
            if not (isinstance(chroma_image, RGBImage) and isinstance(background_image, RGBImage)):
                raise TypeError()
            if background_image.size()!=chroma_image.size():
                raise ValueError()
        except TypeError as e:
            raise
        except ValueError as e:
            raise
        #red = [[background_image.get_pixels()[0][] if x == color[0] for x in elem] for elem in chroma_image.get_pixels()[0]]

        background_image1 = background_image.copy()
        chroma_image1 = chroma_image.copy()
        for elem in range(chroma_image1.size()[0]): #rows
            for x in range(chroma_image1.size()[1]): #columns
                a = chroma_image1.get_pixel(elem, x)
                if a == color:
                    chroma_image1.set_pixel(elem,x,background_image1.get_pixel(elem, x))
        return chroma_image1

    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        Function returns a rotated version of the image.
        """
        # YOUR CODE GOES HERE #
        new = [[e[::-1] for e in row[::-1]] for row in image.get_pixels()]
        return RGBImage(new)

# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    This class defines methods that are used to predict the nearest label and most popular label.
    """

    def __init__(self,n_neighbors):
        """
        Initializes the instance of the ImageKNNClassifier and takes in/stores
        data about the number of neighbors.
        """
        # YOUR CODE GOES HERE #
        self.n_neighbors=n_neighbors
        self.data = []

    def fit(self, data):
        """
        Stores all of the data in the instance.
        """
        # YOUR CODE GOES HERE #
        try:
            if not len(data) > self.n_neighbors:
                raise ValueError()
            if len(self.data)!= 0:
                raise ValueError()
        except ValueError as e:
            raise
        except TypeError as e:
            raise
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        Calculates the Euclidean distance between two images using the RGB colorspace
        matrix.
        """

        try:
            if image1.size() != image2.size():
                raise ValueError()
            if not (isinstance(image1, RGBImage) and isinstance(image2, RGBImage)):
                raise TypeError()
        except TypeError as e:
            raise
        except ValueError as e:
            raise
        row= image1.size()[0]
        col= image1.size()[1]
        channel = 3
        return sum([(image1.pixels[chan][i][j]-image2.pixels[chan][i][j]) **2 \
        for j in range(0, col) for i in range(0, row) for chan in range(0, channel)]) ** 0.5


    @staticmethod
    def vote(candidates):
        """
        Finds and returns the most popular label from a given list of labels.
        """
        max_list = {}
        alr_appeared = []
        for i in candidates:
            if i not in alr_appeared:
                alr_appeared.append(i)
                max_list[candidates.count(i)] = i
        val = max(max_list.keys())
        return max_list[val]


    def predict(self, image):
        """
        Uses the previously mentioned functions to predict and return the given image's label.
        """
        # YOUR CODE GOES HERE #
        try:
            if self.data == 0:
                raise ValueError()
        except ValueError as e:
            raise

        find_list = list(map(lambda i: (ImageKNNClassifier.distance(image, i[0]), i[1]), self.data))
        find_list.sort(key = lambda y: y[0])
        find_from = find_list[0: self.n_neighbors]
        send_list = [i[1] for i in find_from]
        return ImageKNNClassifier.vote(send_list)
