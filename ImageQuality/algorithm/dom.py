from .include import *
class DOM(object):

    def __init__(self):
        self.image = None
        self.Im = None
        self.edgex, self.edgey = None, None

    @staticmethod
    def load(img, blur=False, blurSize=(5, 5)):
        """ Handle img src, 3/2 channel image
        :param imgPath: image path or array
        :type: str, np.ndarray
        :param blur: to blur original image
        :type: boolean
        :param blurSize: size of gausian filter
        :type: tuple (n,n)
        :return image: grayscale image
        :type: np.ndarray
        :return Im: median filtered grayscale image
        :type: np.ndarray
        """

        if isinstance(img, str):
            if os.path.exists(img):
                # Load image as grayscale
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            else:
                raise FileNotFoundError('Image is not found on your system')
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                image = img
            else:
                raise ValueError('Image is not in correct shape')
        else:
            raise ValueError('Only image can be passed to the constructor')

        # Add Gaussian Blur
        if blur:
            image = cv2.GaussianBlur(image, blurSize)

        # Perform median blur for removing Noise
        Im = cv2.medianBlur(image, 3, cv2.CV_64F).astype("double") / 255.0
        return image, Im

    @staticmethod
    def dom(Im):
        """ Find DOM at each pixel
        :param Im: median filtered image
        :type: np.ndarray
        :return domx: diff of diff on x axis
        :type: np.ndarray
        :return domy: diff of diff on y axis
        :type: np.ndarray
        """
        median_shift_up = np.pad(Im, ((0, 2), (0, 0)), 'constant')[2:, :]
        median_shift_down = np.pad(Im, ((2, 0), (0, 0)), 'constant')[:-2, :]
        domx = np.abs(median_shift_up - 2 * Im + median_shift_down)

        median_shift_left = np.pad(Im, ((0, 0), (0, 2)), 'constant')[:, 2:]
        median_shift_right = np.pad(Im, ((0, 0), (2, 0)), 'constant')[:, :-2]
        domy = np.abs(median_shift_left - 2 * Im + median_shift_right)

        return domx, domy

    @staticmethod
    def contrast(Im):
        """ Find contrast at each pixel
        :param Im: median filtered image
        :type: np.ndarray
        :return Cx: contrast on x axis
        :type: np.ndarray
        :return Cy: contrast on y axis
        :type: np.ndarray
        """
        Cx = np.abs(Im - np.pad(Im, ((1, 0), (0, 0)), 'constant')[:-1, :])
        Cy = np.abs(Im - np.pad(Im, ((0, 0), (1, 0)), 'constant')[:, :-1])
        return Cx, Cy

    @staticmethod
    def smoothenImage(image, transpose=False, epsilon=1e-8):
        """ Smmoth image with ([0.5, 0, -0.5]) 1D filter
        :param image: grayscale image
        :type: np.ndarray
        :param transpose: to apply filter on vertical axis
        :type: boolean
        :param epsilon: small value to defer div by zero
        :type: float
        :return image_smoothed: smoothened image
        :type: np.ndarray
        """
        fil = np.array([0.5, 0, -0.5])  # Smoothing Filter

        # change image axis for column convolution
        if transpose:
            image = image.T

        # Convolve grayscale image with smoothing filter
        image_smoothed = np.array([np.convolve(image[i], fil, mode="same") for i in range(image.shape[0])])

        # change image axis after column convolution
        if transpose:
            image_smoothed = image_smoothed.T

        # Normalize smoothened grayscale image
        image_smoothed = np.abs(image_smoothed) / (np.max(image_smoothed) + epsilon)
        return image_smoothed

    def edges(self, image, edge_threshold=0.0001):
        """ Get Edge pixels
        :param image: grayscale image
        :type: np.ndarray
        :param edge_threshold: threshold to consider pixel as edge if its value is greater
        :type: float
        :assign edgex: edge pixels matrix in x-axis as boolean
        :type: np.ndarray
        :assign edgey: edge pixels matrix in y-axis as boolean
        :type: np.ndarray
        """
        smoothx = self.smoothenImage(image, transpose=True)
        smoothy = self.smoothenImage(image)
        self.edgex = smoothx > edge_threshold
        self.edgey = smoothy > edge_threshold

    def sharpness_matrix(self, Im, width=2, debug=False):
        """ Find sharpness value at each pixel
        :param Im: median filtered grayscale image
        :type: np.ndarray
        :param width: edge width
        :type: int
        :param debug: to show intermediate results
        :type: boolean
        :return Sx: sharpness value matrix computed in x-axis
        :type: np.ndarray
        :return Sy: sharpness value matrix computed in y-axis
        :type: np.ndarray
        """
        # Compute dod measure on both axis
        domx, domy = self.dom(Im)

        # Compute sharpness
        Cx, Cy = self.contrast(Im)

        # Filter out sharpness at pixels other than edges
        Cx = np.multiply(Cx, self.edgex)
        Cy = np.multiply(Cy, self.edgey)

        # initialize sharpness matriz with 0's
        Sx = np.zeros(domx.shape)
        Sy = np.zeros(domy.shape)

        # Compute Sx
        for i in range(width, domx.shape[0] - width):
            num = np.abs(domx[i - width:i + width, :]).sum(axis=0)
            dn = Cx[i - width:i + width, :].sum(axis=0)
            Sx[i] = [(num[k] / dn[k] if dn[k] > 1e-3 else 0) for k in range(Sx.shape[1])]

        # Compute Sy
        for j in range(width, domy.shape[1] - width):
            num = np.abs(domy[:, j - width: j + width]).sum(axis=1)
            dn = Cy[:, j - width:j + width].sum(axis=1)
            Sy[:, j] = [(num[k] / dn[k] if dn[k] > 1e-3 else 0) for k in range(Sy.shape[0])]

        if debug:
            print("domx {domx.shape}: {[(i,round(np.quantile(domx, i/100), 2)) for i in range(0, 101, 25)]}")
            print("domy {domy.shape}: {[(i,round(np.quantile(domy, i/100), 2)) for i in range(0, 101, 25)]}")
            print("Cx {Cx.shape}: {[(i,round(np.quantile(Cx, i/100),2)) for i in range(50, 101, 10)]}")
            print("Cy {Cy.shape}: {[(i,round(np.quantile(Cy, i/100),2)) for i in range(50, 101, 10)]}")
            print("Sx {Sx.shape}: {[(i,round(np.quantile(Sx, i/100),2)) for i in range(50, 101, 10)]}")
            print("Sy {Sy.shape}: {[(i,round(np.quantile(Sy, i/100),2)) for i in range(50, 101, 10)]}")

        return Sx, Sy

    def sharpness_measure(self, Im, width, sharpness_threshold, debug=False, epsilon=1e-8):
        """ Final Sharpness Value
        :param Im: median filtered grayscale image
        :type: np.ndarray
        :param width: edge width
        :type: int
        :param sharpness_threshold: thresold to consider if a pixel is sharp
        :type: float
        :param debug: to show intermediate results
        :type: boolean
        :return S: sharpness measure(0<S<sqrt(2))
        :type: float
        """
        Sx, Sy = self.sharpness_matrix(Im, width=width, debug=debug)
        Sx = np.multiply(Sx, self.edgex)
        Sy = np.multiply(Sy, self.edgey)

        n_sharpx = np.sum(Sx >= sharpness_threshold)
        n_sharpy = np.sum(Sy >= sharpness_threshold)

        n_edgex = np.sum(self.edgex)
        n_edgey = np.sum(self.edgey)

        Rx = n_sharpx / (n_edgex + epsilon)
        Ry = n_sharpy / (n_edgey + epsilon)

        S = np.sqrt(Rx ** 2 + Ry ** 2)

        if debug:
            print("Sharpness: {S}")
            print("Rx: {Rx}, Ry: {Ry}")
            print("Sharpx: {n_sharpx}, Sharpy: {n_sharpy}, Edges: {n_edgex, n_edgey}")
        return S

    def get_sharpness(self, img, width=2, sharpness_threshold=2, edge_threshold=0.0001, debug=False):
        """ Image Sharpness Assessment
        :param img: img src or image matrix
        :type: str or np.ndarray
        :param width: text edge width
        :type: int
        :param sharpness_threshold: thresold to consider if a pixel is sharp
        :type: float
        :param edge_threshold: thresold to consider if a pixel is an edge pixel
        :type: float
        :param debug: to show intermediate results
        :type: boolean

        :return score: image sharpness measure(0<S<sqrt(2))
        :type: float
        """
        image, Im = self.load(img)
        # Initialize edge(x|y) matrices
        self.edges(image, edge_threshold=edge_threshold)
        score = self.sharpness_measure(Im, width=width, sharpness_threshold=sharpness_threshold)
        return score