import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def print_menu():
    print("Welcome to the Faction Selection Menu!")
    print("1. Display the Histogram")
    print("2. Display the Equalized image")
    print("3. Whitening the Image")
    print("4. Darkening the Image")
    print("5. Inverting the Image")
    print("6. Demonstrates the optimal thresholding process")
    print("0. All Factions")

def choose_faction():
    while True:
        print_menu()
        choice = input("Enter your choice (0-6): ")

        if choice == "1":
            print("You have chosen to display the Histogram.")
            histogram_image(Io)
            break

        elif choice == "2":
            print("You have chosen Equalized image.")
            # Call the function to get the equalized image.
            Ie = histogram_equalized_image(np.array(Io))
            # Display the equalized image.
            Image.fromarray(Ie).show()
            break

        elif choice == "3":
            print("You have chosen to Whitening the Image.")
            image_whitening(Io)
            break

        elif choice == "4":
            print("You have chosen to Darkening the Image.")
            image_darkening(Io)
            break

        elif choice == "5":
            print("You have chosen to Inverting the Image.")
            image_inverting(Io)
            break

        elif choice == "6":
            print("You have chosen to demonstrates the optimal thresholding process.")
            image_thresholding(Io)
            break

        elif choice == "0":
            print("You have chosen to display all Factions")
            histogram_image(Io)
            # Call the function to get the equalized image.
            Ie = histogram_equalized_image(np.array(Io))
            # Display the equalized image.
            Image.fromarray(Ie).show()
            image_whitening(Io)
            image_darkening(Io)
            image_inverting(Io)
            image_thresholding(Io)
            return
        else:
            print("Invalid choice. Please try again.")

def histogram_equalized_image(Io):

    # Convert the original image to double.
    Io = Io.astype(float)

    # Get the size of the original image.
    width, height, colours = Io.shape

    if colours == 3:
        # Get the constituent color components in separate matrices.
        IoR = Io[:, :, 0]
        IoG = Io[:, :, 1]
        IoB = Io[:, :, 2]

        # Reshape color constituent matrices into vectors.
        IoR = IoR.reshape(1, width * height)
        IoG = IoG.reshape(1, width * height)
        IoB = IoB.reshape(1, width * height)

        # Get the unique intensity values for each color component.
        IoRunique = np.unique(IoR)
        IoGunique = np.unique(IoG)
        IoBunique = np.unique(IoB)

        # Convert the unique intensity values to integers.
        IoRunique = IoRunique.astype(int)
        IoGunique = IoGunique.astype(int)
        IoBunique = IoBunique.astype(int)

        # Get the histogram vectors for each color component.
        HoR, _ = np.histogram(IoR.flatten(), bins=IoRunique)
        HoG, _ = np.histogram(IoG.flatten(), bins=IoGunique)
        HoB, _ = np.histogram(IoB.flatten(), bins=IoBunique)

        # Get the cumulative probability distribution function for each color component.
        CoR = np.cumsum(HoR) / (width * height)
        CoG = np.cumsum(HoG) / (width * height)
        CoB = np.cumsum(HoB) / (width * height)

        # Scale the cumulative histogram values to [0..255] range.
        RangeR = 255
        CoRmin = np.min(CoR)
        CoRmax = np.max(CoR)
        CoR = np.round((RangeR / (CoRmax - CoRmin)) * (CoR - CoRmin))

        RangeG = 255
        CoGmin = np.min(CoG)
        CoGmax = np.max(CoG)
        CoG = np.round((RangeG / (CoGmax - CoGmin)) * (CoG - CoGmin))

        RangeB = 255
        CoBmin = np.min(CoB)
        CoBmax = np.max(CoB)
        CoB = np.round((RangeB / (CoBmax - CoBmin)) * (CoB - CoBmin))

        # Define the mapping function for each color component.
        RFrequencyMap = {k: v for k, v in zip(IoRunique, CoR)}
        GFrequencyMap = {k: v for k, v in zip(IoGunique, CoG)}
        BFrequencyMap = {k: v for k, v in zip(IoBunique, CoB)}

        # Map the intensity levels of each pixel to the equalized values.
        IeR = np.array([RFrequencyMap.get(int(pixel), 0) for pixel in IoR.flatten()])
        IeG = np.array([GFrequencyMap.get(int(pixel), 0) for pixel in IoG.flatten()])
        IeB = np.array([BFrequencyMap.get(int(pixel), 0) for pixel in IoB.flatten()])

        # Reshape the arrays for each color component to their original dimensions.
        IeR = IeR.reshape(width, height)
        IeG = IeG.reshape(width, height)
        IeB = IeB.reshape(width, height)

        # Create the equalized image.
        Ie = np.zeros((width, height, colours), dtype=np.uint8)
        Ie[:, :, 0] = IeR
        Ie[:, :, 1] = IeG
        Ie[:, :, 2] = IeB

    else:
        Io = Io.reshape(1, width * height)
        Iounique = np.unique(Io)

        # Get the histogram and cumulative distribution function for the grayscale image.
        Ho, _ = np.histogram(Io.flatten(), bins=Iounique)
        Co = np.cumsum(Ho) / (width * height)

        # Scale the cumulative histogram values to [0..255] range.
        Range = 255
        Comin = np.min(Co)
        Comax = np.max(Co)
        Co = np.round((Range / (Comax - Comin)) * (Co - Comin))

        # Define the mapping function for the grayscale image.
        FrequencyMap = {k: v for k, v in zip(Iounique, Co)}

        # Map the intensity levels of each pixel to the equalized values.
        Ie = np.array([FrequencyMap.get(int(pixel), 0) for pixel in Io.flatten()])
        Ie = Ie.reshape(width, height)

    return Ie.astype(np.uint8)

def histogram_image(Io):
    plt.figure('Original Image')
    plt.imshow(Io)

    # Get image dimensions.
    width, height, colors = Io.shape

    # Get constituent color matrices corresponding to Red, Green, and Blue.
    IoR = Io[:, :, 0].reshape(1, width * height)
    IoG = Io[:, :, 1].reshape(1, width * height)
    IoB = Io[:, :, 2].reshape(1, width * height)

    # Compute intensity histograms for each color component.
    HoR, _ = np.histogram(IoR.flatten(), bins=range(257))
    HoG, _ = np.histogram(IoG.flatten(), bins=range(257))
    HoB, _ = np.histogram(IoB.flatten(), bins=range(257))

    # Plot intensity histograms for each color component.
    plt.figure('Original Image Color Histograms')
    plt.subplot(3, 1, 1)
    plt.bar(range(256), HoR, color='r')
    plt.xlabel('Red Color Intensity Levels')
    plt.ylabel('Number of pixels')

    plt.subplot(3, 1, 2)
    plt.bar(range(256), HoG, color='g')
    plt.xlabel('Green Color Intensity Levels')
    plt.ylabel('Number of pixels')

    plt.subplot(3, 1, 3)
    plt.bar(range(256), HoB, color='b')
    plt.xlabel('Blue Color Intensity Levels')
    plt.ylabel('Number of pixels')

    plt.show()

def image_whitening(Io):

    # Brightening the original image
    Iw = Io.astype(float) / 255.0

    # Initialize the gain and level parameters for the whitening operation
    k = 1.06
    l = 0.05
    Iw = k * Iw + l

    # Pixel values mapped to normalized intensity levels greater than 1 will be mapped to 1
    over_indices = np.where(Iw > 1)
    Iw[over_indices] = 1

    # Convert the whitened image back to uint8
    Iw = (Iw * 255).astype(np.uint8)

    plt.figure('Whitened Image')
    plt.imshow(Iw)

    plt.show(block=True)

def image_darkening(Io):

    # Brightening the original image
    Iw = Io.astype(float) / 255.0

    # Initialize the gain and level parameters for the whitening operation
    k = 0.75
    l = 0.00
    Iw = k * Iw + l

    # Pixel values mapped to normalized intensity levels greater than 1 will be mapped to 1
    over_indices = np.where(Iw > 1)
    Iw[over_indices] = 1

    # Convert the whitened image back to uint8
    Iw = (Iw * 255).astype(np.uint8)

    plt.figure('Whitened Image')
    plt.imshow(Iw)

    plt.show(block=True)

def image_inverting(Io):

    # Initialize the gain and level parameters for the whitening operation
    k = -1.00
    l = 1.00

    # Inverting the original image
    Iinv = Io.astype(float) / 255.0
    Iinv = k * Iinv + l
    Iinv = (Iinv * 255).astype(np.uint8)

    plt.figure('Inverted Image')
    plt.imshow(Iinv)

    plt.show(block=True)

def image_thresholding(Io):

    Io = np.array(Image.open('This.jpg').convert('L'))  # Convert to grayscale
    plt.figure('Original Image')
    plt.imshow(Io, cmap='gray')

    # Perform optimal thresholding
    Io = Io.astype(float)
    width, height = Io.shape
    Tmin = 0
    Tmax = 255
    Swithin = np.zeros(Tmax + 1)

    for T in range(Tmax + 1):
        BackgroundIndices = np.where(Io < T)
        Ibackground = Io[BackgroundIndices]
        ForegroundIndices = np.where(Io >= T)
        Iforeground = Io[ForegroundIndices]
        Pbackground, _ = np.histogram(Ibackground, bins=np.arange(Tmin, T))
        Pforeground, _ = np.histogram(Iforeground, bins=np.arange(T, Tmax + 1))
        Nbackground = np.sum(Pbackground)
        Nforeground = np.sum(Pforeground)
        Sbackground = np.var(Ibackground)
        Sforeground = np.var(Iforeground)
        Swithin[T] = Nbackground * Sbackground + Nforeground * Sforeground

    Tmin = np.argmin(Swithin)
    Io = Io.reshape(width, height)
    Io = np.clip(-(Io - Tmin), 0, 1)

    plt.figure('Thresholded Image')
    plt.show()

# Load an image.
print('Image Load it.')
Io = Image.open('This.jpg')
Io = np.array(Io)

# Call the function to start the menu
choose_faction()


