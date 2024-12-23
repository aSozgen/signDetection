import cv2

# 1. Load the input image 2 4 7 14
image = cv2.imread('tsign2.jpeg')
if image is None:
    raise FileNotFoundError("Image file not found. Please check the file path.")

# Show original image
cv2.imshow('Original Image', image)

# 2. Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Contrast Enhancement
# 3.1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced_image = clahe.apply(gray_image)

# Show enhanced image
cv2.imshow('Enhanced Image (CLAHE)', enhanced_image)
# cv2.imwrite('tsign14_enhanced.jpg', enhanced_image)

# 4. Noise Reduction
# 4.1. Apply Gaussian Blur for noise reduction
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

# Show Gaussian Blured image
cv2.imshow('Gaussian Image', blurred_image)
# cv2.imwrite('tsign14_blured.jpg', blurred_image)

# 5. Edge Detection
# 5.1. Apply Canny Edge Detection
edges_canny = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Show edge-detected image
cv2.imshow('Canny Edge Detection', edges_canny)
# cv2.imwrite('tsign14_canny.jpg', edges_canny)

# 6. Contour Detection and Shape Analysis
# Find contours from the edge-detected image
contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the enhanced image
output_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 275:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Draw the contour on the image
        color = (0, 255, 0)
        cv2.drawContours(output_image, [approx], -1, color, 2)

# Show the final output
cv2.imshow('Detected Signs', output_image)

# 7. Save the output image
# cv2.imwrite('tsign14_output_detected_signs.jpg', output_image)

# 8. Close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
