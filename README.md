# An exploration of data generation methods

I don't know about you, but I have troubled myself over what the data means when working on Machine Learning algorithms. Most of the time, the dataset iss to big to have an intuitive sense of what the data represents and how it related to the algorithm which uses it. I've found that there is no solution to this other than experience, and one way of gaining experience is by creating the data itself from scratch. 
This repository was created for this reason alone. Just data generation. 

---

# Problem Statement: Create a dataset that will be used in a supervised machine learning model.

Requirements:
10 classes.
500 datapoints.
algorithm wiht 80% accuracy.

Intuitive solution:
Supervised learning means having a label for a dataset. Lets think abou it from a solution first idealogy. What is the end point or a use case?

I don't want to simply create random points and give them names, that would be too simple. How about having something like an image detection which could be used for color prediction based on regions on the image space. 

So, lets start with the image,  take random region in an image and take the color as the label. It would make it computationally easier on our end. We basiceally won't have to code a lot.

  ### required libraries, use this command '!pip install opencv-python' or 'pip install opencv-python' if you don't have the library
  import cv2
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

  ### load and preprocess the image
  file_loc = 'image.png'
  image = cv2.imread(file_loc)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  ### apply blurr 
  blurred = cv2.GaussianBlur(image_rgb, (7,7),3)

Lets see what our image is, 
![image](https://github.com/user-attachments/assets/496bb69a-fff1-4d97-9db7-9c6eb7e5c869)

Who's that pokemon??? 
It's a Squirtle!!

Now, We need to define an what the color scheme is going to be (create a hash)
  ### define the classes
  color_centers = {
      'Red':     np.array([255, 0, 0]),
      'Green':   np.array([0, 255, 0]),
      'Blue':    np.array([0, 0, 255]),
      'Yellow':  np.array([255, 255, 0]),
      'Cyan':    np.array([0, 255, 255]),
      'Magenta': np.array([255, 0, 255]),
      'Orange':  np.array([255, 165, 0])
  }
  color_name = list(color_centers.keys())
  centers = np.stack(list(color_centers.values()), axis = 0) # shape is going to be (7,3)

  Now, Convert this to an actual dataset, a DataFrame!
  ### create the dataset from the image
  pixels = blurred.reshape(-1,3)
  
  def assign_label(pixel):
      distances = np.linalg.norm(centers - pixel, axis=1)
      label_index = np.argmin(distances)
      return label_index
  
  ### Vectorize labeling: assign a class (0 to 6) for each pixel.
  labels = np.array([assign_label(pix) for pix in pixels])

We should also add noise, just to make sure the data doesn't overfit the model
  ### add noise to the feature
  noise_std = 20  # standard deviation of noise
  noisy_pixels = pixels + np.random.normal(0, noise_std, pixels.shape)
  noisy_pixels = np.clip(noisy_pixels, 0, 255).astype(np.uint8)

---

  Now that we have a dataset, we have to see if this actually works with different models.
  ### Train a supervised classifier?
  X = noisy_pixels
  y = labels
  
  ### Split the dataset into training and testing sets.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  ### Train a RandomForest classifier.
  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(X_train, y_train)
  
  ### Evaluate the classifier.
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Test Accuracy: {accuracy:.2f}")
  
  ### 6. Use the model to reconstruct the image
  
  ### Predict the label for every (noisy) pixel.
  predicted_labels = clf.predict(noisy_pixels)
  ### Map each label back to its representative color (the color center).
  reconstructed_pixels = np.array([centers[label] for label in predicted_labels], dtype=np.uint8)
  ### Reshape back to the original image shape.
  reconstructed_image = reconstructed_pixels.reshape(blurred.shape)
  
  ### 7. Visualize the results
  plt.figure(figsize=(12, 6))
  
  plt.subplot(1, 2, 1)
  plt.imshow(blurred)
  plt.title("Blurred Image")
  plt.axis('off')
  
  plt.subplot(1, 2, 2)
  plt.imshow(reconstructed_image)
  plt.title("Reconstructed Image from Classifier")
  plt.axis('off')
  
  plt.show()
  
  ## Test Accuracy: 0.89

  ![image](https://github.com/user-attachments/assets/f8f8d87d-07e5-43e0-a9fa-f67ebb65b93b)

Seems like it works! 

