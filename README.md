*Malaria Detection AI Model*

A fine-tuned deep learning model that detects malaria in microscopic blood smear images with 95.81% accuracy.

This project is focused on the development of a high-accuracy deep learning model for malaria detection. It features a fine-tuned Convolutional Neural Network (CNN) that can classify *ANY* microscopic cell image as either "Parasitized" or "Uninfected" with high precision. The final model achieves a validation accuracy of 95.81% and is served via a simple Flask API for easy testing.


Here is a video of its working:




https://github.com/user-attachments/assets/961d3abf-dd2d-4aab-a6f1-0724ce9e090f



*Core AI Features*

High Accuracy: The model achieves 95.81% validation accuracy after a strategic training process.

Advanced Architecture: Built on the powerful and efficient MobileNetV2 architecture, which is pre-trained on the massive ImageNet dataset.

Transfer Learning: Leverages the pre-trained knowledge of MobileNetV2, allowing the model to understand complex visual features from the start.

Two-Phase Fine-Tuning: Employs a sophisticated training process, first training a custom classifier and then fine-tuning the deeper layers of the base model for specialized accuracy on this specific task.

Robustness through Augmentation: Uses data augmentation (random flips, rotations) during training to create a more diverse dataset, which helps prevent overfitting and improves the model's ability to generalize to new, unseen images.

*Model Architecture & Training Methodology*

The key to achieving high accuracy was not just the model architecture, but the methodology used to train it.

Base Architecture: MobileNetV2
I chose MobileNetV2 as our base model. It is a state-of-the-art, lightweight, and highly efficient computer vision model. Its pre-training on the ImageNet dataset means it already has a powerful, built-in understanding of general visual features like edges, textures, and shapes.


*The Power of Transfer Learning*

Instead of training a neural network from scratch, which would require an enormous amount of data and time, I used transfer learning. This technique involves taking a powerful pre-trained model (like MobileNetV2) and adapting it to a new, specific task. In essence, I took a model that was already an expert in "seeing" and taught it the specialized skill of identifying malaria parasites in blood cells.


Two-Phase Training Process

Phase 1: Feature Extraction

I began by "freezing" all the layers of the base MobileNetV2 model.

Then added a custom classification "head" on top, consisting of a few Dense layers.

In this phase, I trained only this new classifier. This step quickly taught the model how to map the powerful, pre-existing features from MobileNetV2 to our two specific classes: "Parasitized" and "Uninfected".


Phase 2: Fine-Tuning

After the initial training, I "unfroze" the top layers of the base model.

Then continued training the entire model with a very low learning rate. This crucial step subtly adjusted the more specialized feature detectors within the MobileNetV2 architecture, making them even better at spotting the unique visual characteristics of malaria parasites.
This is what pushed the accuracy past the 95% mark.
