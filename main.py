import cv2
import torch
import timm
import urllib.request
from torchvision import transforms
from PIL import Image

# Set device: GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained Swin Transformer model for image classification.
model_name = "swin_tiny_patch4_window7_224"
model = timm.create_model(model_name, pretrained=True)
model.eval()
model.to(device)

# Load all 1000 ImageNet class labels (matches model output indices)
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode().splitlines()

# Define the transformation pipeline (resize, convert to tensor, normalize).
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Function to preprocess a video frame.
def preprocess_frame(frame):
    # Convert frame from BGR (OpenCV) to RGB (PIL Image format).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    # Apply the defined transforms.
    img_tensor = transform(pil_img)
    # Add batch dimension.
    return img_tensor.unsqueeze(0).to(device)

# Function to annotate the frame with the predicted label.
def annotate_frame(frame, label_text):
    cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

# Start video capture (use index 0 for the primary webcam).
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the video source.")
    exit()

print("Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the current frame.
    input_tensor = preprocess_frame(frame)

    # Run inference.
    with torch.no_grad():
        outputs = model(input_tensor)
        # Compute probabilities via softmax.
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    # Convert the prediction to a label.
    top_idx = top_idx.item()
    label = f"{imagenet_classes[top_idx]}: {top_prob.item()*100:.1f}%"

    # Annotate the frame.
    annotated_frame = annotate_frame(frame.copy(), label)

    # Display the annotated frame.
    cv2.imshow("Real-Time ImageNet Classification", annotated_frame)

    # Break loop on 'q' key press.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up.
cap.release()
cv2.destroyAllWindows()