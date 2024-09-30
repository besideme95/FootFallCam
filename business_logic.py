
import cv2
import os
import numpy as np
from ultralytics import YOLO

class VideoProcessor:
    """
    A class to process video files using OpenCV.
    """

    def __init__(self, video_path, output_folder):
        """
        Initialize the VideoProcessor with the path to the video file and output folder.
        
        Parameters:
        video_path (str): The path to the video file.
        output_folder (str): The folder to save the extracted frames.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError('Error: Could not open the video file. Please check.')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def extract_frames(self, count=0):
        """
        Extract frames from the video file and save them as image files.
        """
        frame_count = count
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_filename = os.path.join(self.output_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        self.cap.release()
        print(f'Total frames extracted: {frame_count}')

    def process_video(self):
        """
        Process the video file, display each frame, and overlay the frame count.
        """
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Overlay the frame count on the frame
            cv2.putText(
                frame, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sample-Video', frame)
            # cv2.moveWindow('Sample-Video', 0, 0)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


    def __del__(self):
        """
        Release the video capture and destroy all OpenCV windows.
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


class ObjectDetection:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model = YOLO(model_path)

    def detect_objects(self):
        results = self.model(
            source=self.video_path, show=True, conf=0.4, save=True, stream=True)

        for r in results:
            boxes = r.boxes  # Bounding box outputs
            masks = r.masks  # Segmentation masks outputs
            probs = r.probs  # Class probabilities for classification outputs

        print("Object detection completed.")


class ImageRetriever:
    def __init__(self):
        # Initialize SIFT and FLANN-based matcher
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)  # FLANN parameters
        search_params = dict(checks=50)           # Search parameters
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_sift_features(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2):
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        
        return len(good_matches)

    def find_best_match(self, target_image_path, dataset_folder):
        target_keypoints, target_descriptors = self.extract_sift_features(target_image_path)
        
        best_match_count = 0
        best_match_image = None
        
        for image_file in os.listdir(dataset_folder):
            image_path = os.path.join(dataset_folder, image_file)
            
            _, dataset_descriptors = self.extract_sift_features(image_path)
            
            match_count = self.match_features(target_descriptors, dataset_descriptors)
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_match_image = image_file
        
        return best_match_image