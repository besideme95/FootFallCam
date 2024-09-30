from business_logic import *

video_path = 'media\\project\\sample-video.mp4'
output_folder = 'media\\extracted_frames'

model_path = 'Models\\yolov8n.pt'
# Model name shortcut
## human_model_8x.pt
## nametag_8x_v1.pt
## targeted_staff_8n_v2(100epoch).pt
## targeted_staff_8x_v1.pt
## yolov8n.pt
## yolov8x.pt


# Video Process Main Function
# processor = VideoProcessor(video_path, output_folder)
# processor.process_video()
# processor.extract_frames()

# detector = ObjectDetection(video_path, model_path)
# detector.detect_objects()

# Search image Main function
# if __name__ == "__main__":
#     target_image_path = 'media\\project\\white-shirt-staff.png'  # Path to the image you want to find
#     dataset_folder = 'media\\extracted_frames'             # Folder containing the sample of 100 images

#     # Create an instance of ImageRetriever
#     image_retriever = ImageRetriever()
#     best_match = image_retriever.find_best_match(target_image_path, dataset_folder)

#     print(f"The image most similar to the target is: {best_match}")
