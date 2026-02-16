import cv2
import numpy as np
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "pedestrians.mp4"
    output_data_path = project_root / "data" / "pedestrian_trajectories.npy"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Background Subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    all_frames_data = []

    print("Processing video (centroid detections per frame)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Background subtraction
        fgmask = fgbg.apply(frame)

        # 2) Remove shadows + threshold to binary
        #    MOG2 shadows often appear as 127; keep only strong foreground (255)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # 3) Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4) Find contours and compute centroids
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:  # ignore small noise blobs
                continue

            M = cv2.moments(cnt)
            if M["m00"] <= 1e-9:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            frame_centroids.append([cx, cy])

        all_frames_data.append(np.array(frame_centroids, dtype=np.float32))

    cap.release()
    cv2.destroyAllWindows()

    np.save(output_data_path, np.array(all_frames_data, dtype=object))
    print(f"Saved pedestrian detections: {output_data_path}")
    print(f"Frames processed: {len(all_frames_data)}")


if __name__ == "__main__":
    main()
