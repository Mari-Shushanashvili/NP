import cv2
import numpy as np
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "pedestrians.mp4"
    output_data_path = project_root / "data" / "pedestrian_trajectories.npy"

    cap = cv2.VideoCapture(str(video_path))

    # Background Subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    all_frames_data = []

    print("Processing video... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Apply background subtraction
        fgmask = fgbg.apply(frame)

        # 2. Clean up noise (Morphology)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

        # 3. Find Contours (the pedestrians)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_peds = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 150:  # Ignore small noise
                continue

            # Get center of the pedestrian
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_frame_peds.append([cx, cy])

                # Visual feedback
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

        all_frames_data.append(current_frame_peds)

        cv2.imshow('Pedestrian Tracking (Green dots = Detected obstacles)', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save data for the simulation
    np.save(output_data_path, np.array(all_frames_data, dtype=object))
    print(f"Extraction complete! Saved {len(all_frames_data)} frames of data to {output_data_path}")


if __name__ == "__main__":
    main()