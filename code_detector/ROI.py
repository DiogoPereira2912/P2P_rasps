import cv2
import numpy as np
import yaml
import os


class ROIHandler:
    """
    Class to handle Region of Interest (ROI) selection and frame cropping.
    """

    def __init__(self, roi_file=None):
        """
        Initialize the ROIHandler.

        Args:
            roi_file: Path to the YAML file containing the ROI coordinates.
        """
        self.roi = None  
        if roi_file:
            self.load_roi_from_yaml(roi_file)

    def select_and_save_roi(self, frame, count, yaml_path="roi_coords.yaml"):
        """
        Interactively selects a Region of Interest (ROI) on the given frame.

        Args:
            frame: The input frame on which to select the ROI.

        Returns:
            A tuple containing the coordinates of the selected ROI:
                - x1: Top-left x-coordinate of the ROI.
                - y1: Top-left y-coordinate of the ROI.
                - x2: Bottom-right x-coordinate of the ROI.
                - y2: Bottom-right y-coordinate of the ROI.
            Or None if no ROI was selected.
        """
        roi = cv2.selectROI("Select ROI", frame, False, False)
        cv2.destroyAllWindows()  # Close the ROI selection window after selection
        if roi == (0, 0, 0, 0):  # No ROI selected
            print("No ROI selected.")
            return None
        x1, y1, width, height = roi
        x2 = x1 + width
        y2 = y1 + height
        self.roi = (x1, y1, x2, y2)

        data = {}
        data[f"roi_{count}"] = {
            "x1": self.roi[0],
            "y1": self.roi[1],
            "x2": self.roi[2],
            "y2": self.roi[3],
        }

        with open(yaml_path, "a") as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"ROI saved: {self.roi}")
        return self.roi

    def crop_frame(self, frame):
        """
        Crop the input frame based on the stored ROI coordinates.

        Args:
            frame: Input frame to be cropped.

        Returns:
            The cropped frame, or the original frame if no ROI is set.
        """
        if self.roi is None:
            print("Warning: No ROI set. Returning the original frame.")
            return frame
        x1, x2, y1, y2 = self.roi
        height, width = frame.shape[:2]
        # Ensure ROI coordinates are within frame bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return frame[y1:y2, x1:x2]
