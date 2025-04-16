
import numpy as np
import pydicom as dcm
import cv2 as cv
import os
from datetime import datetime

class ModifyRtFiles:
    def __init__(self, dcm_path=None, tensor=None, angle=None, CT_bool=False):
        self.dcm_path = dcm_path
        self.coordinates = [[90, 263, 405], [135, 265, 368], [165, 267, 405]] #AP
                        #  [[52, 242, 384,], [96, 280, 384], [128, 242, 384]] LR
        self.CT_flag = CT_bool

        if CT_bool:
            self.CT_dir = self.sort_dir_by_number()

        if angle is None:
            self.angle = self.calc_angle()
        else:
            self.angle = angle

        if dcm_path is None:
            self.tensor = tensor
        elif dcm_path.split(".")[-1] == "dcm" and not CT_bool:
            self.tensor = self.load_dose()
        elif CT_bool:
            self.tensor = self.load_CT()
        else:
            self.tensor = None

    def calc_angle(self):
        center_coordinates = self.coordinates
        return np.abs(90 - (np.arctan2(center_coordinates[2][0] - center_coordinates[0][0],
                            center_coordinates[0][1] - center_coordinates[2][1]) * (180 / np.pi)))

    def sort_dir_by_number(self):
        CT_dirc = os.listdir(self.dcm_path)

        if 'fixed' in CT_dirc[0]:
            CT_dirc.sort(key=lambda x: int(x.split('.')[-2].replace('_fixed', '')))
        elif 'fused' in CT_dirc[0]:
            CT_dirc.sort(key=lambda x: int(x.split('.')[-2].replace('_fused', '')))
        else:
            CT_dirc.sort(key=lambda x: int(x.split('.')[-1]))

        return CT_dirc

    def load_CT(self):
        CT_slcs = []
        CT_dir = self.sort_dir_by_number()

        for slc in CT_dir:
            slc_path = os.path.join(self.dcm_path, slc)
            CT_slice = dcm.dcmread(slc_path).pixel_array
            CT_slcs.append(CT_slice)

        return np.array(CT_slcs)

    def load_dose(self):
        Dose = dcm.dcmread(self.dcm_path)
        return Dose.pixel_array

    @staticmethod
    def get_pixel_array(slice_path):
        slc_dcm = dcm.dcmread(slice_path)
        return slc_dcm.pixel_array

    @staticmethod
    def rotate_90_axial(tensor):
        # Swap the height and width, and reverse the height axis
        return np.flip(np.transpose(tensor, (0, 2, 1)), axis=1)

    @staticmethod
    def rotate_to_sagital(tensor):
        # Swap the depth and width, and reverse the width axis
        return np.flip(np.transpose(tensor, (2, 1, 0)), axis=2)

    @staticmethod
    def rotate_to_coronal(tensor):
        # Swap the depth and height, and reverse the depth axis
        return np.flip(np.transpose(tensor, (1, 0, 2)), axis=0)

    def rotate_tensor_90(self, tensor, axis, k=1):
        """
        Rotate a 3D tensor by 90 degrees k times around the specified axis.

        Parameters:
        tensor (np.ndarray): The input 3D tensor.
        axis (int): The axis to rotate around (0 for x, 1 for y, 2 for z).
        k (int): Number of 90-degree rotations to perform.

        Returns:
        np.ndarray: The rotated tensor.
        """
        # Normalize the number of rotations
        k = k % 4

        if axis == 0:  # x-axis
            for _ in range(k):
                tensor = self.rotate_90_axial(tensor)
        elif axis == 1:  # y-axis
            for _ in range(k):
                tensor = self.rotate_to_sagital(tensor)
        elif axis == 2:  # z-axis
            for _ in range(k):
                tensor = self.rotate_to_coronal(tensor)
        else:
            raise ValueError("Axis must be 0 (axial), 1 (coronal), or 2 (sagital).")

        return tensor

    def rotate_2D(self, array):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

        height, width = array.shape[:2]  # image shape has 3 dimensions
        x, y = self.coordinates[0][2], self.coordinates[0][0]
        rotation_center = (y, x)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv.getRotationMatrix2D(rotation_center, -self.angle, 1)

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv.warpAffine(array, rotation_mat, array.shape[::-1], flags=cv.INTER_LANCZOS4)

        return rotated_mat

    def rotate_tensor(self, orientation="sagital"):
        """Použít pro rotaci celého tensoru. V rámci třídy to znemaná rotace dávky nebo struktur."""

        if orientation == "sagital":
            tensor = self.rotate_tensor_90(self.tensor, 1)   # Potřeba "orotovat", abych iteroval přes sagitální
        elif orientation == "coronal":                            # řezy, jinak bych rotoval tensor podle longu
            tensor = self.rotate_tensor_90(self.tensor, 2)
        else:
            tensor = self.tensor

        new_tensor = np.zeros_like(tensor)

        for i, slc in enumerate(tensor):
            new_tensor[i] = self.rotate_2D(slc)

        if orientation == "sagital":
            try:
                self.save_modified_file(self.rotate_tensor_90(new_tensor, 1, k=3))
            except TypeError:
                return self.rotate_tensor_90(new_tensor, axis=1, k=3)

        elif orientation == "coronal":
            try:
                self.save_modified_file(self.rotate_tensor_90(new_tensor, 2, k=3))
            except TypeError:
                self.tensor = new_tensor
        else:
            try:
                self.save_modified_file(new_tensor)
            except TypeError:
                self.tensor = new_tensor
    # def rotate_CT(self):
    #     slices = self.load_CT_slices()
    #     for slc in slices:
    #         CT_slice = self.get_pixel_array(slc)
    #         self.save_modified_file(self.rotate_2D(CT_slice), slc)

    def save_modified_file(self, modified_array: np.ndarray):
        """Zachovává DICOM hodnoty souboru stejné, jen změní hodnoty pixel array. """

        if not self.CT_flag:
            array_dcm = dcm.dcmread(self.dcm_path)

            try:
                if array_dcm.pixel_array.shape != modified_array.shape:
                    raise ValueError
                else:
                    data_bytes = modified_array.tobytes()
                    array_dcm.PixelData = data_bytes
                    array_dcm.save_as(filename=self.dcm_path, write_like_original=False)
            except ValueError:
                print("Rozměry dávky po změně nesedí.")
                quit()

        else:
            for rotated_slice, slc_dir in zip(modified_array, self.CT_dir):
                slc_path = os.path.join(self.dcm_path, slc_dir)
                slc_data = dcm.dcmread(slc_path)
                try:
                    if slc_data.pixel_array.shape != rotated_slice.shape:
                        raise ValueError
                    slc_data.PixelData = rotated_slice.tobytes()
                    slc_data.save_as(slc_path, False)
                except ValueError:
                    print("Rozměry CT po změně nesedí.")
                    quit()


if __name__ == "__main__":
    # CT_path = r"C:\Users\Jakub\Dropbox\Zlodějíčkovo slůj\Práce\Export\CT\SBRT_90"
    #
    # CT = ModifyRtFiles(dcm_path=CT_path, CT_bool=True)
    # CT.rotate_tensor()

    RDRS_dir_path = r"C:\Users\Jakub\Dropbox\Zlodějíčkovo slůj\Práce\Export\Dose_export_19.8\Dose"
    doses_RS = os.listdir(RDRS_dir_path)

    for path in doses_RS:
        RDRS_path = os.path.join(RDRS_dir_path, path)

        if "90" in path and os.path.getmtime(path) < datetime(2024, 8, 20).timestamp():
            if RDRS_path.split(".")[-1] == "dcm" and "RS" not in RDRS_path:
                RDRS = ModifyRtFiles(dcm_path=RDRS_path)
                RDRS.rotate_tensor()


"""Načítání RS struktur komlikovanější, protože mám jenom souřadnice v prostoru (vztaženého vůči počátku CT) a ne v rámci tenzoru. 
Kontrola Film_dosi implementace v rámci načítání struktur ze dvou různých CT(jestli nebude problém).  """


