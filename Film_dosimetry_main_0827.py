"""
Film Dosimetry Analysis Tool
============================

Tento skript provádí komplexní analýzu srovnání dávkových distribucí v antropomorfním fantomu.
Porovnává dávky vypočítané v plánovacím systému s dávkami naměřenými pomocí gafchromických
filmů EBT4.

Hlavní funkcionality:
- Výpočet gamma pass rate pro kvantitativní srovnání dávkových distribucí
- Porovnání dávkových profilů mezi plánovanou a měřenou dávkou
- Srovnání DVH (Dose Volume Histogram) křivek
- Prostorová registrace pomocí lineárního posunu podle polohy fiduciálů v CT a na filmu

Vstupy:
- Naskenované gafchromické filmy ve formátu TIFF
- CT snímky s definovanou polohou fantomu
- Dávkové distribuce z plánovacího systému ve formátu DICOM
- Struktury z plánovacího systému ve formátu DICOM RS

Výstupy:
- Vizualizace gamma analýzy s vyznačením oblastí nesouladu
- Grafické porovnání dávkových profilů
- DVH křivky pro definované struktury
- Kvantitativní hodnocení shody pomocí gamma analýzy

Použití:
Program lze spustit z příkazové řádky s různými parametry:
  -f, --film       Název souboru naskenovaného filmu
  -t, --RD         Název DICOM souboru dávky z TPS
  -a, --DTA        Hodnota 'distance to agreement' v mm (výchozí: 2)
  -d, --DD         Hodnota 'dose difference' v % (výchozí: 3)
  -c, --cut_off    Procentuální hodnota minimální uvažované dávky (výchozí: 10)
  -r, --resolution Rozlišení gamma analýzy v mm (výchozí: 0.2)
  -s, --RS         Název ROI z plánovacího systému (výchozí: 'IC_EX')
  -l, --slice      Číslo řezu z CT, na kterém se nachází film (výchozí: 383)

 - lze ale nikdy jsem to reálně nepoužil
"""
import csv
import optparse
import os
from scipy import spatial
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pydicom as dcm
import tifffile as tiff
import time
import Rotate_RT_tensor as RotRT
from scipy.optimize import minimize
import Optimization_calibration_curve as OptCal


# NTB
filepath = r"C:\Users\kubad\Dropbox\Zlodějíčkovo slůj\Python\UVN_scripts\Film_dosimetry"

"""CT for loading CT information...slice thickness etc."""
CT_add_path = r"\CT&Dose\srpen_24\CT\SBRT_90"
CT_dir = filepath + CT_add_path

try:
    CT_slice = os.listdir(CT_dir)[-1]
except FileNotFoundError:
    filepath = r"C:\Users\kubad\Dropbox\Zlodějíčkovo slůj\Python\UVN_scripts\Film_dosimetry"
    CT_dir = filepath + CT_add_path
    CT_slice = os.listdir(CT_dir)[-1]

CT_for_info = dcm.dcmread(CT_dir + "\\" + CT_slice)
CT_pixel_spacing = CT_for_info.PixelSpacing[0], CT_for_info.SliceThickness, CT_for_info.SliceThickness  # L-R, A-P, S-I - pixel spacing
CT_image_position = CT_for_info.ImagePositionPatient    # [x, x, x] LR, AP, SI - počáteční poloha je u hlavy nahoře v pravo aneb S,A,R
# CT_image_position[2] = -999.5
# CT_image_position[2] = -1016.50   #Při vyhodnocování jednoho z filmů z DCM headeru vycházela divná poloha, tak jsem to přepsal
CT_dimensions = len(os.listdir(CT_dir)), CT_for_info.Rows, CT_for_info.Columns

"""Výpis možností při spouštění z cmd."""
start_time = time.time()
parser = optparse.OptionParser()

parser.add_option("-f", "--film", action="store", dest="film_dose",
                  help="- název souboru naskenovaného filmu, který má být vyhodnocen")  # Nakonec udělat tak, aby se měnilo jenom jméno souboru
parser.add_option("-t", "--RD", action="store", dest="RT_dose",
                  help="- název dcm souboru dávky, nahrané z TPS (CT není potřeba)")
parser.add_option("-a", "--DTA", action="store", dest="DTA",
                  help="- hodnota 'distance to agreement'(v milimetrech)", default=2)
parser.add_option("-d", "--DD", action="store", dest="DD",
                  help="- hodnota 'get_dose difference' (v procentech)", default=3)
parser.add_option("-c", "--cut_off", action="store", dest="cut_off",
                  help="- procentuální hodnota, od které níže již není dávka uvažována", default=10)
parser.add_option("-r", "--resolution", action="store", dest="resolution",
                  help="- rozlišení gamma analýzy (v milimetrech)", default=0.2)
parser.add_option('-s', '--RS', action='store', dest='RS_name', default=['IC EX'],
                  help='- název ROI z plánovačky')
# Hodnoty řezů CT pokud je orientace filmu psána jako LR (tzn. 0 aneb rotace insertu do horizontální polohy)
#                               -> 374 (z listopadovýho měření), 381 (sprnové), 384 (24_srpen)

# Hodnoty řezů CT pokud je orientace filmu psána jako AP (tzn. 90 aneb rotace insertu do vodorovné polohy)
#                               -> 261 (z listopadovýho měření), 264 (srpnové), 267 (24_srpen)
parser.add_option('-l', '--slice_number', action="store", dest='slice_number', default=267,
                  help='-číslo řezu z CT, na kterém se nachází film')

gamma_options, args = parser.parse_args()


"""Začátek samotného program definováním jednotlivých tříd a operací pro - Film_dose, CT, TPS_dose, Gamma, RS, Vizualizace"""


def print_center_mean(film_data, region_size=40, called_where=None):
    try:
        if np.mean(film_data[:10, 0]) > 20000:
            film_data = np.where(film_data < 40000, film_data, 0)
            max_dose_point = np.array(Visualization.max_array_points(200, film_data))
            center_x = int(np.median(max_dose_point[0, :]))
            center_y = int(np.median(max_dose_point[1, :]))
            center_region = film_data[center_y - region_size:center_y + region_size,
                                      center_x - region_size:center_x + region_size]
            mean_value = center_region.mean()
            print("Mean value surrounding max dose: ", called_where, " ", mean_value)

    except IndexError:
        pass


class Film:
    def __init__(self, film_path):

        # Apply transformations based on the file path
        if "LR" in film_path:
            self.triple_film = np.transpose(np.array(tiff.imread(os.path.abspath(film_path))), axes=(2, 0, 1))[:, :, :]
            self.triple_film = self.triple_film[[1, 0, 2], :, :]
            self.film_for_reg = self.triple_film[2, :, :]
        else:
            self.triple_film = np.transpose(np.array(tiff.imread(os.path.abspath(film_path))), axes=(2, 0, 1))[:, :, ::-1]

            self.film_for_reg = self.triple_film[2, :, :]

        self.h, self.w = self.film_for_reg.shape

    def get_key_points_and_prep(self, do_print=False):

        film = self.film_for_reg

        # Apply smoothing filter
        kernel = np.ones((3, 3)) / 9

        film_thr = cv.morphologyEx(film, cv.MORPH_ERODE, np.ones((10, 10)) / 100)
        film_thr = cv.morphologyEx(film_thr, cv.MORPH_DILATE, np.ones((7, 7)) / 49)
        ret, film_thr = cv.threshold(film_thr, 19000, 255, cv.THRESH_BINARY_INV)
        # ret, film_thr = cv.threshold(film_thr, 128, 255, cv.THRESH_BINARY_INV)


        # Cut film
        film_cut_lat = (50,  #vlevo
                        70)  #vpravo
        film_cut_long = (30, #nahore
                         30)  #dole
        film_thr = film_thr[film_cut_long[0]:-film_cut_long[1], film_cut_lat[0]:-film_cut_lat[1]]

        plt.imshow(film_thr)
        plt.show()

        # Find contours
        contours, _ = cv.findContours(film_thr.astype(np.uint8), mode=cv.RETR_LIST,
                                      method=cv.CHAIN_APPROX_TC89_KCOS)
        contours = [contour for contour in contours if 0 not in contour]
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]
        coordinates = []

        print("Klíčové body na filmu: ")
        print("_______________________")

        for c in contours:
            M = cv.moments(c)

            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print('centroid: X:{}, Y:{}'.format(cX, cY))

            coordinates.append([cX, cY])

            if do_print:
                cv.circle(film_thr, (cX, cY), 5, (255, 255, 255), -1)
                cv.putText(film_thr, f"({cX}),({cY})", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 2)
                plt.imshow(film_thr, "gray")
                plt.show()

        coordinates = sorted(coordinates, key=lambda x: x[1])
        coordinates = np.array(coordinates)

        if do_print:
            cv.drawContours(film_thr, contours, -1, (200, 200, 0), thickness=30)
            plt.imshow(film_thr)
            plt.show()

        center = coordinates[0, 1].astype(np.float32), coordinates[0, 0].astype(np.float32)


        angle = - (np.arctan2(coordinates[2, 0] - coordinates[0, 0], coordinates[2, 1] - coordinates[0, 1]) * (180 / np.pi))
        scale = 1

        rot_mat = cv.getRotationMatrix2D(center, angle, scale)

        final_film = cv.warpAffine(np.array(film), rot_mat, dsize=(film.shape[1], film.shape[0]),
                                   flags=cv.INTER_LANCZOS4)
        final_film = final_film[film_cut_long[0]:-film_cut_long[1], film_cut_lat[0]:-film_cut_lat[1]]

        for i, channel in enumerate(self.triple_film):
            self.triple_film[i, :, :] = cv.warpAffine(np.array(channel), rot_mat,
                                                      dsize=(film.shape[1], film.shape[0]),flags=cv.INTER_LANCZOS4)
        self.triple_film = self.triple_film[:, film_cut_long[0]:-film_cut_long[1], film_cut_lat[0]:-film_cut_lat[1]]

        # ret, film_thr = cv.threshold(film_thr, 128, 255, cv.THRESH_BINARY_INV)

        final_film = cv.filter2D(final_film, -1, np.ones((5, 5)) / 25)

        film_thr = cv.morphologyEx(final_film, cv.MORPH_ERODE, np.ones((10, 10)) / 100)
        film_thr = cv.morphologyEx(film_thr, cv.MORPH_DILATE, np.ones((7, 7)) / 49)

        ret, film_thr = cv.threshold(film_thr, 20000, 255, cv.THRESH_BINARY_INV)


        plt.imshow(film_thr)
        plt.show()

        contours, _ = cv.findContours(film_thr.astype(np.uint8), mode=cv.RETR_LIST,
                                      method=cv.CHAIN_APPROX_TC89_KCOS)
        contours = [contour for contour in contours if 0 not in contour]
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]
        coordinates = []
        print("_________________________________")
        print("Klíčové body na filmu po rotaci: ")
        for c in contours:
            if [[0, 0]] in c:
                continue
            else:
                try:
                    M = cv.moments(c)

                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print('centroid: X:{}, Y:{}'.format(cX, cY))

                    coordinates.append([cX, cY])
                except ZeroDivisionError:
                    print('Neplecha na 190. řádku při hledání referenčních bodů na filmu.')
                    continue

        coordinates = sorted(coordinates, key=lambda x: x[1])
        coordinates = np.array(coordinates)
        max_min_vertical_cen = (np.max(coordinates[:, 1]), np.min(coordinates[:, 1]))
        max_min_horizontal_cen = (np.max(coordinates[:, 0]), np.min(coordinates[:, 0]))

        return coordinates, max_min_vertical_cen, max_min_horizontal_cen, final_film


    @staticmethod
    def rational_fun(x, a, b, c):
        return a / (x + b) + c

    def apply_calibration(self, input_value):
        """
        Aplikuje kalibraci. Druhá klaibrace pro novou várku filmů zohledňovala dlouha/kratka a data byla ve formátu:  x_fit = data[kalibrace::2, 2].astype(np.float64)...1 = DD, 2 = DK.
        Třetí kalibrace se soustředila na faceUp/faceDown. 1 = fU, 12 = fD
        :param input_value:
        :return:
        """
        # with open(filepath + r'\\Filmy\\kalibrace_08252024.txt', 'r') as fit_data:
        #     data = csv.reader(fit_data)
        #     data = np.array(list(data))
        #
        #     w_R = 0.022017731074578856
        #     w_G = 0.8497782878279837
        #     w_B = 0.12820398109743744
        #
        #     x_fit = (w_R * data[:, 1].astype(np.float64) +
        #              w_G * data[:, 2].astype(np.float64) +
        #              w_B * data[:, 3].astype(np.float64))
        #     y_fit = data[:, 0].astype(np.float64)

        fit_params_red = OptCal.fit_individual_channels(OptCal.D_known, OptCal.P_R)
        fit_params_green = OptCal.fit_individual_channels(OptCal.D_known, OptCal.P_G)
        fit_params_blue = OptCal.fit_individual_channels(OptCal.D_known, OptCal.P_B)

        weights = minimize(OptCal.objective, np.array([1/3, 1/3, 1/3]),
                           constraints=OptCal.constraints, bounds=[(0, 1), (0, 1), (0, 1)])

        w_R, w_G, w_B = weights.x

        film_dose = (w_R * self.rational_fun(input_value[0, :, :], *fit_params_red) +
                     w_G * self.rational_fun(input_value[1, :, :], *fit_params_green) +
                     w_B * self.rational_fun(input_value[2, :, :], *fit_params_blue))

        print(f"Optimal weights: w_R = {w_R}, w_G = {w_G}, w_B = {w_B}")

        # params, _ = curve_fit(self.rational_fun, x_fit, y_fit, p0=[50000, 20000, -1])
        # a, b, c = params
        #
        # plt.scatter(x_fit, y_fit)
        # x_range_fit = np.linspace(int(np.min(x_fit)), int(max(x_fit)) * 1.1, 20)
        # plt.plot(x_range_fit, Film.rational_fun(x_range_fit, a, b, c))
        # plt.title(f'a = {a:.2f}, b = {b:.2f}, c = {c:.2f}')
        # print(f'Hodnoty fitu jsou: a = {a:.6g}, b = {b:.6g}, c = {c:.6g}')
        # plt.show()
        #
        # x = input_value
        #
        # film_dose = a / (x + b) + c

        return film_dose

    def resize_film_and_get_new_coors(self, distance_ratio_ver, distance_ratio_hor, film_cut):
        w, h = film_cut.shape[1], film_cut.shape[0]

        film_resized = cv.resize(film_cut, (int(w / distance_ratio_hor), int(h / distance_ratio_ver)))  # cv2, takže prohozený řádky a sloupce
        resized_triple_film = np.zeros((3, int(h / distance_ratio_ver), int(w / distance_ratio_hor)))

        kernel = np.ones((4, 4)) / 16

        for i, channel in enumerate(self.triple_film):
            channel = cv.filter2D(channel, -1, np.ones([13, 13]) / 169)
            resized_triple_film[i, :, :] = cv.resize(channel,
                                                     (int(w / distance_ratio_hor), int(h / distance_ratio_ver)))

        film_thr = cv.morphologyEx(film_resized, cv.MORPH_ERODE, np.ones((10, 10)) / 100)
        film_thr = cv.morphologyEx(film_thr, cv.MORPH_DILATE, np.ones((7, 7)) / 49)

        ret, film_thr = cv.threshold(film_thr, 26800, 255, cv.THRESH_BINARY_INV)
        # ret, film_thr = cv.threshold(film_thr, 128, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(film_thr.astype(np.uint8), mode=cv.RETR_LIST,
                                      method=cv.CHAIN_APPROX_TC89_KCOS)
        contours = [contour for contour in contours if 0 not in contour]
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]
        coordinates = []

        print("_____________________________________")
        print("Klíčové body na přeškálovaném filmu: ")

        for c in contours:
            if [[0, 0]] in c:
                continue
            else:
                try:
                    M = cv.moments(c)

                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    print('centroid: X:{}, Y:{}'.format(cX, cY))

                    coordinates.append([cX, cY])
                except ZeroDivisionError:
                    continue

        print("\n")
        coordinates = sorted(coordinates, key=lambda x: x[1])
        new_coordinates = np.array(coordinates)

        max_min_hor = [np.max(new_coordinates[:, 0]), np.min(new_coordinates[:, 0])]
        max_min_vert = [np.max(new_coordinates[:, 1]), np.min(new_coordinates[:, 1])]

        film_resized = cv.filter2D(film_resized, -1, np.ones((3, 3))/9)

        plt.imshow(film_resized)
        plt.show()

        return resized_triple_film, max_min_vert, max_min_hor


class CT:
    def __init__(self, CT_path):
        self.CT_dir = self.sort_dir_by_number(CT_path)
        self.CT_path = CT_path
        self.CT = self.load_CT()

        if '90' in self.CT_path:
            self.interval = gamma_options.slice_number - 1
            self.CT_slice_for_loc = np.median(np.array(self.CT)[:, self.interval:self.interval + 2, :], axis=1)
             #beru tři slicy v oblasti fiducialů
        else:
            self.interval = gamma_options.slice_number - 1
            self.CT_slice_for_loc = np.median(np.array(self.CT)[:, :, self.interval:self.interval + 2], axis=2)

    @staticmethod
    def sort_dir_by_number(CT_path):
        CT_dirc = os.listdir(CT_path)

        if 'fixed' in CT_dirc[0]:
            CT_dirc.sort(key=lambda x: int(x.split('.')[-2].replace('_fixed', '')))
        elif 'fused' in CT_dirc[0]:
            CT_dirc.sort(key=lambda x: int(x.split('.')[-2].replace('_fused', '')))
        else:
            CT_dirc.sort(key=lambda x: int(x.split('.')[-1]))

        return CT_dirc

    def load_CT(self):
        CT = []
        for slc in self.CT_dir:
            CT_slice = dcm.dcmread(self.CT_path + '\\' + slc).pixel_array
            CT.append(CT_slice)

        return CT

    def get_key_points(self, do_print=False):
        """
        Hledá střed kontrastních bodů na CT.

        :return:
        """
        plt.imshow(self.CT_slice_for_loc)
        plt.show()

        ret, CT_thr = cv.threshold(np.array(self.CT_slice_for_loc), 1500, 255, cv.THRESH_BINARY)

        try:
            CT_res = cv.resize(CT_thr, (int(CT_dimensions[1] * (CT_pixel_spacing[0]/gamma_options.resolution)),
                 int(CT_dimensions[0] * (CT_pixel_spacing[1]/gamma_options.resolution))))
        except cv.error:
            print('Problém s přeškálováním CT, pravděpodobně nebyl trefen správný slice a tím pádem při thresholdingu vznikla prázdná matice.')
            exit()

        #   plt.imshow(CT_res)
        #   plt.show()

        contours, _ = cv.findContours(CT_res.astype(np.uint8), mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)  # Změna aproximace na SIMPLE za účelem šetření paměti
        coordinates = []
        print("______________________________________")
        print("SOUŘADNICE KLÍČOVÝCH BODŮ:")
        print("____________________")
        print("Klíčové body na CT: ")
        try:
            for c in contours:
                M = cv.moments(c)

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print('centroid: X:{}, Y:{}'.format(cX, cY))

                if do_print:
                    cv.circle(CT_res, (cX, cY), 5, (255, 255, 255), -1)
                    cv.putText(CT_res, f"({cX}),({cY})", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 2)

                    # display the image
                    plt.imshow(CT_res, "gray")
                    plt.show()

                coordinates.append([cX, cY])

            coordinates = np.array(coordinates)
            max_min_vertical = (np.max(coordinates[:, 1]), np.min(coordinates[:, 1]))
            max_min_horizontal = (np.max(coordinates[:, 0]), np.min(coordinates[:, 0]))

        except ZeroDivisionError or IndexError:
            print(f'\n V CT se nepodařilo najít body potřebné pro registraci. Zkus jiné slicy. Nalezené kontury: {contours}')
            exit()

        return CT_res, max_min_vertical, max_min_horizontal


class Dose:
    def __init__(self, dose_path):
        self.dose_path = dose_path
        self.dcm_dose = dcm.dcmread(dose_path)
        self.slice_number = gamma_options.slice_number

        if "90" in dose_path:
            self.interval = (self.slice_number - 1, self.slice_number + 1)
            self.pixel_array = np.mean(self.dcm_dose.pixel_array[::, self.interval[0]:self.interval[1], :], axis=1)

        elif "0" in dose_path:
            self.interval = (self.slice_number - 1, self.slice_number + 1)
            self.pixel_array = np.mean(self.dcm_dose.pixel_array[::, :, self.interval[0]:self.interval[1]], axis=2)

        else:
            print("Něco se pokazilo při nahrávání dat dávky.")

        test_dose = np.max(self.dcm_dose.pixel_array)
        self.max_dose = self.dcm_dose.DVHNormalizationDoseValue
        self.dose_grid = self.dcm_dose.DoseGridScaling
        print(f"Maximální dávka v celém objemu: {test_dose * self.dose_grid}")

        self.get_dose = self.resized_dose()  # převod z pixelové hodnoty na dávku a naškálování do rozlišení gamy

    def resized_3D_dose(self):
        shape = [int(CT_dimensions[0] * (CT_pixel_spacing[1]/gamma_options.resolution)),
                 int(CT_dimensions[1] * (CT_pixel_spacing[0]/gamma_options.resolution))]

        tensor = np.empty(shape, dtype=float)
        tensor = np.expand_dims(tensor, axis=0)
        dose = self.pixel_array

        if 'AP' in self.dose_path:
            shape_d = np.shape(dose)[2]
        else:
            shape_d = np.shape(dose)[1]

        for i in range(0, shape_d, 1):
            if 'AP' in self.dose_path:
                slc = dose[:, :, i]
            else:
                slc = dose[:, i, :]
            array_slice = slc / (np.max(self.pixel_array) / self.max_dose)  # Převod z pixelů na dávku

            array_slice = cv.resize(array_slice, shape[::-1])  # Dostávám dávku v rozlišení pro gamu...z nějakého důvodu musí být trnasponované
            array_slice = np.expand_dims(array_slice, axis=0)

            tensor = np.append(tensor, array_slice, axis=0)

        plt.imshow(tensor[4, ...])
        plt.show()

        return tensor[1:, :, :]

    def resized_dose(self):
        shape = [int(CT_dimensions[0] * (CT_pixel_spacing[1]/gamma_options.resolution)),
                 int(CT_dimensions[1] * (CT_pixel_spacing[0]/gamma_options.resolution))]
        dose = self.pixel_array * self.dose_grid  # Převod z pixelů na dávku
        dose = cv.resize(dose, shape[::-1])  # Dostávám dávku v rozlišení pro gamu...transponované, protože ten kvádr CT vnímám jako položený na stranu
        print(f"Maximální dávka na daném řezu: {np.max(dose)}")
        dose = cv.filter2D(dose, -1, np.ones((6, 6)) / 36)
        plt.imshow(dose)
        plt.title('Nahraná dávka z TPS')
        plt.show()

        return dose


class RTStructures:
    def __init__(self, RS_name, RS_filename):
        self.RS_path = filepath + r'\CT&Dose\srpen_24\RTD_RTS\0819_2024' + RS_filename
        self.RS_name = RS_name  # je list, takže se dá použít i pro více struktur
        self.RS = dcm.dcmread(self.RS_path)
        self.RSs_tensor = self.get_tensor()
        # Vrací slovník {RS_name: RS_tensor}

        if "not now" in RS_filename:
            self.rotate_RS_tensor()

        self.RS_mask_in_gamma_res = self.resize_to_gama_resolution()

    def rotate_RS_tensor(self):

        for name, RS_ten in self.RSs_tensor.items():
            RS_tensor = RotRT.ModifyRtFiles(tensor=RS_ten)
            self.RSs_tensor[name] = RS_tensor.rotate_tensor(orientation="sagital")

    def resize_to_gama_resolution(self):
        RS_resized_masks = {}
        shape = [int(CT_dimensions[0] * (CT_pixel_spacing[1]/gamma_options.resolution)),
                 int(CT_dimensions[1] * (CT_pixel_spacing[0]/gamma_options.resolution))]

        for RS_name, RS_tensor in self.RSs_tensor.items():
            try:
                if '90' not in self.RS_path:
                    to_resize = RS_tensor[:, :,  gamma_options.slice_number]
                else:
                    to_resize = RS_tensor[:, gamma_options.slice_number, :]
            except ValueError:
                print('Něco se pokazilo při převodu kontur do rozlišení gamy.')
                exit()

            RS_array = np.flipud(cv.resize(to_resize, shape[::-1]))
            RS_resized_masks[RS_name] = RS_array

        return RS_resized_masks

    def get_roi_contour_sequence(self):
        RT_structures = {}
        for name in self.RS_name:
            ROI_name_to_number_map = {
                structure_set.ROIName: structure_set.ROINumber
                for structure_set in self.RS.StructureSetROISequence
            }

            ROI_number_to_contour_map = {
                contour.ReferencedROINumber: contour
                for contour in self.RS.ROIContourSequence
            }

            try:
                ROI_number = ROI_name_to_number_map[name]
            except KeyError:
                raise ValueError("Struktura s daným jménem nebyla nalezena (pozor na malá/velká písmena)")

            roi_contour_sequence = ROI_number_to_contour_map[ROI_number]

            RT_structures[name] = roi_contour_sequence.ContourSequence

        return RT_structures

    def get_tensor(self):
        RSs = self.get_roi_contour_sequence()
        RS_collection = {}
        for RS_name, RS_coordinates in RSs.items():

            voxel_coordinates = [0, 0, 0]
            RS_tensor = np.zeros(CT_dimensions)

            for slice_coordinates in RS_coordinates:    # Tady dostávám indexy pro daný slice
                col = 0
                slice_indices = []

                for coordinate in slice_coordinates.ContourData:  # Tady indexy pro daný slice přepočítám...formát: sagitál, koronál, axiál
                    if col == 0 or col == 1:
                        voxel_coordinates[col] = int(np.abs(CT_image_position[col] - coordinate) / CT_pixel_spacing[0])
                    else:
                        voxel_coordinates[col] = int(np.abs(CT_image_position[col] - coordinate) / CT_pixel_spacing[1])

                    if (col + 1) % 3 == 0 and col != 0:
                        appending_vector = np.copy(voxel_coordinates[:])
                        slice_indices.append(appending_vector)
                        col = -1

                    col += 1

                while True:
                    mat_to_comp = np.copy(RS_tensor[voxel_coordinates[2], ...])
                    depth_dimension = voxel_coordinates[2]

                    for ind_1 in slice_indices:  # Porovná všechny indexy a spojí horizontálně, vertikálně a diagonálně

                        for ind_2 in slice_indices:
                            diff_c = ind_1[0] - ind_2[0]
                            diff_r = ind_1[1] - ind_2[1]

                            if diff_c == 0 and diff_r ** 2 > 1:   # Ekviv. ind1[0] < ind2[0]
                                RS_tensor[depth_dimension, ind_1[1]:ind_2[1], ind_1[0]] = 1

                            elif diff_r == 0 and diff_c ** 2 > 1:  # Ekviv. ind1[1] < ind2[1]
                                RS_tensor[depth_dimension, ind_1[1], ind_1[0]:ind_2[0]] = 1

                            elif diff_r ** 2 == diff_c ** 2 and diff_c < 0:
                                list_c = np.linspace(ind_1[0], ind_2[0], np.abs(diff_c) + 1, endpoint=True)
                                list_r = np.linspace(ind_1[1], ind_2[1], np.abs(diff_r) + 1, endpoint=True)

                                for c, r in zip(list_c, list_r):
                                    RS_tensor[depth_dimension, int(r), int(c)] = 1

                    slice_indices = np.array(RS_tensor[depth_dimension, ...].nonzero())[[1, 0]].T
                    comparison = np.equal(RS_tensor[depth_dimension, ...], mat_to_comp)
                    if False not in comparison:
                        break
            kde_mam_hledat_strukturu_v_tensoru = np.where(RS_tensor > 0)

            RS_collection[RS_name] = RS_tensor

        return RS_collection


class Gamma:
    def __init__(self, film_dose, TPS_dose, shift, RS_mask):
        """
        Třída pro výpočet gamy mezi filmem a dávkou z TPS.
        Parametry:  film_dose, TPS_dose, shift
        :param TPS_dose:
        :param film_dose:
        :param shift:
        """
        self.DTA = gamma_options.DTA
        self.DD = gamma_options.DD
        self.resolution = gamma_options.resolution
        self.RS_mask_in_gamma_res = RS_mask
        self.TPS_dose = TPS_dose
        self.film_dose = np.where(np.array(film_dose) <= np.max(self.TPS_dose) * 2, film_dose, 0)
        self.cut_off_film = np.where(self.film_dose >= (gamma_options.cut_off / 100) * np.max(self.TPS_dose), self.film_dose, 0)
        self.shift = shift
        self.row_range = np.expand_dims(np.arange(np.shape(self.film_dose)[0]), axis=1)
        self.col_range = np.expand_dims(np.arange(np.shape(self.film_dose)[1]), axis=0)
        self.gamma_array = self.gamma_function(self.TPS_dose, mask=None)
        self.pass_rate = self.calculate_pass_rate(mask=None)

    def calculate_pass_rate(self, mask=None):
        gamma_array = self.gamma_array
        evaluated_dose = self.cut_off_film
        if mask:
            mask_RS = self.RS_mask_in_gamma_res[mask][self.shift[0]: self.shift[0] + evaluated_dose.shape[0],
                                                      self.shift[1]: self.shift[1] + evaluated_dose.shape[1]]
            mask_RS = mask_RS > 0
        else:
            mask_RS = evaluated_dose > 0

        gamma_in_cut_off = np.logical_and(gamma_array <= 1, mask_RS)
        num_of_passed = gamma_in_cut_off.sum()
        num_of_evaluated = mask_RS.sum()
        pass_rate = num_of_passed / num_of_evaluated * 100

        return pass_rate

    def gamma_function(self, reference_dose, mask=None, debbuging=False):  # ZKONTROLOVÁNO, NEOTESTOVÁNO
        if debbuging:
            try:
                with open(filepath + r'\gamy\\'
                          + f'gamma_array.csv', 'r') as f:
                    reader = csv.reader(f)
                    array_list = list(reader)

                    gamma_array = np.array(array_list).astype("float")
            except FileNotFoundError:
                print('Gama matrix not found.')
        else:
            evaluated_dose = self.cut_off_film

            if mask:
                mask_RS = self.RS_mask_in_gamma_res[mask][self.shift[0]: self.shift[0] + evaluated_dose.shape[0],
                                                          self.shift[1]: self.shift[1] + evaluated_dose.shape[1]]
                evaluated_dose = np.where(mask_RS, evaluated_dose, 0)

            plt.imshow(evaluated_dose)
            plt.show()
            pixel_DTA = gamma_options.DTA / gamma_options.resolution
            gamma_array = np.zeros_like(evaluated_dose)
            # Dostanu gama mapu jen o velikosti filmu...pracovat s větší by bylo zbytečný
            DD_dose = gamma_options.DD / 100 * np.max(reference_dose)

            for index, measured_dose in np.ndenumerate(evaluated_dose):
                if measured_dose > 0:
                    gamma_index_threshold = np.inf
                    reference_index = np.array([index[0] + self.shift[0], index[1] + self.shift[1]])

                    delta_d = np.abs((reference_dose[reference_index[0], reference_index[1]] - measured_dose) / DD_dose)

                    r = pixel_DTA * min(1, delta_d)

                    circle_mask = (self.row_range - index[0]) ** 2 + (self.col_range - index[1]) ** 2 < r ** 2
                    iteration_matrix = np.where(circle_mask)

                    delta_d = ((reference_dose[reference_index[0], reference_index[1]] - self.film_dose[iteration_matrix])
                               / DD_dose)
                    delta_r = (np.sqrt(((iteration_matrix[0] - index[0]) * gamma_options.resolution) ** 2
                                      + ((iteration_matrix[1] - index[1]) * gamma_options.resolution) ** 2)
                               / gamma_options.DTA)

                    for d_d, d_r in zip(delta_d, delta_r):
                        if d_d < 0:
                            gamma_index = - np.sqrt(d_d ** 2 + d_r ** 2)
                        elif d_d > 0:
                            gamma_index = np.sqrt(d_d ** 2 + d_r ** 2)

                        if np.abs(gamma_index) < gamma_index_threshold:
                            gamma_index_threshold = np.abs(gamma_index)

                    gamma_array[index] = gamma_index_threshold

            with open(filepath + r'\gamy\\'
                      + f'gamma_array.csv', 'w', newline='') as f:
                gamma_rows = list(gamma_array)
                writer = csv.writer(f, lineterminator='\r\n')
                writer.writerows(gamma_rows)

        return gamma_array


class Visualization(Gamma, RTStructures):
    def __init__(self, film_dose, film_path, RS_name, RS_filename, TPS_dose, shift, CT_print):
        RTStructures.__init__(self, RS_name, RS_filename)
        self.film_dir, self.film_name = (os.path.normpath(film_path).split(os.sep)[-3]
                                                     , os.path.basename(film_path))
        self.TPS_dose = TPS_dose
        self.film_dose = np.where(np.array(film_dose) < np.max(self.TPS_dose) * 2, film_dose, 0)
        self.profiles = self.dose_profiles(shift)

        self.print_film_TPS_dose(TPS_dose, film_dose, shift, gamma_options.RS_name[0], CT_print, self.profiles[2])
        #   calc_bool = input("Spustit výpočet nebo vyhodnotit hodnotu pro další film?  v(ýpočet)/d(alší)") or "v"
        calc_bool = "n"
        if calc_bool == "v":
            print("______________________")
            print("MANUÁLNÍ KOREKCE SHIFTU:")
            print("______________________")
            shift_hor_mod = input('Horizontální posun v shiftu (v pixelech) (šoupu dutý puntíky):') or 0
            shift_ver_mod = input('Vertikální posun v shiftu (v pixelech) (šoupu dutý puntíky):') or 0
            dose_scale = input('Hodnota škálování dávky podle reference (v %): ') or 0
            film_dose *= 1 + np.float64(dose_scale)/100
            shift[0] += int(shift_ver_mod)
            shift[1] += int(shift_hor_mod)

            if shift_hor_mod != 0 or shift_ver_mod != 0:
                self.print_film_TPS_dose(TPS_dose, film_dose, shift, gamma_options.RS_name[0], CT_print, self.profiles[2])
                shift_hor_mod = int(input('Horizontální posun v shiftu (v pixelech) (šoupu dutý puntíky):'))
                shift_ver_mod = int(input('Vertikální posun v shiftu (v pixelech) (šoupu dutý puntíky):'))
                shift[0] += shift_ver_mod
                shift[1] += shift_hor_mod

        self.profiles = self.dose_profiles(shift)
        self.TPS_mean, self.film_mean = self.print_film_TPS_dose(TPS_dose, film_dose, shift, gamma_options.RS_name[0], CT_print, self.profiles[2])
        Gamma.__init__(self, film_dose, TPS_dose, shift, self.RS_mask_in_gamma_res)
        self.DVH_data = self.DVH()
        self.hist = self.gamma_histogram()
        self.print_results()

        # flag = input("Pokračovat? (y/n)")
        flag = "y"
        if flag not in ["y", "Y", "yes", "Yes"]:
            quit()


    def print_film_TPS_dose(self, TPS_dose, film_dose, shift, RS_name, CT_print, profiles):
        RS = self.RS_mask_in_gamma_res[RS_name]
        film_visualisation = np.where(np.logical_and(film_dose <= 12, film_dose >= 0), film_dose, 0)
        film_visualisation_array = np.zeros_like(RS)

        for index, value in np.ndenumerate(film_visualisation):
            film_visualisation_array[index[0] + shift[0], index[1] + shift[1]] = value

        D_mean_film = np.mean(film_visualisation_array[RS > 0])
        film_to_visualise = film_visualisation_array / np.max(film_visualisation_array)
        film_visualisation[profiles[0], :] = 0.5
        film_visualisation[:, profiles[1]] = 0.5

        TPS_visualisation = cv.resize(TPS_dose, CT_print.shape[::-1])
        D_mean_TPS = np.mean(TPS_visualisation[RS > 0])
        TPS_visualisation = TPS_visualisation/np.max(TPS_visualisation)
        TPS_visualisation[profiles[0] + shift[0], :] = 0.5
        TPS_visualisation[:, profiles[1] + shift[1]] = 0.5

        CT_visualisation = CT_print/np.max(CT_print)
        RS = cv.resize(RS, CT_print.shape[::-1])
        # D_mean_film = np.mean(film_to_visualise[RS > 0] * np.max(film_visualisation_array))
        # D_mean_TPS = np.mean(TPS_visualisation[RS > 0] * np.max(TPS_dose))

        print(f'Hodnoty průměrné dávky v TPS a ve filmu v oblasti {RS_name}:')
        print('______________________________________________________')
        print(f'Průměrná dávka v TPS v oblasti CTV je: {D_mean_TPS:.2f} Gy')
        print(f'Průměrná dávka na filmu v oblasti CTV je: {D_mean_film:.2f} Gy')
        print(f'Rozdíl v průměrné dávce v CTV mezi TPS a filmem je: {(D_mean_TPS - D_mean_film) / D_mean_TPS * 100:.2f} %')

        center_y, center_x = np.array(RS.shape) // 2

        margin = 600
        y_min = max(0, center_y - margin)
        y_max = min(RS.shape[0], center_y + margin)
        x_min = max(0, center_x - margin)
        x_max = min(RS.shape[1], center_x + margin)

        TPS_visualisation = TPS_visualisation[y_min:y_max, x_min:x_max]
        film_to_visualise = film_to_visualise[y_min:y_max, x_min:x_max]
        RS = RS[y_min:y_max, x_min:x_max]
        CT_visualisation = CT_visualisation[y_min:y_max, x_min:x_max]

        try:
            plt.figure(figsize=(6.4 * 2, 4.8 * 2))
            plt.imshow(TPS_visualisation, cmap='gray', alpha=1)
            plt.imshow(film_to_visualise, cmap='seismic', alpha=0.2)
            plt.imshow(RS + CT_visualisation, alpha=0.2, cmap='cividis')
            plt.show()

            return D_mean_TPS, D_mean_film
        except IndexError:
            print('Potenciální neshoda velikosti tisknutých matic.')

    @staticmethod
    def max_array_points(n, array):
        gamma_values = array.flatten()
        top5_array = np.argpartition(np.abs(gamma_values), kth=-n)[-n:]
        top5_array = top5_array[np.argsort(-gamma_values[top5_array])]

        return np.unravel_index(top5_array, array.shape)

    def dose_profiles(self, shift):
        max_points_in_array = self.max_array_points(5000, self.film_dose)
        dose_center_coordinate_ver, dose_center_coordinate_hor = (int(np.mean(max_points_in_array[0])),
                                                                  int(np.mean(max_points_in_array[1])))

        hor_vector_film_TPS = (self.film_dose[dose_center_coordinate_ver, :],
                               self.TPS_dose[dose_center_coordinate_ver + shift[0], :])

        ver_vector_film_TPS = (self.film_dose[:, dose_center_coordinate_hor],
                               self.TPS_dose[:, dose_center_coordinate_hor + shift[1]])

        return hor_vector_film_TPS, ver_vector_film_TPS, (dose_center_coordinate_ver, dose_center_coordinate_hor)

    def gamma_histogram(self):  # ZKONTROLOVÁNO, NEOTESTOVÁNO
        """
        Určuje distribuce gamma hodnot...dostanu histogram distribuce. Může být dobré, pro určení jestli je dávka
        případně všeobecně vyšší nebo nižší.
        :return:
        """
        bin_width = 0.1
        bins_interval = (-5, 5)
        bins_count = int(np.abs(bins_interval[0] - bins_interval[1]) / bin_width) - 1
        counts, bins = np.histogram(self.gamma_array, range=bins_interval, bins=bins_count)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        return bins, counts, bin_centers

    def DVH(self):  # ZKONTROLOVÁNO, NEOTESTOVÁNO

        DVHs_for_RS = {}  # List v listu v listu
        doses = [self.film_dose, self.TPS_dose]
        dose_names = ['film', 'TPS']

        for evaluated_dose, evaluated_dose_name in zip(doses, dose_names):
            for rs_key, rs_mask in self.RS_mask_in_gamma_res.items():
                # V případě vetšího množství struktur, vytvoří hodnoty pro DVH pro každou RTS uvedenou
                print_counter = 0
                rs_mask = np.where(rs_mask > 0, True, False)

                if evaluated_dose_name == "film":
                    try:
                        rs_mask = rs_mask[self.shift[0]: self.shift[0] + evaluated_dose.shape[0],
                                          self.shift[1]: self.shift[1] + evaluated_dose.shape[1]]
                        if rs_mask.shape != evaluated_dose.shape:
                            print('Tvary masky a dávky se při aplikaci za účelem vytvoření RS_dose neshodují.')
                            exit()
                        RS_dose = evaluated_dose[rs_mask]
                    except IndexError:  # Aby program nespadl, když bude struktura mimo oblast, kterou zabírá film
                        print('Část nebo celá RS se nachází mimo oblast filmu.')
                        exit()
                else:
                    try:
                        RS_dose = evaluated_dose[rs_mask]
                        if rs_mask.shape != evaluated_dose.shape:
                            print('Tvary masky a dávky se při aplikaci za účelem vytvoření RS_dose neshodují.')
                            exit()
                    except IndexError:  # Aby program nespadl, když bude struktura mimo oblast, kterou zabírá film
                        print('Část nebo celá RS se nachází mimo oblast TPS_dose.')
                        exit()

                norm_dose = RS_dose / np.max(doses[1])

                DVH_max_dose = 1
                if np.max(norm_dose) > DVH_max_dose:
                    DVH_max_dose = np.max(norm_dose) + 0.01

                pixel_count = np.size(RS_dose)
                volume_cumulative = []

                dose_cumulative = np.linspace(0, DVH_max_dose, 100)

                for value in dose_cumulative:
                    mask = np.where(norm_dose > value, True, False)
                    if pixel_count == 0:
                        volume = 0
                    else:
                        volume = np.size(norm_dose[mask]) / pixel_count
                    volume_cumulative.append(volume)

                key = rs_key + '_' + evaluated_dose_name
                DVHs_for_RS[key] = [dose_cumulative, np.array(volume_cumulative)]

        return DVHs_for_RS

    def print_results(self):

        if 'AP' in self.film_name:
            geom = 'AP'
        else:
            geom = 'LR'

        txt = (f"Vyhodnocení filmu v {geom} orientaci (podle laterální osy) za pomoci absolutní globální gamma analýzy \n"
               f"s parametry DTA={gamma_options.DTA}, DD={gamma_options.DD}, cut-off={gamma_options.cut_off} % "
               f"a s rozlišením {gamma_options.resolution} mm.\n"
               f"D_mean_TPS = {self.TPS_mean:.2f} Gy;"
               f"  D_mean_film = {self.film_mean:.2f} Gy\n")

        data = {
            'Gamma Map': self.gamma_array,
            'Gamma Histogram': self.hist,
            'Dose Profiles': self.profiles,
            'DVH': self.DVH_data
        }

        # Create the layout for the subplots
        layout = [
            ['Gamma Map',         'Gamma Histogram'],
            ['Dose Profiles_ver', 'Dose Profiles_hor'],
            ['DVH',               'DVH']
        ]

        fig, axs = plt.subplot_mosaic(layout)


        # Plot the data in each subplot
        for label, ax in axs.items():
            trans = mtransforms.ScaledTranslation(5 / 72, -5 / 72, fig.dpi_scale_trans)

            ax.set_title(label, fontstyle='italic')

            ax.text(0.0, 1, label, transform=ax.transAxes + trans,
                    fontsize='small', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='0.95', edgecolor='none', pad=3.0))

            if label == 'Gamma Map':
                ax.imshow(data[label], vmin=-8, vmax=8, cmap='seismic', alpha=0.6)

                ax.set_title(f'{label}: {self.pass_rate:.2f} %', fontstyle='italic')

                gamma_image = axs['Gamma Map'].imshow(self.gamma_array, vmin=-8, vmax=8, cmap='seismic')

            elif label == 'Gamma Histogram':
                ax.bar(data[label][2], data[label][1], width=data[label][2][1] - data[label][2][0],
                       linewidth=0, edgecolor='black', color='khaki')

            elif label == 'Dose Profiles_ver':
                label = 'Dose Profiles'
                y_coordinates_film = self.film_dose.shape[0]
                y_range_film = np.linspace(0 + self.shift[0], y_coordinates_film + self.shift[0], y_coordinates_film)
                mean_dose_f = np.mean(data[label][1][0])
                y_coordinates_TPS = self.TPS_dose.shape[0]

                y_range_TPS = np.linspace(0, y_coordinates_TPS, y_coordinates_TPS)
                mean_dose_TPS = np.mean(data[label][1][1])
                ax.set_ylim(0.1, np.max(self.TPS_dose) * 2)
              #  ax.set_xlim(500, 700)
                ax.plot(y_range_film, data[label][1][0], label=f'Film')
                ax.plot(y_range_TPS, data[label][1][1], label=f'TPS')
                ax.legend()

            elif label == 'Dose Profiles_hor':
                label = 'Dose Profiles'
                x_coordinates_TPS = self.TPS_dose.shape[1]
                x_range_TPS = np.linspace(0, x_coordinates_TPS, x_coordinates_TPS)
                x_coordinates_film = self.film_dose.shape[1]
                x_range_film = np.linspace(0 + self.shift[1], x_coordinates_film + self.shift[1], x_coordinates_film)

                ax.set_ylim(0.1, np.max(self.TPS_dose) * 2)
                #  ax.set_xlim(1200, 1400)

                ax.plot(x_range_film, data[label][0][0], label='Film')
                ax.plot(x_range_TPS, data[label][0][1], label='TPS')
                ax.legend()

            elif label == 'DVH':
                for key, value in data[label].items():
                    if 'film' in key:
                        ax.plot(value[0], value[1], label=key, linestyle='dashed')
                    else:
                        ax.plot(value[0], value[1], label=key, linestyle='solid')
                ax.legend()
                ax.set_xticks(np.arange(0, 1.1, 0.1))

        plt.gcf().set_facecolor("gray")

        gamma_map_shape = np.shape(data['Gamma Map'])

        dx = cv.Scharr(self.RS_mask_in_gamma_res[gamma_options.RS_name[0]][self.shift[0]: self.shift[0] + gamma_map_shape[0],
                       self.shift[1]: self.shift[1] + gamma_map_shape[1]], cv.CV_64F, 1, 0)
        dy = cv.Scharr(self.RS_mask_in_gamma_res[gamma_options.RS_name[0]][self.shift[0]: self.shift[0] + gamma_map_shape[0],
                       self.shift[1]: self.shift[1] + gamma_map_shape[1]], cv.CV_64F, 0, 1)

        RS_border = dx + dy
        RS_border[RS_border == 0] = np.nan

        axs['Gamma Map'].imshow(RS_border, cmap='binary')

        # axs['Gamma Map'].set_facecolor('gray')
        # axs['Dose Profiles'].set_facecolor('lightgray')
        # axs['DVH'].set_facecolor('lightgray')
        # axs['Gamma Histogram'].set_facecolor('lightgray')
        fig.colorbar(gamma_image, ax=axs['Gamma Map'], fraction=0.04, aspect=55)
        fig.set_size_inches(7.2, 10.28, forward=False)
        plt.xlabel(txt, ha='center')
        plt.tight_layout()
        plt.title(self.film_name)
        result_dir = fr'{filepath}\Vysledky\\' + self.film_dir

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        plt.savefig(os.path.join(result_dir, self.film_name))
        plt.show()


def get_films_based_on(film_directory, key_word=None):
    all_films_list = os.listdir(film_directory)
    selected_films = []

    for film in all_films_list:
        if key_word:
            if key_word in film:
                selected_films.append(film)
        else:
            if film.endswith(".tif"):
                selected_films.append(film)

    return selected_films


def load_paths(film, ct_dirs, tps_dir):
    # Mapping film name patterns to CT paths, TPS paths, and RS filenames
    calc_alg = "MC"
    path_mappings = {
        # AP FIX
        ('AP', 'FIX', '01', 'ITV'): (ct_dirs[1], tps_dir + fr'\ITV_01_FIX_90_{calc_alg}.dcm', r'\ITV_01_FIX_90_RS.dcm'),
        ('AP', 'FIX', '07', 'ITV'): (ct_dirs[1], tps_dir + fr'\ITV_07_FIX_90_{calc_alg}.dcm', r'\ITV_07_FIX_90_RS.dcm'),
        ('AP', 'FIX', 'TRACK', '_XL_'): (ct_dirs[1], tps_dir + fr'\TRACK_FIX_XL_90_{calc_alg}.dcm', r'\TRACK_FIX_XL_90_RS.dcm'),
        ('AP', 'FIX', 'TRACK', '_L_'):  (ct_dirs[1], tps_dir + fr'\TRACK_FIX_L_90_{calc_alg}.dcm', r'\TRACK_FIX_L_90_RS.dcm'),
        ('AP', 'FIX', 'TRACK', '_S_'):  (ct_dirs[1], tps_dir + fr'\TRACK_FIX_S_90_{calc_alg}.dcm', r'\TRACK_FIX_S_90_RS.dcm'),
        ('AP', 'FIX', 'STAT',  '_XL_'): (ct_dirs[1], tps_dir + fr'\TRACK_FIX_XL_90_{calc_alg}.dcm', r'\TRACK_FIX_XL_90_RS.dcm'),
        ('AP', 'FIX', 'STAT',  '_L_'): (ct_dirs[1], tps_dir + fr'\TRACK_FIX_L_90_{calc_alg}.dcm', r'\TRACK_FIX_L_90_RS.dcm'),
        ('AP', 'FIX', 'STAT',  '_S_'): (ct_dirs[1], tps_dir + fr'\TRACK_FIX_S_90_{calc_alg}.dcm', r'\TRACK_FIX_S_90_RS.dcm'),
        ('AP', 'FIX', 'SB',    '_XL_'): (ct_dirs[1], tps_dir + fr'\SB_FIX_XL_90_{calc_alg}.dcm', r'\SB_FIX_XL_90_RS.dcm'),
        ('AP', 'FIX', 'SB',    '_L_'): (ct_dirs[1], tps_dir + fr'\SB_FIX_L_90_{calc_alg}.dcm', r'\SB_FIX_L_90_RS.dcm'),
        ('AP', 'FIX', 'SB',    '_XS_'): (ct_dirs[1], tps_dir + fr'\SB_FIX_XS_90_{calc_alg}.dcm', r'\SB_FIX_XS_90_RS.dcm'),
        ('AP', 'FIX', 'SB',    '_S_'): (ct_dirs[1], tps_dir + fr'\SB_FIX_S_90_{calc_alg}.dcm', r'\SB_FIX_S_90_RS.dcm'),

        # AP MLC
        ('AP', 'MLC', '01', 'ITV'): (ct_dirs[1], tps_dir + fr'\ITV_01_MLC_90_{calc_alg}.dcm', r'\ITV_01_MLC_90_RS.dcm'),
        ('AP', 'MLC', '07', 'ITV'): (ct_dirs[1], tps_dir + fr'\ITV_07_MLC_90_{calc_alg}.dcm', r'\ITV_07_MLC_90_RS.dcm'),
        ('AP', 'MLC', 'TRACK', '_XL_'): (ct_dirs[1], tps_dir + fr'\TRACK_MLC_XL_90_{calc_alg}.dcm', r'\TRACK_MLC_XL_90_RS.dcm'),
        ('AP', 'MLC', 'TRACK', '_L_'):  (ct_dirs[1], tps_dir + fr'\TRACK_MLC_L_90_{calc_alg}.dcm', r'\TRACK_MLC_L_90_RS.dcm'),
        ('AP', 'MLC', 'TRACK', '_S_'):  (ct_dirs[1], tps_dir + fr'\TRACK_MLC_S_90_{calc_alg}.dcm', r'\TRACK_MLC_S_90_RS.dcm'),
        ('AP', 'MLC', 'STAT',  '_XL_'):(ct_dirs[1], tps_dir + fr'\TRACK_MLC_XL_90_{calc_alg}.dcm', r'\TRACK_MLC_XL_90_RS.dcm'),
        ('AP', 'MLC', 'STAT',  '_L_'): (ct_dirs[1], tps_dir + fr'\TRACK_MLC_L_90_{calc_alg}.dcm', r'\TRACK_MLC_L_90_RS.dcm'),
        ('AP', 'MLC', 'STAT',  '_S_'): (ct_dirs[1], tps_dir + fr'\TRACK_MLC_S_90_{calc_alg}.dcm', r'\TRACK_MLC_S_90_RS.dcm'),
        ('AP', 'MLC', 'SB',    '_XL_'):(ct_dirs[1], tps_dir + fr'\SB_MLC_XL_90_{calc_alg}.dcm', r'\SB_MLC_XL_90_RS.dcm'),
        ('AP', 'MLC', 'SB',    '_L_'): (ct_dirs[1], tps_dir + fr'\SB_MLC_L_90_{calc_alg}.dcm', r'\SB_MLC_L_90_RS.dcm'),
        ('AP', 'MLC', 'SB',    '_XS_'):(ct_dirs[1], tps_dir + fr'\SB_MLC_XS_90_{calc_alg}.dcm', r'\SB_MLC_XS_90_RS.dcm'),
        ('AP', 'MLC', 'SB',    '_S_'): (ct_dirs[1], tps_dir + fr'\SB_MLC_S_90_{calc_alg}.dcm', r'\SB_MLC_S_90_RS.dcm'),

        # LR FIX
        ('LR', 'FIX', '01', 'ITV'): (ct_dirs[0], tps_dir + fr'\ITV_01_FIX_0_{calc_alg}.dcm', r'\ITV_01_FIX_0_RS.dcm'),
        ('LR', 'FIX', '07', 'ITV'): (ct_dirs[0], tps_dir + fr'\ITV_07_FIX_0_{calc_alg}.dcm', r'\ITV_07_FIX_0_RS.dcm'),
        ('LR', 'FIX', 'STAT',  '_XL_'):(ct_dirs[0], tps_dir + fr'\TRACK_FIX_XL_0_{calc_alg}.dcm', r'\TRACK_FIX_XL_0_RS.dcm'),
        ('LR', 'FIX', 'STAT',  '_L_'): (ct_dirs[0], tps_dir + fr'\TRACK_FIX_L_0_{calc_alg}.dcm', r'\TRACK_FIX_L_0_RS.dcm'),
        ('LR', 'FIX', 'STAT',  '_S_'): (ct_dirs[0], tps_dir + fr'\TRACK_FIX_S_0_{calc_alg}.dcm', r'\TRACK_FIX_S_0_RS.dcm'),
        ('LR', 'FIX', 'TRACK', '_XL_'):(ct_dirs[0], tps_dir + fr'\TRACK_FIX_XL_0_{calc_alg}.dcm', r'\TRACK_FIX_XL_0_RS.dcm'),
        ('LR', 'FIX', 'TRACK', '_L_'):  (ct_dirs[0], tps_dir + fr'\TRACK_FIX_L_0_{calc_alg}.dcm', r'\TRACK_FIX_L_0_RS.dcm'),
        ('LR', 'FIX', 'TRACK', '_S_'):  (ct_dirs[0], tps_dir + fr'\TRACK_FIX_S_0_{calc_alg}.dcm', r'\TRACK_FIX_S_0_RS.dcm'),
        ('LR', 'FIX', 'SB',    '_XL_'): (ct_dirs[0], tps_dir + fr'\SB_FIX_XL_0_{calc_alg}.dcm', r'\SB_FIX_XL_0_RS.dcm'),
        ('LR', 'FIX', 'SB',    '_L_'):  (ct_dirs[0], tps_dir + fr'\SB_FIX_L_0_{calc_alg}.dcm', r'\SB_FIX_L_0_RS.dcm'),
        ('LR', 'FIX', 'SB',    '_XS_'): (ct_dirs[0], tps_dir + fr'\SB_FIX_XS_0_{calc_alg}.dcm', r'\SB_FIX_XS_0_RS.dcm'),
        ('LR', 'FIX', 'SB',    '_S_'):  (ct_dirs[0], tps_dir + fr'\SB_FIX_S_0_{calc_alg}.dcm', r'\SB_FIX_S_0_RS.dcm'),

        # LR MLC
        ('LR', 'MLC', '01', 'ITV'): (ct_dirs[0], tps_dir + fr'\ITV_01_MLC_0_{calc_alg}.dcm', r'\ITV_01_MLC_0_RS.dcm'),
        ('LR', 'MLC', '07', 'ITV'): (ct_dirs[0], tps_dir + fr'\ITV_07_MLC_0_{calc_alg}.dcm', r'\ITV_01_MLC_0_RS.dcm'),
        ('LR', 'MLC', 'STAT',  '_XL_'): (ct_dirs[0], tps_dir + fr'\TRACK_MLC_XL_0_{calc_alg}.dcm', r'\TRACK_MLC_XL_0_RS.dcm'),
        ('LR', 'MLC', 'STAT',  '_L_'): (ct_dirs[0], tps_dir + fr'\TRACK_MLC_L_0_{calc_alg}.dcm', r'\TRACK_MLC_L_0_RS.dcm'),
        ('LR', 'MLC', 'STAT',  '_S_'): (ct_dirs[0], tps_dir + fr'\TRACK_MLC_S_0_{calc_alg}.dcm', r'\TRACK_MLC_S_0_RS.dcm'),
        ('LR', 'MLC', 'TRACK', '_XL_'): (ct_dirs[0], tps_dir + fr'\TRACK_MLC_XL_0_{calc_alg}.dcm', r'\TRACK_MLC_XL_0_RS.dcm'),
        ('LR', 'MLC', 'TRACK', '_L_'): (ct_dirs[0], tps_dir + fr'\TRACK_MLC_L_0_{calc_alg}.dcm', r'\TRACK_MLC_L_0_RS.dcm'),
        ('LR', 'MLC', 'TRACK', '_S_'): (ct_dirs[0], tps_dir + fr'\TRACK_MLC_S_0_{calc_alg}.dcm', r'\TRACK_MLC_S_0_RS.dcm'),
        ('LR', 'MLC', 'SB',    '_XL_'): (ct_dirs[0], tps_dir + fr'\SB_MLC_XL_0_{calc_alg}.dcm', r'\SB_MLC_XL_0_RS.dcm'),
        ('LR', 'MLC', 'SB',    '_L_'): (ct_dirs[0], tps_dir + fr'\SB_MLC_L_0_{calc_alg}.dcm', r'\SB_MLC_L_0_RS.dcm'),
        ('LR', 'MLC', 'SB',    '_XS_'): (ct_dirs[0], tps_dir + fr'\SB_MLC_XS_0_{calc_alg}.dcm', r'\SB_MLC_XS_0_RS.dcm'),
        ('LR', 'MLC', 'SB',    '_S_'): (ct_dirs[0], tps_dir + fr'\SB_MLC_S_0_{calc_alg}.dcm', r'\SB_MLC_S_0_RS.dcm'),
    }

    # Find matching paths for the film
    for (key1, key2, key3, key4), (ct_path, tps_path, rs_filename) in path_mappings.items():

        if key1 in film and key2 in film and key3 in film and key4 in film:
            print('Nahrávám CT:', ct_path)
            print('Nahrávám TPS:', tps_path)
            print('Nahrávám RS:', rs_filename)

            return ct_path, tps_path, rs_filename

    return None, None, None


def main(CT_path, film_path, TPS_path, RS_filename):
    CT_slices = CT(os.path.abspath(CT_path))

    """Nahrání bodů z CT pro účely registrace a škálování """
    CT_kp, CT_max_min_vertical, CT_max_min_horizontal = CT_slices.get_key_points(do_print=False)
    CT_vert_dist = np.abs(CT_max_min_vertical[0] - CT_max_min_vertical[1])
    CT_hor_dist = np.abs(CT_max_min_horizontal[0] - CT_max_min_horizontal[1])

    film = Film(os.path.abspath(film_path))

    """Nahrání stejných bodů jako v CT...opět pro účely registrace a škálování"""
    film_kp_coordinates, film_max_min_vertical, film_max_min_horizontal, prepd_film \
        = film.get_key_points_and_prep(do_print=False)
    film_vert_dist = np.abs(film_max_min_vertical[0] - film_max_min_vertical[1])
    film_hor_dist = np.abs(film_max_min_horizontal[0] - film_max_min_horizontal[1])

    """Škálování filmu vzhledem k CT a tím pádem i dávce, použití původního rozlišení filmu přizpůsobeného na nastavené rozlišení.
        0,2 % chyba zaokroulením na reálný počet pixelů"""
    film_CT_ratio_vert = film_vert_dist / CT_vert_dist
    film_CT_ratio_hor = film_hor_dist / CT_hor_dist

    """Přeškálování a zisk konečných dávek použitých pro gama analýzu"""
    film_resized, max_min_ver, max_min_hor = film.resize_film_and_get_new_coors(film_CT_ratio_vert, film_CT_ratio_hor,
                                                                                prepd_film)

    shift = [int(np.abs(max_min_ver[0] - CT_max_min_vertical[0])),
             int(np.abs(max_min_hor[0] - CT_max_min_horizontal[0]))]

    film_dose = film.apply_calibration(film_resized)
    # film_dose *= 1.075
    # if '90' in film_path:
    #     film_dose = np.fliplr(film_dose)
    # if "0" in film_path:
    #     film_dose = np.flipud(film_dose)
    RT_dose = Dose(TPS_path)
    TPS_dose = RT_dose.get_dose

    Visualization(film_dose=film_dose, film_path=film_path, RS_name=gamma_options.RS_name, RS_filename=RS_filename, TPS_dose=TPS_dose, shift=shift, CT_print=CT_kp)

    end_time = time.time()
    print(f"Time: {end_time - start_time} s")


def process_films(film, ct_dirs, tps_dirs, actual_film_directory):

    print("NÁZEV FILMU A SOUPIS CEST K OSTATNÍM DCM:")
    print('____________________________________\n')
    print(f'Nahrávám film: "{film}"; ze složky {actual_film_directory}')

    film_path = os.path.join(actual_film_directory, film)

    # Load paths based on film name
    ct_path, tps_path, rs_filename = load_paths(film, ct_dirs, tps_dirs)

    print('\n')

    main(ct_path, film_path, tps_path, rs_filename)


if __name__ == "__main__":
    # Define file paths and user details
    PC = "kubad"

    actual_film_directory = fr"{filepath}\Filmy\1312_MLC_EBT4\Averaged_images"

    CT_dirs = [fr"{filepath}\CT&Dose\srpen_24\CT\SBRT_0",
               fr"{filepath}\CT&Dose\srpen_24\CT\SBRT_90"]

    TPS_dir = (fr"{filepath}\CT&Dose\srpen_24\RTD_RTS\0819_2024")


    # Get films from the actual directory

    # films = get_films_based_on(actual_film_directory)

    # Process films
    # Nahrávám konkrétní film, protože vím, že dřív vycházel dobře. Beru ho jako referenci funkčnosti kódu.

    film = rf"C:\Users\kubad\Dropbox\Zlodějíčkovo slůj\Python\UVN_scripts\Film_dosimetry\Filmy\1312_MLC_EBT4\Averaged_images\SB_MLC_AP_XS_average.tif"

    process_films(film, CT_dirs, TPS_dir, actual_film_directory)
