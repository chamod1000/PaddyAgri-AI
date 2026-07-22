#!/usr/bin/env python3
"""
Script to organize PDF files into categorized folders.
"""

import os
import shutil

# Define the mapping of files to target directories
file_mapping = {
    "Paddy_Diseases_and_Pests": [
        "Paddy-disease-1.pdf",
        "Paddy-pest.pdf",
        "IPMcalender.jpg",
        "KabanikaPohora_2021.pdf",
        "Wee_vagawa_pohora_nirdeshaya_2013_9.pdf",
        "ආරක්_ෂිත ගෘහ තුල බෝග වගාවට අත්වැලක්.pdf",
        "වී වගාවේ ක්ෂේත්_ර අත්පොත.pdf",
    ],
    "Fertilizer_and_Chemicals": [
        "Organic-Fertilizer.pdf",
        "OrganicB.pdf",
        "control_of_pesticides_act_no_33_of_1980.pdf",
        "control_of_pesticides_amendment_act_no_6_of_1994.pdf",
    ],
    "Policy_and_Acts": [
        "1951 අංක 25 දරණ පාංශු සංරක්ෂණ පනත (ඉංග්‍රීසි).pdf",
        "1996 අංක 24 දරණ පාංශු සංරක්ෂණ පනත (ඉංග්‍රීසි).pdf",
        "1999 අංක 35 දරණ ශාක ආරක්ෂක පනත.pdf",
        "seed act-eng_2003.pdf",
    ],
    "General_Cultivation_Guidelines": [
        "Paddy Cultivation.pdf",
        "Paddy Seed Production Sinhala Full Book.pdf",
        "Recommended Rice Varieties in Sri Lanka_1958_2023.pdf",
        "Rice Varietal 2021-23 final.pdf",
        "Rice-Congress-2010.pdf",
        "Rice-Varietal2015.pdf",
        "Rice-Varietal2016.pdf",
        "Climate Smart technology 11-6-2024.pdf",
        "SL E-Agri Strategy.pdf",
        "VRC 2015.pdf",
        "RainProtectedHouse_book.pdf",
    ],
    "Imports_Quarantine": [
        "10_phytosanitary_certificate.pdf",
        "1_Import-of-Plants-Plant-Products-and-Organisms.pdf",
        "2_General-Procedure-for-Import-of-Plants-Plant-Products.pdf",
        "2009_1623_11_Sin.pdf",
        "3_NPQS_Application.pdf",
        "4_Application-for-soil-importation-permit.pdf",
        "5_Application-for-organism-importation-permit.pdf",
        "6_Information-required-for-quarantine-clearance.pdf",
        "7_Licence-to-Import-Fresh-Fruit.pdf",
        "8_Permit-to-import-fresh-fruit.pdf",
        "9_application_for_a_phytosanitary_certificate.pdf",
        "2009 අංක 01 දරණ පස සංරක්ෂණ නියෝග (ඉංග්‍රීසි).pdf",
    ],
}

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for target_dir, files in file_mapping.items():
        target_path = os.path.join(base_dir, target_dir)
        os.makedirs(target_path, exist_ok=True)

        for filename in files:
            src = os.path.join(base_dir, filename)
            dst = os.path.join(target_path, filename)

            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"Moved: {filename} -> {target_dir}/")
            else:
                print(f"NOT FOUND: {filename}")

    print("\nDone organizing files!")

if __name__ == "__main__":
    main()