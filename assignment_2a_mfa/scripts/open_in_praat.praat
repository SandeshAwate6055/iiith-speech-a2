# Praat helper script: opens all 6 corpus WAVs paired with their english_us_arpa TextGrids.
# Usage in Praat: Praat -> Open Praat script... -> select this file -> Run (Ctrl+R)

form Open MFA Alignments
    sentence Base_dir D:/IIITH A2/assignment_2a_mfa
endform

files$ [1] = "f2b/F2BJRLP1"
files$ [2] = "f2b/F2BJRLP2"
files$ [3] = "f2b/F2BJRLP3"
files$ [4] = "isle_sess0131/ISLE_SESS0131_BLOCKD02_01_sprt1"
files$ [5] = "isle_sess0131/ISLE_SESS0131_BLOCKD02_02_sprt1"
files$ [6] = "isle_sess0131/ISLE_SESS0131_BLOCKD02_03_sprt1"

for i from 1 to 6
    s = Read from file: base_dir$ + "/corpus/" + files$ [i] + ".wav"
    t = Read from file: base_dir$ + "/outputs/textgrids/english_us_arpa/" + files$ [i] + ".TextGrid"
endfor

# Select the first ISLE pair for immediate inspection
selectObject: "Sound ISLE_SESS0131_BLOCKD02_01_sprt1"
plusObject: "TextGrid ISLE_SESS0131_BLOCKD02_01_sprt1"
View & Edit
