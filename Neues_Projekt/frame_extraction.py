import os, cv2

def frame_extraction():
    # inspired by https://stackoverflow.com/questions/63666638/convert-video-to-frames-in-python-1-fps, accessed Oct. 26th, 2023
    KPS = fpsextractor['kps'] # Target Keyframes Per Second
    VIDEO_PATH = os.path.join(dirpath, dirname, filename) # path to current video
    FPS_OUPUT= fpsextractor['output']
    YEAR = dirname.replace("ADs_IG_", "")
    YEAR_OUTPUT_DIR = FPS_OUPUT+ "/" + YEAR
    OUTPUT_PATH = YEAR_OUTPUT_DIR + "/" + dirname + "/"
    EXTENSION = "." + fpsextractor['extension'] # file extension of exported images
    fileNameOfVideoWithoutExtension = filename[:-len("." + fpsextractor['extension'])];
    # print(OUTPUT_PATH) # e.g., ./outputs/fps_extractor/ADs_IG_2018/
    #print(OUTPUT_PATH + fileNameOfVideoWithoutExtension) # e.g., ./outputs/fps_extractor/ADs_IG_2018/AD0576

    # Ordner erstellen, in welchem je Video die Frames gepseichert werden
    # Ordner mit Jahreszahl
    if not os.path.exists(FPS_OUPUT):
        os.mkdir(FPS_OUPUT)
    if not os.path.exists(YEAR_OUTPUT_DIR):
        os.mkdir(YEAR_OUTPUT_DIR)        
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    # Unterordner mit Video-ID
    if not os.path.exists(OUTPUT_PATH + fileNameOfVideoWithoutExtension):
        os.mkdir(OUTPUT_PATH + fileNameOfVideoWithoutExtension)

    # print(KPS, IMAGE_PATH, EXTENSION)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    # exit()
    hop = round(fps / KPS)
    curr_frame = 0
    while(True):
        ret, frame = cap.read()
        if not ret: break
        if curr_frame % hop == 0:
            name = OUTPUT_PATH + fileNameOfVideoWithoutExtension + "/" + fileNameOfVideoWithoutExtension + "_Frame_" + str(curr_frame) + EXTENSION
            # print(name)
            cv2.imwrite(name, frame)
        curr_frame += 1
    cap.release()