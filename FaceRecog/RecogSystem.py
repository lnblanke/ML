# A complete system of face recognization
# @Time: 8/17/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: RecogSystem.py

import numpy, dlib, os, cv2, time

def recognize(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    names = []
    desp = []

    for _, _, files in os.walk("Dataset"):
        for file in files:
            pic = cv2.imread(os.path.join("Dataset", file))

            faces = detector(pic, 1)

            for i, face in enumerate(faces):
                shape = predictor(pic, face)

                descriptor = model.compute_face_descriptor(pic, shape)

                vec = numpy.array(descriptor)

                desp.append(vec)
                names.append(file[: -4])

    faces = detector(img, 1)

    dist = []
    for i, face in enumerate(faces):
        shape = predictor(img, face)

        descriptor = model.compute_face_descriptor(img, shape)

        vect = numpy.array(descriptor)

        for iter in desp:
            d = numpy.linalg.norm(iter - vect)

            dist.append(d)

    diction = dict(zip(names, dist))

    diction = sorted(diction.items(), key = lambda d: d[1])

    return diction

while 1:
    print("Menu")
    print("1. Add a new person")
    print("2. Recognize")
    print("3. Delete a existing person")
    print("4. Exit")

    choose = int(input(":"))

    if choose == 1:
        usrname = input("Please input username: ")
        print("The photo will be taken shortly...")

        time.sleep(1)

        print("Taking picture...")

        cap = cv2.VideoCapture(0)
        _, img = cap.read()

        cap.release()

        print("Finding interruptions...")

        diction = recognize(img)

        if diction and float(diction[0][1]) <= 0.7:
            print("Adding user failed")
            print("You are", diction[0][0])

            continue

        print("Saving...")

        cv2.imwrite(os.path.join("Dataset", usrname + ".jpg"), img)

        print("Done!")
    elif choose == 2:
        flag = 0

        for _, _, files in os.walk("Dataset"):
            if not files:
                flag = 1
                break

        if flag:
            print("There is no user in the dataset!")
            continue

        print("Taking picture...")

        cap = cv2.VideoCapture(0)
        _, img = cap.read()

        cap.release()

        print("Recognizing...")

        diction = recognize(img)

        if not diction or float(diction[0][1]) > 0.7:
            print("Failed recognization!")
        else:
            print("You are", diction[0][0])
    elif choose == 3:
        inpu = input("Please input the username:")

        flag = 0

        print("Finding...")

        for _, _, files in os.walk("Dataset"):
            for file in files:
                if file == inpu + ".jpg":
                    print("Deleting...")

                    os.system("del " + os.path.join("Dataset", inpu + ".jpg"))
                    flag = 1

                    print("Done!")

                    break

        if not flag:
            print("User does not exist!")
    else:
        break

print("Thanks for using face recognize system!")
