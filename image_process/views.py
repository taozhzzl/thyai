from django.shortcuts import render
from django.http import  HttpResponse, HttpResponseRedirect
import cv2
import numpy as np
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
import detector
import json
import base64
import os
import threading

isrunning = False
lock=threading.Lock()

def base64_to_bytes(input_):
    base = input_.split("base64,")[-1]
    return base64.b64decode(base)

def get_key(dct, value):
    return [k for (k,v) in dct.items() if v == value]
# Create your views here.
@csrf_exempt
def index(request):
    global isrunning
    if request.method == 'GET':
        return render(request,'image_process/form.html')
    if request.method == 'POST':
        # content = 'image/png'
        # response = HttpResponse(content_type=content)
        # print(request.POST)
        if "path" in request.POST:
            filepath = request.POST["path"]
            content = 'image/jpeg'
            response = HttpResponse(content_type=content)
            if os.path.exists(filepath):
                image = cv2.imread(filepath)
            else:
                filepath = filepath + ".jpg"
                folder_path = os.getcwd().replace("/image_process","")
                filepath = os.path.join(folder_path +"/vis_outputs/have",filepath)
                image = cv2.imread(filepath)
            if image is not None:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                final = Image.fromarray(image)
                final.save(response, "jpeg")
                return response
            else:
                return response
        elif "status" in request.POST:
            if isrunning == True:
                return HttpResponse("True",content_type="text/plain")
            else:
                return HttpResponse("False",content_type="text/plain")
        #                             content_type="application/json,charset=utf-8")
        # elif "image" in request.POST:
        #     pic = request.POST["image"]
        #
        #     img_bytes = base64_to_bytes(pic)
        #     image = np.asarray(bytearray(img_bytes), dtype="uint8")
        #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #     # cv2.imwrite("decode.jpg",image)
        #     image, result = detector.detect(image3)
        #     final = Image.fromarray(image)
        #     # final.save(response,"png")
        #     response = HttpResponse(json.dumps(result, ensure_ascii=False),
        #                             content_type="application/json,charset=utf-8")
        #     # response.content = result
        #     return response
        elif "image" in request.POST:
            lock.acquire()
            isrunning = True
            lock.release()
            pic = request.POST["image"]
            #print(pic)
            if len(pic) > 0:
                pic = pic.replace("%2B", "+").replace("%3D", "=")
                img_byte = base64_to_bytes(pic)
                img_np_arr = np.fromstring(img_byte, np.uint8)
                image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
                # cv2.imwrite("decode.jpg",image)
                image, result = detector.detect(image)
                final = Image.fromarray(image)
                # final.save(response,"png")
                response = HttpResponse(json.dumps(result, ensure_ascii=False),
                                        content_type="application/json,charset=utf-8")
                # response.content = result
                lock.acquire()
                isrunning = False
                lock.release()
                return response
            else:
                lock.acquire()
                isrunning = False
                lock.release()
                return HttpResponse(None)
        elif len(request.FILES) > 0:
            image = request.FILES['image'].read()
            image = np.asarray(bytearray(image), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image, result = detector.detect(image)
            final = Image.fromarray(image)
            # final.save(response,"png")
            response = HttpResponse(json.dumps(result,ensure_ascii=False), content_type="application/json,charset=utf-8")
            # response.content = result
            return response
    return HttpResponse(None)
