from flask import Flask, render_template, session, redirect, Response
from functools import wraps
import pymongo
from importlib import import_module
import os

app = Flask(__name__)
app.secret_key = b'4\xc2\xe9G\xd8}\xae\x93\xcd\xdb\xeae\xf5\xda\xc4\xc2'

# database
client = pymongo.MongoClient('localhost',27017)
db = client.user_login_system

#Decorators 
# def loginRequired(f):
#     @wraps(f)
#     def wrap(*args, **kwargs):
#         if 'logged_in' in session:
#             return f(*args,**kwargs)
#         else:
#             return redirect('/')
       
#     return wrap


# Routes
from admin import routes

@app.route("/")
def home():
    return render_template('user/index.html')

@app.route("/login/")
def login():
    return render_template('admin/login.html')

@app.route("/forgot-password/")
def forgetPassword():
    return render_template('admin/forgot-password.html')    

@app.route("/admin/")
def admin():
    return render_template("admin/index.html")
    
@app.route("/client/")
def client():
    return render_template("client/index.html")

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_start')
def video_feed():

    # import camera driver
    if os.environ.get('CAMERA'):
        Camera = import_module('camera_' + os.environ['CAMERA']).Camera
    else:
        from camera import Camera

    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.errorhandler(404)
def notFound(e):
    return render_template('admin/404.html'), 404
